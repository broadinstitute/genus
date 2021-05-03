import torch
from typing import Union, Tuple
import numpy
import torch.nn.functional as F
from .cropper_uncropper import Uncropper, Cropper
from .unet import UNet
from .conv import DecoderConv, EncoderInstance, DecoderInstance
from .util import convert_to_box_list, invert_convert_to_box_list, compute_average_in_box, compute_ranking
from .util_ml import compute_entropy_bernoulli, compute_logp_bernoulli, Grid_DPP, sample_and_kl_diagonal_normal
from .namedtuple import Inference, NmsOutput, BB, UNEToutput, MetricMiniBatch, DIST, TT, GECO
from .non_max_suppression import NonMaxSuppression


@torch.no_grad()
def optimal_bb(mixing_k1wh: torch.Tensor,
               bounding_boxes_k: BB,
               pad_size: int,
               min_box_size: float,
               max_box_size: float) -> BB:
    """ Given the mixing probabilities it computes the optimal bounding_boxes

        Args:
            mixing_k1wh: torch.Tensor of shape :math:`(*, K, 1, W, H)`
            bounding_boxes_k: the bounding boxes predicted by the CNN of type :class:`BB` and shape :math:`(*, K)`
            pad_size: padding around the mask. If :attr:`pad_size` = 0 then the bounding box is body-fitted
            min_box_size: minimum allowed size for the bounding_box
            max_box_size: maximum allowed size for the bounding box

        Returns:
            The optimal bounding boxes in :class:`BB` of shape :math:`(*, K)`

        Note:
            The optimal bounding_box is body-fitted around :math:`mask=(mixing > 0.5)`
            with a padding of size `attr:pad_size` pixels. If the mask is small (or completely empty)
            the optimal bounding_box is a box of the minimum_allowed size.

        Note:
            It works with any number of leading dimensions. Each leading dimension is treated independently.
    """


    # Compute the ideal Bounding boxes
    mask_kwh = (mixing_k1wh.squeeze(-3) > 0.5).int()
    mask_kh = torch.max(mask_kwh, dim=-2)[0]
    mask_kw = torch.max(mask_kwh, dim=-1)[0]
    mask_k = torch.max(mask_kw, dim=-1)[0]  # 0 if empty, 1 if non-empty

    plus_h = torch.arange(start=0, end=mask_kh.shape[-1], step=1,
                          dtype=torch.float, device=mixing_k1wh.device) + 1
    plus_w = torch.arange(start=0, end=mask_kw.shape[-1], step=1,
                          dtype=torch.float, device=mixing_k1wh.device) + 1
    minus_h = plus_h[-1] - plus_h + 1
    minus_w = plus_w[-1] - plus_w + 1

    # Find the coordinates of the full bounding boxes
    full_x1_k = (torch.argmax(mask_kw * minus_w, dim=-1) - pad_size).clamp(min=0.0, max=mask_kw.shape[-1]).float()
    full_x3_k = (torch.argmax(mask_kw * plus_w,  dim=-1) + pad_size).clamp(min=0.0, max=mask_kw.shape[-1]).float()
    full_y1_k = (torch.argmax(mask_kh * minus_h, dim=-1) - pad_size).clamp(min=0.0, max=mask_kh.shape[-1]).float()
    full_y3_k = (torch.argmax(mask_kh * plus_h,  dim=-1) + pad_size).clamp(min=0.0, max=mask_kh.shape[-1]).float()

    # Find the coordinates of the empty bounding boxes
    # TODO: empty bounding box should be centered
    empty_x1_k = (bounding_boxes_k.bx - 0.5 * min_box_size).clamp(min=0.0, max=mask_kw.shape[-1])
    empty_x3_k = (bounding_boxes_k.bx + 0.5 * min_box_size).clamp(min=0.0, max=mask_kw.shape[-1])
    empty_y1_k = (bounding_boxes_k.by - 0.5 * min_box_size).clamp(min=0.0, max=mask_kh.shape[-1])
    empty_y3_k = (bounding_boxes_k.by + 0.5 * min_box_size).clamp(min=0.0, max=mask_kh.shape[-1])

    # Ideal_bb depends whether box is full or empty
    empty_k = (mask_k == 0)
    ideal_x1_k = torch.where(empty_k, empty_x1_k, full_x1_k)
    ideal_x3_k = torch.where(empty_k, empty_x3_k, full_x3_k)
    ideal_y1_k = torch.where(empty_k, empty_y1_k, full_y1_k)
    ideal_y3_k = torch.where(empty_k, empty_y3_k, full_y3_k)

    # From the 4 corners to the center and size of bb
    ideal_bw_k = (ideal_x3_k - ideal_x1_k).clamp(min=min_box_size, max=max_box_size)
    ideal_bh_k = (ideal_y3_k - ideal_y1_k).clamp(min=min_box_size, max=max_box_size)
    ideal_bx_k = 0.5 * (ideal_x3_k + ideal_x1_k)
    ideal_by_k = 0.5 * (ideal_y3_k + ideal_y1_k)

    return BB(bx=ideal_bx_k, by=ideal_by_k, bw=ideal_bw_k, bh=ideal_bh_k)


def tgrid_to_bb(t_grid,
                rawimage_size: Tuple[int, int],
                min_box_size: float,
                max_box_size: float,
                convert_to_box: bool=True) -> BB:
    """
    Convert the output of the zwhere decoder to a list of bounding boxes

    Args:
        t_grid: tensor of shape :math:`(B,4,w_grid,h_grid)` with values in (0,1)
        rawimage_size: width and height of the raw image
        min_box_size: minimum allowed size for the bounding boxes
        max_box_size: maximum allowed size for the bounding boxes
        convert_to_box: boo, if False keep the same shape, it true convert to list of boxes

    Returns:
        A container of type :class:`BB` with the bounding boxes of shape :math:`(B,N)`
        where :math:`N = w_grid * h_grid`.
    """
    tx_grid, ty_grid, tw_grid, th_grid = torch.split(t_grid, split_size_or_sections=1, dim=-3)
    with torch.no_grad():
        grid_width, grid_height = t_grid.shape[-2:]
        ix_grid = torch.arange(start=0, end=grid_width, dtype=t_grid.dtype,
                               device=t_grid.device).unsqueeze(-1).expand_as(tx_grid)
        iy_grid = torch.arange(start=0, end=grid_height, dtype=t_grid.dtype,
                               device=t_grid.device).unsqueeze(-2).expand_as(tx_grid)

    if convert_to_box:
        tt = TT(tx=convert_to_box_list(tx_grid).squeeze(-1),
                ty=convert_to_box_list(ty_grid).squeeze(-1),
                tw=convert_to_box_list(tw_grid).squeeze(-1),
                th=convert_to_box_list(th_grid).squeeze(-1),
                ix=convert_to_box_list(ix_grid).squeeze(-1),
                iy=convert_to_box_list(iy_grid).squeeze(-1))
    else:
        tt = TT(tx=tx_grid,
                ty=ty_grid,
                tw=tw_grid,
                th=th_grid,
                ix=ix_grid,
                iy=iy_grid)

    return tt_to_bb(tt=tt,
                    rawimage_size=rawimage_size,
                    tgrid_size=t_grid.shape[-2:],
                    min_box_size=min_box_size,
                    max_box_size=max_box_size)


def tt_to_bb(tt: TT,
             rawimage_size: Tuple[int, int],
             tgrid_size: Tuple[int, int],
             min_box_size: float,
             max_box_size: float) -> BB:
    """ Transformation from :class:`TT` to :class:`BB` """
    # TODO: Remove this assert
    assert (tt.ix >= 0).all() and (tt.ix < tgrid_size[0]).all()
    assert (tt.iy >= 0).all() and (tt.iy < tgrid_size[1]).all()
    assert (tt.tx >= 0).all() and (tt.tx <= 1.0).all()
    assert (tt.ty >= 0).all() and (tt.ty <= 1.0).all()
    assert (tt.tw >= 0).all() and (tt.tw <= 1.0).all()
    assert (tt.th >= 0).all() and (tt.th <= 1.0).all()

    bx = (tt.ix + tt.tx) * float(rawimage_size[0]) / tgrid_size[0] # values in (0,width_input_image)
    by = (tt.iy + tt.ty) * float(rawimage_size[1]) / tgrid_size[1] # values in (0,height_input_image)
    bw = min_box_size + (max_box_size - min_box_size) * tt.tw  # values in (min_box_size, max_box_size)
    bh = min_box_size + (max_box_size - min_box_size) * tt.th  # values in (min_box_size, max_box_size)
    return BB(bx=bx, by=by, bw=bw, bh=bh)


def bb_to_tt(bb: BB,
             rawimage_size: Tuple[int, int],
             tgrid_size: Tuple[int, int],
             min_box_size: float,
             max_box_size: float) -> TT:
    """ Transformation from :class:`BB` to :class:`TT` """
    tw = ((bb.bw - min_box_size) / (max_box_size - min_box_size)).clamp(min=0.0, max=1.0)
    th = ((bb.bh - min_box_size) / (max_box_size - min_box_size)).clamp(min=0.0, max=1.0)
    ix_plus_tx = (bb.bx * float(tgrid_size[0]) / rawimage_size[0]).clamp(min=0, max=tgrid_size[0])
    iy_plus_ty = (bb.by * float(tgrid_size[1]) / rawimage_size[1]).clamp(min=0, max=tgrid_size[1])

    ix = ix_plus_tx.clamp(min=0, max=tgrid_size[0]-1).long()
    iy = iy_plus_ty.clamp(min=0, max=tgrid_size[1]-1).long()
    tx = ix_plus_tx - ix
    ty = iy_plus_ty - iy
    return TT(tx=tx, ty=ty, tw=tw, th=th, ix=ix, iy=iy)


def linear_exp_activation(x: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """ y = f(x) where f is linear for small x and exponential for large x.
        The function f is continuous with continuous first derivative.
    """
    if isinstance(x, float):
        y = x if x < 1.0 else numpy.exp(x-1.0)
    elif isinstance(x, torch.Tensor):
        y = torch.where(x < 1.0, x, (x-1.0).exp())
    else:
        raise Exception("input type should be either float or torch.Tensor ->", type(x))
    return y


def inverse_linear_exp_activation(y: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """ x = f^(-1)(y) where f is linear for small x and exponential for large x.
        The function f is continuous with continuous first derivative.
    """
    if isinstance(y, float):
        x = y if y < 1.0 else numpy.log(y) + 1.0
    elif isinstance(y, torch.Tensor):
        x = torch.where(y < 1.0, y, y.log() + 1.0)
    else:
        raise Exception("input type should be either float or torch.Tensor ->", type(y))
    return x


class GecoParameter(torch.nn.Module):
    """ Dynamical parameter with value in [min_value, max_value].
        Note:
            if constraint > 0, parameter will be increased
            if constraint < 0, parameter will be decreased
    """
    def __init__(self, initial_value: float,
                 min_value: Union[float, None]=None,
                 max_value: Union[float, None]=None,
                 linear_exp: bool=False):
        super().__init__()

        self.linear_exp = linear_exp
        assert initial_value >= 0
        assert (min_value is None) or (min_value >= 0)
        assert (max_value is None) or (max_value >= 0)
        if min_value is not None and max_value is not None:
            assert max_value > min_value, "max_value {0} should be larger than min_value {1}".format(max_value, min_value)

        if linear_exp:
            self.geco_raw_lambda = torch.nn.Parameter(torch.tensor(inverse_linear_exp_activation(float(initial_value)),
                                                                        dtype=torch.float), requires_grad=True)
            self.raw_min_value = None if min_value is None else inverse_linear_exp_activation(float(min_value))
            self.raw_max_value = None if max_value is None else inverse_linear_exp_activation(float(max_value))
        else:
            self.geco_raw_lambda = torch.nn.Parameter(data=torch.tensor(initial_value, dtype=torch.float),
                                                      requires_grad=True)
            self.raw_min_value = min_value
            self.raw_max_value = max_value

    @torch.no_grad()
    def get(self):
        self.geco_raw_lambda.data.clamp_(min=self.raw_min_value, max=self.raw_max_value)
        if self.linear_exp:
            transformed_lambda = linear_exp_activation(self.geco_raw_lambda.data)
        else:
            transformed_lambda = self.geco_raw_lambda.data
        return transformed_lambda.detach()

    def forward(self, constraint):
        # THis is first since it clamps lambda in the allowed range
        transformed_lambda = self.get()

        # Minimization of the loss function leads to:
        # if constraint > 0, parameter will be increased
        # if constraint < 0, parameter will be decreased
        loss_geco = - self.geco_raw_lambda * constraint.detach()

        return GECO(loss=loss_geco, hyperparam=transformed_lambda)


class InferenceAndGeneration(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        # TODO: Remove these two
        self.PMIN = 1E-3
        self.bb_regression_always_active = config["simulation"]["bb_regression_always_active"]

        # variables
        self.target_fgfraction_min = config["input_image"]["target_fgfraction_min_max"][0]
        self.target_fgfraction_max = config["input_image"]["target_fgfraction_min_max"][1]

        self.min_average_objects_per_patch = config["input_image"]["min_average_objects_per_patch"]
        self.bb_regression_strength = config["loss"]["bounding_box_regression_penalty_strength"]
        self.mask_overlap_strength = config["loss"]["mask_overlap_penalty_strength"]
        self.n_mc_samples = config["loss"]["n_mc_samples"]

        self.min_box_size = config["input_image"]["range_object_size"][0]
        self.max_box_size = config["input_image"]["range_object_size"][1]
        self.glimpse_size = config["architecture"]["glimpse_size"]
        self.pad_size_bb = config["loss"]["bounding_box_regression_padding"]

        # modules
        self.grid_dpp = Grid_DPP(length_scale=config["input_image"]["DPP_length"],
                                 weight=config["input_image"]["DPP_weight"],
                                 learnable_params=False)

        self.unet: UNet = UNet(scale_factor_initial_layer=config["architecture"]["unet_scale_factor_initial_layer"],
                               scale_factor_background=config["architecture"]["unet_scale_factor_background"],
                               scale_factor_boundingboxes=config["architecture"]["unet_scale_factor_boundingboxes"],
                               ch_in=config["input_image"]["ch_in"],
                               ch_out=config["architecture"]["unet_ch_feature_map"],
                               ch_before_first_maxpool=config["architecture"]["unet_ch_before_first_maxpool"],
                               dim_zbg=config["architecture"]["zbg_dim"],
                               dim_zwhere=config["architecture"]["zwhere_dim"],
                               dim_logit=1)

        # Encoder-Decoders
        self.decoder_zbg: DecoderConv = DecoderConv(ch_in=config["architecture"]["zbg_dim"],
                                                    ch_out=config["input_image"]["ch_in"],
                                                    scale_factor=config["architecture"]["unet_scale_factor_background"])

        self.decoder_zwhere: torch.nn.Module = torch.nn.Conv2d(in_channels=config["architecture"]["zwhere_dim"],
                                                               out_channels=4,
                                                               kernel_size=1,
                                                               groups=4)

        self.decoder_zinstance: DecoderInstance = DecoderInstance(glimpse_size=config["architecture"]["glimpse_size"],
                                                                  dim_z=config["architecture"]["zinstance_dim"],
                                                                  ch_out=config["input_image"]["ch_in"] + 1)

        self.encoder_zinstance: EncoderInstance = EncoderInstance(glimpse_size=config["architecture"]["glimpse_size"],
                                                                  ch_in=config["architecture"]["unet_ch_feature_map"],
                                                                  dim_z=config["architecture"]["zinstance_dim"])

        # Observation model
        self.sigma_fg = torch.nn.Parameter(data=torch.tensor(config["input_image"]["target_mse_max"],
                                                             dtype=torch.float)[..., None, None], requires_grad=False)
        self.sigma_bg = torch.nn.Parameter(data=torch.tensor(config["input_image"]["target_mse_max"],
                                                             dtype=torch.float)[..., None, None], requires_grad=False)

        self.geco_fgfraction_min = GecoParameter(initial_value=1.0,
                                                 min_value=0.0,
                                                 max_value=config["loss"]["lambda_fgfraction_max"],
                                                 linear_exp=True)
        self.geco_fgfraction_max = GecoParameter(initial_value=1.0,
                                                 min_value=0.0,
                                                 max_value=config["loss"]["lambda_fgfraction_max"],
                                                 linear_exp=True)
        self.geco_mse_max = GecoParameter(initial_value=1.0,
                                          min_value=config["loss"]["lambda_mse_min_max"][0],
                                          max_value=config["loss"]["lambda_mse_min_max"][1],
                                          linear_exp=True)
        self.geco_annealing_factor = GecoParameter(initial_value=1.0, min_value=0.0, max_value=1.0, linear_exp=False)

        assert config["loss"]["lambda_entropy_max"] > 1.0
        self.geco_entropy_factor = GecoParameter(initial_value=1.0,
                                                 min_value=1.0,
                                                 max_value=config["loss"]["lambda_entropy_max"],
                                                 linear_exp=True)


    def forward(self, imgs_bcwh: torch.Tensor,
                generate_synthetic_data: bool,
                iom_threshold: float,
                k_objects_max: int,
                topk_only: bool,
                noisy_sampling: bool) -> (Inference, MetricMiniBatch):

        # 0. preparation
        batch_size, ch_raw_image, width_raw_image, height_raw_image = imgs_bcwh.shape

        # 1. UNET
        unet_output: UNEToutput = self.unet.forward(imgs_bcwh, verbose=False)

        # 2. Background decoder
        zbg_mu, zbg_std = torch.split(unet_output.zbg,
                                      split_size_or_sections=unet_output.zbg.shape[-3]//2,
                                      dim=-3)
        zbg: DIST = sample_and_kl_diagonal_normal(posterior_mu=zbg_mu,
                                                  posterior_std=F.softplus(zbg_std),
                                                  noisy_sampling=noisy_sampling,
                                                  sample_from_prior=generate_synthetic_data)
        out_background_bcwh = self.decoder_zbg(zbg.value)

        # 3. Bounding-Box decoding
        zwhere_mu, zwhere_std = torch.split(unet_output.zwhere,
                                            split_size_or_sections=unet_output.zwhere.shape[-3]//2,
                                            dim=-3)
        zwhere: DIST = sample_and_kl_diagonal_normal(posterior_mu=zwhere_mu,
                                                     posterior_std=F.softplus(zwhere_std),
                                                     noisy_sampling=noisy_sampling,
                                                     sample_from_prior=generate_synthetic_data)
        t_grid = torch.sigmoid(self.decoder_zwhere(zwhere.value))  # shape B, 4, small_w, small_h
        # TODO; Remove but add clamp
        assert (t_grid >= 0.0).all() and (t_grid <=1.0).all(), "problem min={0}, max={1}".format(t_grid.max().item(),
                                                                                                 t_grid.min().item())
        bounding_box_bn: BB = tgrid_to_bb(t_grid=t_grid,
                                          rawimage_size=imgs_bcwh.shape[-2:],
                                          min_box_size=self.min_box_size,
                                          max_box_size=self.max_box_size)

        with torch.no_grad():

            # 4. From logit to binarized configuration of having an object at a certain location
            unet_prob_b1wh = torch.sigmoid(unet_output.logit)
            if generate_synthetic_data:
                # sample from dpp prior
                c_grid_before_nms_mcsamples = self.grid_dpp.sample(size=unet_prob_b1wh.size()).unsqueeze(dim=0)
            else:
                # sample from posterior
                prob_expanded = unet_prob_b1wh.expand(self.n_mc_samples, -1,-1,-1,-1)
                c_grid_before_nms_mcsamples = (torch.rand_like(prob_expanded) < prob_expanded)
                if not noisy_sampling:
                    c_grid_before_nms_mcsamples[0] = (prob_expanded[0] > 0.5)

            # 5. NMS + top-K operation
            # only one mcsamples is used downstream (note that this is the sample which can be noisy or not)
            c_grid_before_nms = c_grid_before_nms_mcsamples[0]

            annealing_factor = self.geco_annealing_factor.get()
            if annealing_factor == 0:
                prob_from_ranking_grid = torch.zeros_like(unet_prob_b1wh)
            else:
                av_intensity_in_box_bn = compute_average_in_box(delta_imgs=(imgs_bcwh-out_background_bcwh).abs(),
                                                                bounding_box=bounding_box_bn)
                ranking_bn = compute_ranking(av_intensity_in_box_bn)
                prob_from_ranking_bn = (ranking_bn + 1).float() / (ranking_bn.shape[-1] + 1)
                prob_from_ranking_grid = invert_convert_to_box_list(prob_from_ranking_bn.unsqueeze(dim=-1),
                                                                    original_width=unet_prob_b1wh.shape[-2],
                                                                    original_height=unet_prob_b1wh.shape[-1])

            # During pretraining, annealing factor is > 0 and I select the boxes according to:
            # score = (1-a) * (c+p) + a * ranking
            score_grid = (torch.ones_like(annealing_factor) - annealing_factor) * \
                         (c_grid_before_nms.float() + unet_prob_b1wh) + annealing_factor * prob_from_ranking_grid

            combined_topk_only = topk_only or generate_synthetic_data  # if generating from DPP do not do NMS
            nms_output: NmsOutput = NonMaxSuppression.compute_mask_and_index(score=convert_to_box_list(score_grid).squeeze(dim=-1),
                                                                             bounding_box=bounding_box_bn,
                                                                             iom_threshold=iom_threshold,
                                                                             k_objects_max=k_objects_max,
                                                                             topk_only=combined_topk_only)
            nms_mask = invert_convert_to_box_list(nms_output.chosen_mask.unsqueeze(dim=-1),
                                                  original_width=score_grid.shape[-2],
                                                  original_height=score_grid.shape[-1])
            assert nms_mask.shape == c_grid_before_nms.shape
            c_grid_after_nms = c_grid_before_nms * nms_mask

        # 6. Gather the probability and bounding_boxes which survived the NMS+TOP-K operation
        bounding_box_bk: BB = BB(bx=torch.gather(bounding_box_bn.bx, dim=-1, index=nms_output.indices_k),
                                 by=torch.gather(bounding_box_bn.by, dim=-1, index=nms_output.indices_k),
                                 bw=torch.gather(bounding_box_bn.bw, dim=-1, index=nms_output.indices_k),
                                 bh=torch.gather(bounding_box_bn.bh, dim=-1, index=nms_output.indices_k))
        assert unet_prob_b1wh.shape == c_grid_after_nms.shape
        prob_bk = torch.gather(convert_to_box_list(unet_prob_b1wh).squeeze(-1), dim=-1, index=nms_output.indices_k)
        c_detached_bk = torch.gather(convert_to_box_list(c_grid_after_nms).squeeze(-1), dim=-1, index=nms_output.indices_k)
        c_attached_bk = (c_detached_bk.float() - prob_bk).detach() + prob_bk  # straight-through estimator
        zwhere_kl_bk = torch.gather(convert_to_box_list(zwhere.kl).mean(dim=-1), dim=-1, index=nms_output.indices_k)

        # 7. Crop the unet_features according to the selected boxes
        batch_size, k_boxes = bounding_box_bk.bx.shape
        unet_features_expanded = unet_output.features.unsqueeze(-4).expand(batch_size, k_boxes, -1, -1, -1)
        cropped_feature_map = Cropper.crop(bounding_box=bounding_box_bk,
                                           big_stuff=unet_features_expanded,
                                           width_small=self.glimpse_size,
                                           height_small=self.glimpse_size)

        # 8. Encode, Sample, Decode zinstance
        zinstance_tmp = self.encoder_zinstance.forward(cropped_feature_map)
        zinstance_mu, zinstance_std = torch.split(zinstance_tmp, split_size_or_sections=zinstance_tmp.shape[-1]//2, dim=-1)
        zinstance: DIST = sample_and_kl_diagonal_normal(posterior_mu=zinstance_mu,
                                                        posterior_std=F.softplus(zinstance_std),
                                                        noisy_sampling=noisy_sampling,
                                                        sample_from_prior=generate_synthetic_data)
        # print(zinstance.kl.shape) # --> batch, n_boxes, zinstance_dim
        zinstance_kl_bk = zinstance.kl.mean(dim=-1)
        small_imgs_out, small_weights_out = torch.split(self.decoder_zinstance.forward(zinstance.value),
                                                        split_size_or_sections=(ch_raw_image, 1),
                                                        dim=-3)


        # 9. Apply sigmoid non-linearity to obtain small mask and then use STN to paste on a zero-canvas
        out_img_bkcwh = Uncropper.uncrop(bounding_box=bounding_box_bk,
                                         small_stuff=small_imgs_out,
                                         width_big=width_raw_image,
                                         height_big=height_raw_image)
        out_mask_bk1wh = Uncropper.uncrop(bounding_box=bounding_box_bk,
                                          small_stuff=torch.sigmoid(small_weights_out),
                                          width_big=width_raw_image,
                                          height_big=height_raw_image)

        # Make a shortcut so that foreground can always try to learn a little bit
        # TODO: Do I need to make a shortcut so that background and foreground always learn?
        #  No! I compute small_imgs_in just for debug
        small_imgs_in = Cropper.crop(bounding_box=bounding_box_bk,
                                     big_stuff=imgs_bcwh.unsqueeze(-4).expand(batch_size, k_boxes, -1, -1, -1),
                                     width_small=self.glimpse_size,
                                     height_small=self.glimpse_size)

        # 10. Compute the mixing (using a softmax-like function)
        # TODO: In the paper I say that I use c with straight-through estimator
        p_times_mask_bk1wh = prob_bk[..., None, None, None] * out_mask_bk1wh
        mixing_bk1wh = p_times_mask_bk1wh / torch.sum(p_times_mask_bk1wh, dim=-4, keepdim=True).clamp(min=1.0)
        mixing_fg_b1wh = mixing_bk1wh.sum(dim=-4)  # sum over k_boxes
        mixing_bg_b1wh = torch.ones_like(mixing_fg_b1wh) - mixing_fg_b1wh

        # 11. Compute the KL divergences
        # Compute KL divergence between the DPP prior and the posterior:
        # KL(a,DPP) = \sum_c q(c|a) * [ log_q(c|a) - log_p(c|DPP) ] = -H[q] - sum_c q(c|a) log_p(c|DPP)
        # The first term is the negative entropy of the Bernoulli distribution.
        # It can be computed analytically and its minimization w.r.t. "a" leads to high entropy posteriors.
        # The second term can be estimated by REINFORCE ESTIMATOR and makes
        # the posterior have more weight on configuration which are likely under the prior
        entropy_ber = compute_entropy_bernoulli(logit=unet_output.logit).sum(dim=(-1, -2, -3)).mean()
        logp_ber_before_nms_mb = compute_logp_bernoulli(c=c_grid_before_nms_mcsamples.detach(),
                                                        logit=unet_output.logit).sum(dim=(-1, -2, -3))
        with torch.no_grad():
            logp_dpp_before_nms_mb = self.grid_dpp.log_prob(value=c_grid_before_nms_mcsamples.squeeze(-3).detach())
            baseline_b = logp_dpp_before_nms_mb.mean(dim=-2)
            d_mb = (logp_dpp_before_nms_mb - baseline_b)
        reinforce_ber = (logp_ber_before_nms_mb * d_mb.detach()).mean()

        # Compute the KL divergences of the Gaussian Posterior
        # KL is at full strength if the object is certain and lower strength otherwise.
        # I clamp indicator_bk to PMIN to avoid numerical instabilities since
        # all of the instances are used in the shortcut.
        # TODO: Use as indicator c_bk?
        area_bk = (bounding_box_bk.bw * bounding_box_bk.bh).detach()
        indicator_bk = prob_bk.clamp(min=self.PMIN).detach()

        # print(zbg.kl.shape)  # -> batch, ch, w_small, h_small
        zbg_kl_av = zbg.kl.mean()

        # print(zwhere_kl_bk.shape) # --> batch, n_boxes (because I have already done the average over zwhere_dim)
        # print(zinstance_kl_bk.shape) # --> batch, n_boxes (because I have already done the average over zinstance_dim)
        zwhere_kl_av = (zwhere_kl_bk * indicator_bk * area_bk).sum() / (width_raw_image * height_raw_image)
        zinstance_kl_av = (zinstance_kl_bk * indicator_bk * area_bk).sum() / (width_raw_image * height_raw_image)

        # 12. Observation model
        mse_fg_bkcwh = ((out_img_bkcwh - imgs_bcwh.unsqueeze(-4)) / self.sigma_fg).pow(2)
        mse_bg_bcwh = ((out_background_bcwh - imgs_bcwh) / self.sigma_bg).pow(2)
        mse_av = torch.mean((mixing_bk1wh * mse_fg_bkcwh).sum(dim=-4) + mixing_bg_b1wh * mse_bg_bcwh)

        # 13. Other cost functions
        # Cost for non-overlapping masks
        mask_overlap_b1wh = mixing_bk1wh.sum(dim=-4).pow(2) - mixing_bk1wh.pow(2).sum(dim=-4)
        mask_overlap_cost = mask_overlap_b1wh.mean()

        # Cost for ideal bounding boxes
        with torch.no_grad():
            # Compute the ideal bounding boxes from the mixing probabilities
            bb_ideal_bk: BB = optimal_bb(mixing_k1wh=mixing_bk1wh,
                                         bounding_boxes_k=bounding_box_bk,
                                         pad_size=self.pad_size_bb,
                                         min_box_size=self.min_box_size,
                                         max_box_size=self.max_box_size)

            # Convert bounding_boxes to the t_variable in [0,1]
            tt_ideal_bk = bb_to_tt(bb=bb_ideal_bk,
                                   rawimage_size=imgs_bcwh.shape[-2:],
                                   tgrid_size=t_grid.shape[-2:],
                                   min_box_size=self.min_box_size,
                                   max_box_size=self.max_box_size)

            # I now know the K ideal values of t_variable.
            # I fill now a grid with the default and ideal value of t_variable
            bb_mask_grid = torch.zeros_like(t_grid)
            tgrid_ideal = torch.zeros_like(t_grid)
            tgrid_ideal[:2] = 0.5  # i.e. default center of the box is in the middle of cell
            tgrid_ideal[-2:] = 0.5  # i.e. default size of bounding boxes is average
            # tgrid_ideal[-2:] = 0.0 # i.e. default size of bounding boxes is minimal
            # tgrid_ideal[-2:] = 1.0 # i.e. default size of bounding boxes is maximal

            # TODO: double check this
            assert (tt_ideal_bk.ix >= 0).all() and (tt_ideal_bk.ix < tgrid_ideal.shape[-2]).all()
            assert (tt_ideal_bk.iy >= 0).all() and (tt_ideal_bk.iy < tgrid_ideal.shape[-1]).all()
            assert torch.isfinite(tt_ideal_bk.tx).all()
            assert torch.isfinite(tt_ideal_bk.ty).all()
            assert torch.isfinite(tt_ideal_bk.tw).all()
            assert torch.isfinite(tt_ideal_bk.th).all()

            b_index = torch.arange(tt_ideal_bk.ix.shape[0]).unsqueeze(-1).expand_as(tt_ideal_bk.ix)
            bb_mask_grid[b_index, :, tt_ideal_bk.ix, tt_ideal_bk.iy] = 1.0
            tgrid_ideal[b_index, 0, tt_ideal_bk.ix, tt_ideal_bk.iy] = tt_ideal_bk.tx
            tgrid_ideal[b_index, 1, tt_ideal_bk.ix, tt_ideal_bk.iy] = tt_ideal_bk.ty
            tgrid_ideal[b_index, 2, tt_ideal_bk.ix, tt_ideal_bk.iy] = tt_ideal_bk.tw
            tgrid_ideal[b_index, 3, tt_ideal_bk.ix, tt_ideal_bk.iy] = tt_ideal_bk.th

        # Outside torch.no_grad() compute the bb_regression_cost
        # TODO: Compute regression for all or only some of the boxes?
        # bb_regression_cost = torch.mean(bb_mask_grid * (t_grid - tgrid_ideal.detach()).pow(2))
        bb_regression_cost = (t_grid - tgrid_ideal.detach()).pow(2).mean()

        # GECO (i.e. make the hyper-parameters dynamical)
        # if constraint > 0, parameter will be increased
        # if constraint < 0, parameter will be decreased

        geco_mse: GECO = self.geco_mse_max.forward(constraint=2*(mse_av > 1.0).float()-1.0)
        geco_annealing: GECO = self.geco_annealing_factor.forward(constraint=2*(mse_av > 5.0).float()-1.0)

        ncell_lenient_av = (prob_bk > 0.3).sum(dim=-1).float().mean()
        ncell_av = (prob_bk > 0.49).sum(dim=-1).float().mean()
        geco_entropy: GECO = self.geco_entropy_factor.forward(constraint=2*(ncell_av <
                                                                            self.min_average_objects_per_patch).float()-1)

        fgfrac_av = (mixing_fg_b1wh > 0.5).float().mean()
        geco_fgfraction_min: GECO = self.geco_fgfraction_min.forward(constraint=2.0*(fgfrac_av < self.target_fgfraction_min).float() - 1.0)
        geco_fgfraction_max: GECO = self.geco_fgfraction_max.forward(constraint=2.0*(fgfrac_av > self.target_fgfraction_max).float() - 1.0)

        # Put all the losses together
        loss_geco = geco_annealing.loss + geco_entropy.loss + geco_mse.loss + \
                    geco_fgfraction_min.loss + geco_fgfraction_max.loss

        logit_kl_av = - geco_entropy.hyperparam * entropy_ber - reinforce_ber
        geco_fgfrac_hyperparam =  geco_fgfraction_max.hyperparam - geco_fgfraction_min.hyperparam

        loss_vae = geco_mse.hyperparam * (mse_av + mask_overlap_cost) + \
                   geco_fgfrac_hyperparam * out_mask_bk1wh.mean() + \
                   zinstance_kl_av + zbg_kl_av + logit_kl_av + \
                   bb_regression_cost + zwhere_kl_av
        loss_tot = loss_vae + loss_geco

        inference = Inference(logit_grid=unet_output.logit.detach(),
                              prob_from_ranking_grid=prob_from_ranking_grid.detach(),
                              background_cwh=out_background_bcwh.detach(),
                              foreground_kcwh=out_img_bkcwh.detach(),
                              mask_overlap_1wh=mask_overlap_b1wh.detach(),
                              mixing_k1wh=mixing_bk1wh.detach(),
                              sample_c_grid_before_nms=c_grid_before_nms.detach(),
                              sample_c_grid_after_nms=c_grid_after_nms.detach(),
                              sample_prob_k=prob_bk.detach(),
                              sample_bb_k=bounding_box_bk,
                              sample_bb_ideal_k=bb_ideal_bk,
                              small_imgs_in=small_imgs_in.detach(),
                              small_imgs_out=small_imgs_out.detach(),
                              feature_map=unet_output.features.detach())

        similarity_l, similarity_w = self.grid_dpp.similiraty_kernel.get_l_w()

        metric = MetricMiniBatch(loss=loss_tot,
                                 # monitoring
                                 mse_av=mse_av.detach().item(),
                                 fgfraction_av=fgfrac_av.detach().item(),
                                 ncell_av=ncell_av.detach().item(),
                                 ncell_lenient_av=ncell_lenient_av.detach().item(),
                                 prob_av=prob_bk.sum(dim=-1).mean().detach().item(),
                                 # terms in the loss function
                                 cost_mse=(geco_mse.hyperparam * mse_av).detach().item(),
                                 cost_mask_overlap_av=(geco_mse.hyperparam * mask_overlap_cost).detach().item(),
                                 cost_fgfraction=(geco_fgfrac_hyperparam * out_mask_bk1wh.mean()).detach().item(),
                                 cost_bb_regression_av=bb_regression_cost.detach().item(),
                                 kl_zinstance=zinstance_kl_av.detach().item(),
                                 kl_zbg=zbg_kl_av.detach().item(),
                                 kl_zwhere=zwhere_kl_av.detach().item(),
                                 kl_logit=logit_kl_av.detach().item(),
                                 # debug
                                 similarity_l=similarity_l.detach().item(),
                                 similarity_w=similarity_w.detach().item(),
                                 annealing_factor=geco_annealing.hyperparam.detach().item(),
                                 lambda_mse=geco_mse.hyperparam.detach().item(),
                                 lambda_fgfraction=geco_fgfrac_hyperparam.detach().item(),
                                 lambda_fgfraction_max=geco_fgfraction_max.hyperparam.detach().item(),
                                 lambda_fgfraction_min=geco_fgfraction_min.hyperparam.detach().item(),
                                 lambda_entropy=geco_entropy.hyperparam.detach().item(),
                                 entropy_ber=entropy_ber.detach().item(),
                                 reinforce_ber=reinforce_ber.detach().item(),
                                 # count accuracy
                                 count_prediction=(prob_bk > 0.5).int().sum(dim=-1).detach().cpu().numpy(),
                                 wrong_examples=-1 * numpy.ones(1),
                                 accuracy=-1.0)

        return inference, metric
