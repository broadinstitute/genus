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
    mask_kwh = (mixing_k1wh.squeeze(-3) > 0.25).int()
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
    bx = (tt.ix + tt.tx) * float(rawimage_size[0]) / tgrid_size[0] # values in (0, rawimage_size[0])
    by = (tt.iy + tt.ty) * float(rawimage_size[1]) / tgrid_size[1] # values in (0, rawimage_size[1])
    bw = min_box_size + (max_box_size - min_box_size) * tt.tw  # values in (min_box_size, max_box_size)
    bh = min_box_size + (max_box_size - min_box_size) * tt.th  # values in (min_box_size, max_box_size)
    return BB(bx=bx, by=by, bw=bw, bh=bh)


# def bb_to_tt(bb: BB,
#              rawimage_size: Tuple[int, int],
#              tgrid_size: Tuple[int, int],
#              min_box_size: float,
#              max_box_size: float) -> TT:
#     """ Transformation from :class:`BB` to :class:`TT` """
#     tw = ((bb.bw - min_box_size) / (max_box_size - min_box_size)).clamp(min=0.0, max=1.0)
#     th = ((bb.bh - min_box_size) / (max_box_size - min_box_size)).clamp(min=0.0, max=1.0)
#     ix_plus_tx = (bb.bx * float(tgrid_size[0]) / rawimage_size[0]).clamp(min=0, max=tgrid_size[0])
#     iy_plus_ty = (bb.by * float(tgrid_size[1]) / rawimage_size[1]).clamp(min=0, max=tgrid_size[1])
#
#     ix = ix_plus_tx.clamp(min=0, max=tgrid_size[0]-1).long()
#     iy = iy_plus_ty.clamp(min=0, max=tgrid_size[1]-1).long()
#     tx = ix_plus_tx - ix
#     ty = iy_plus_ty - iy
#     return TT(tx=tx, ty=ty, tw=tw, th=th, ix=ix, iy=iy)


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
            if constraint < 0, parameter will be decreased
            if constraint > 0, parameter will be increased
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

        self.first_iter = True

    @torch.no_grad()
    def set(self, value: float):
        raw_value = inverse_linear_exp_activation(value) if self.linear_exp else value
        self.geco_raw_lambda.data.fill_(raw_value)
        self.geco_raw_lambda.clamp_(min=self.raw_min_value, max=self.raw_max_value)

    @torch.no_grad()
    def get(self):
        self.geco_raw_lambda.data.clamp_(min=self.raw_min_value, max=self.raw_max_value)
        if self.linear_exp:
            transformed_lambda = linear_exp_activation(self.geco_raw_lambda.data)
        else:
            transformed_lambda = self.geco_raw_lambda.data
        return transformed_lambda.detach()

    def forward(self, constraint):
        # This is first since it clamps lambda in the allowed range
        transformed_lambda = self.get()

        # Minimization of the loss function leads to:
        # Constraint satisfied -----> constraint < 0 ---> reduce the parameter
        # Constraint is violated ---> constraint > 0 ---> increase the parameter
        loss_geco = - self.geco_raw_lambda * constraint.detach()

        self.first_iter = False

        return GECO(loss=loss_geco, hyperparam=transformed_lambda)


class InferenceAndGeneration(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        # TODO: Remove shortcut? No!
        self.GMM = config["loss"]["GMM"]
        self.min_prob=1E-2
        self.n_mc_samples = config["loss"]["n_mc_samples"]

        # variables
        self.target_mse_min = config["input_image"]["target_mse_min_max"][0]
        self.target_mse_max = config["input_image"]["target_mse_min_max"][1]
        self.target_fgfraction_min = config["input_image"]["target_fgfraction_min_max"][0]
        self.target_fgfraction_max = config["input_image"]["target_fgfraction_min_max"][1]
        self.bb_regression_strength = config["loss"]["bounding_box_regression_penalty_strength"]
        self.mask_overlap_strength = config["loss"]["mask_overlap_penalty_strength"]

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
        self.sigma_mse = torch.nn.Parameter(data=torch.tensor(config["input_image"]["sigma_mse"],
                                                             dtype=torch.float)[..., None, None], requires_grad=False)

        # Dynamical parameter controlled by GECO
        self.geco_mse_max = GecoParameter(initial_value=config["input_image"]["lambda_mse_min_max"][1],
                                          min_value=config["input_image"]["lambda_mse_min_max"][0],
                                          max_value=config["input_image"]["lambda_mse_min_max"][1],
                                          linear_exp=True)

        # The coupling is (geco_fgfraction_max - geco_fgfraction_min).
        # The initial value gaurantees that the model start from the empty solution
        self.geco_fgfraction_max = GecoParameter(initial_value=config["input_image"]["lambda_fgfraction_min_max"][0],
                                                 min_value=config["input_image"]["lambda_fgfraction_min_max"][0],
                                                 max_value=config["input_image"]["lambda_fgfraction_min_max"][1],
                                                 linear_exp=True)
        self.geco_fgfraction_min = GecoParameter(initial_value=0.0,
                                                 min_value=0.0,
                                                 max_value=config["input_image"]["lambda_fgfraction_min_max"][1],
                                                 linear_exp=True)

        # This params controls the warm-up phase and are annealed linearly from ONE to ZERO
        self.geco_reinforce = GecoParameter(initial_value=1.0, min_value=0.0, max_value=1.0, linear_exp=False)
        self.geco_annealing_factor = GecoParameter(initial_value=1.0, min_value=0.0, max_value=1.0, linear_exp=False)

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
        unet_prob_b1wh = torch.sigmoid(unet_output.logit)

        # 2. Background decoder
        zbg_mu, zbg_std = torch.split(unet_output.zbg,
                                      split_size_or_sections=unet_output.zbg.shape[-3]//2,
                                      dim=-3)
        zbg: DIST = sample_and_kl_diagonal_normal(posterior_mu=zbg_mu,
                                                  posterior_std=F.softplus(zbg_std)+1E-3,
                                                  noisy_sampling=noisy_sampling,
                                                  sample_from_prior=generate_synthetic_data)
        out_background_bcwh = self.decoder_zbg(zbg.value)

        # 3. Bounding-Box decoding
        zwhere_mu, zwhere_std = torch.split(unet_output.zwhere,
                                            split_size_or_sections=unet_output.zwhere.shape[-3]//2,
                                            dim=-3)
        zwhere: DIST = sample_and_kl_diagonal_normal(posterior_mu=zwhere_mu,
                                                     posterior_std=F.softplus(zwhere_std)+1E-3,
                                                     noisy_sampling=noisy_sampling,
                                                     sample_from_prior=generate_synthetic_data)
        t_grid = torch.sigmoid(self.decoder_zwhere(zwhere.value))  # shape B, 4, small_w, small_h
        # TODO; Remove but add clamp
        assert (t_grid >= 0.0).all() and (t_grid <=1.0).all(), "problem min={0}, max={1}".format(t_grid.max().item(),
                                                                                                 t_grid.min().item())
        bounding_box_bn: BB = tgrid_to_bb(t_grid=t_grid,
                                          rawimage_size=imgs_bcwh.shape[-2:],
                                          min_box_size=self.min_box_size,
                                          max_box_size=self.max_box_size,
                                          convert_to_box=True)

        with torch.no_grad():

            # 4. From logit to binarized configuration of having an object at a certain location
            if generate_synthetic_data:
                # sample from dpp prior
                c_grid_before_nms = self.grid_dpp.sample(size=unet_prob_b1wh.size())
            else:
                # sample from posterior
                c_grid_before_nms = (torch.rand_like(unet_prob_b1wh) < unet_prob_b1wh) if noisy_sampling \
                    else (unet_prob_b1wh > 0.5)

            # 5. NMS + top-K operation
            # During pretraining, annealing factor is annealed 1-> 0
            # The boxes are selected according to:
            # score = (1 - annealing) * (unet_c + unet_prob) + annealing * ranking
            annealing_factor = self.geco_annealing_factor.get()
            if annealing_factor == 0:
                # does not need to compute the ranking
                prob_from_ranking_grid = torch.zeros_like(unet_prob_b1wh)
            else:
                # compute the ranking
                av_intensity_in_box_bn = compute_average_in_box(delta_imgs=(imgs_bcwh-out_background_bcwh).abs(),
                                                                bounding_box=bounding_box_bn)
                ranking_bn = compute_ranking(av_intensity_in_box_bn)
                prob_from_ranking_bn = (ranking_bn + 1).float() / (ranking_bn.shape[-1] + 1)
                prob_from_ranking_grid = invert_convert_to_box_list(prob_from_ranking_bn.unsqueeze(dim=-1),
                                                                    original_width=unet_prob_b1wh.shape[-2],
                                                                    original_height=unet_prob_b1wh.shape[-1])

            # compute: score = (1 - annealing) * (unet_c + unet_prob) + annealing * ranking
            score_grid = (1.0 - annealing_factor) * (c_grid_before_nms.float() + unet_prob_b1wh) + \
                         annealing_factor * prob_from_ranking_grid

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
        zwhere_kl_bk = torch.gather(convert_to_box_list(zwhere.kl).mean(dim=-1), dim=-1, index=nms_output.indices_k)
        logit_bk = torch.gather(convert_to_box_list(unet_output.logit).squeeze(-1), dim=-1, index=nms_output.indices_k)
        prob_bk = torch.sigmoid(logit_bk)
        c_detached_bk = torch.gather(convert_to_box_list(c_grid_after_nms).squeeze(-1), dim=-1, index=nms_output.indices_k)

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
                                                        posterior_std=F.softplus(zinstance_std)+1E-3,
                                                        noisy_sampling=noisy_sampling,
                                                        sample_from_prior=generate_synthetic_data)
        zinstance_kl_bk = zinstance.kl.mean(dim=-1)  # average over the latent dimension
        small_imgs_out, small_weights_out = torch.split(self.decoder_zinstance.forward(zinstance.value),
                                                        split_size_or_sections=(ch_raw_image, 1),
                                                        dim=-3)

        # 9. Apply sigmoid non-linearity to obtain small mask and then use STN to paste on a zero-canvas
        # TODO: Remove the squares
        big_stuff = Uncropper.uncrop(bounding_box=bounding_box_bk,
                                     small_stuff=torch.cat((small_imgs_out, torch.sigmoid(small_weights_out), torch.ones_like(small_weights_out)), dim=-3),
                                     width_big=width_raw_image,
                                     height_big=height_raw_image)  # shape: batch, n_box, ch, w, h
        out_img_bkcwh,  out_mask_bk1wh, out_square_bk1wh = torch.split(big_stuff,
                                                                       split_size_or_sections=(ch_raw_image, 1, 1),
                                                                       dim=-3)

        # 10. Compute the mixing (using a softmax-like function).
        # I need two versions. One with probability attached and one detached.

        # This is the mixing that will be used in the reconstruction. It's gradient will push probability up and down
        # Gradients --> go through prob.
        # Values    --> is max(p,self.min_prob)
        p_detached_bk = prob_bk.clamp(min=self.min_prob).detach()
        p_attached_bk = (p_detached_bk - prob_bk).detach() + prob_bk
        p_times_mask_bk1wh = p_attached_bk[..., None, None, None] * out_mask_bk1wh
        mixing_bk1wh = p_times_mask_bk1wh / torch.sum(p_times_mask_bk1wh, dim=-4, keepdim=True).clamp(min=1.0)
        mixing_fg_b1wh = mixing_bk1wh.sum(dim=-4)
        mixing_bg_b1wh = torch.ones_like(mixing_fg_b1wh) - mixing_fg_b1wh

        # I use p_detached here b/c the mask_overlap_cost should change the masks not the probabilities.
        p_detached_times_mask_bk1wh = p_detached_bk[..., None, None, None].detach() * out_mask_bk1wh
        mixing_p_detached_bk1wh = p_detached_times_mask_bk1wh / torch.sum(p_detached_times_mask_bk1wh, dim=-4, keepdim=True).clamp(min=1.0)
        mask_overlap_b1wh = mixing_p_detached_bk1wh.sum(dim=-4).pow(2) - mixing_p_detached_bk1wh.pow(2).sum(dim=-4)
        mask_overlap_cost = self.mask_overlap_strength * mask_overlap_b1wh.mean()

        # 12. Observation model
        if self.GMM:
            rec_img_fg_bcwh = (mixing_bk1wh * out_img_bkcwh).sum(dim=-4)  # sum over boxes
            mse_fg_bcwh = ((rec_img_fg_bcwh - imgs_bcwh) / self.sigma_mse).pow(2)
            mse_bg_bcwh = ((out_background_bcwh - imgs_bcwh) / self.sigma_mse).pow(2)
            mse_av = (mixing_fg_b1wh * mse_fg_bcwh + mixing_bg_b1wh * mse_bg_bcwh).mean()
            delta_msefg_msebg_bcwh = mse_fg_bcwh - mse_bg_bcwh
        else:
            mse_fg_bkcwh = ((out_img_bkcwh - imgs_bcwh.unsqueeze(-4)) / self.sigma_mse).pow(2)
            mse_bg_bcwh = ((out_background_bcwh - imgs_bcwh) / self.sigma_mse).pow(2)
            mse_av = ((mixing_bk1wh * mse_fg_bkcwh).sum(dim=-4) + mixing_bg_b1wh * mse_bg_bcwh).mean()
            delta_msefg_msebg_bcwh = torch.min(mse_fg_bkcwh - mse_bg_bcwh.unsqueeze(-4), dim=-4)[0]

        # TODO: which indicator should I use? p and 1?
        # Since reconstruction gradient are proportional to c_p the KL should also be proportional to c_p
        indicator_bk = p_detached_bk.detach()
        # indicator_bk = torch.ones_like(p_detached_bk).detach()
        area_bk = (bounding_box_bk.bw * bounding_box_bk.bh).detach()
        zinstance_kl_av = (zinstance_kl_bk * area_bk * indicator_bk).sum() / (batch_size * width_raw_image * height_raw_image)
        zbg_kl_av = zbg.kl.mean()

        # 14. Cost for bounding boxes not being properly fitted around the object
        bb_ideal_bk: BB = optimal_bb(mixing_k1wh=mixing_bk1wh,
                                     bounding_boxes_k=bounding_box_bk,
                                     pad_size=self.pad_size_bb,
                                     min_box_size=self.min_box_size,
                                     max_box_size=self.max_box_size)
        bb_regression_bk = torch.abs(bb_ideal_bk.bx - bounding_box_bk.bx) + \
                           torch.abs(bb_ideal_bk.by - bounding_box_bk.by) + \
                           torch.abs(bb_ideal_bk.bw - bounding_box_bk.bw) + \
                           torch.abs(bb_ideal_bk.bh - bounding_box_bk.bh)
        bb_regression_cost = self.bb_regression_strength * (indicator_bk * bb_regression_bk).sum() / batch_size
        zwhere_kl_av = (indicator_bk * zwhere_kl_bk).sum() / batch_size

        # 11. Compute the KL divergences
        # Compute KL divergence between the DPP prior and the posterior:
        # KL(a,DPP) = \sum_c q(c|a) * [ log_q(c|a) - log_p(c|DPP) ] = -H[q] - sum_c q(c|a) log_p(c|DPP)
        # The first term is the negative entropy of the Bernoulli distribution.
        # It can be computed analytically and its minimization w.r.t. "a" leads to high entropy posteriors.
        # The second term can be estimated by REINFORCE ESTIMATOR and makes
        # the posterior have more weight on configuration which are likely under the prior
        entropy_ber = compute_entropy_bernoulli(logit=unet_output.logit).sum(dim=(-1, -2, -3)).mean()

        # Use the Bernoulli distribution at Temperature = 1 to evaluate the KL divergence
        prob_expanded_nb1wh = unet_prob_b1wh.expand([self.n_mc_samples, -1, -1, -1, -1])  # keep: batch,ch,w,h
        c_mcsamples_nb1wh = (torch.rand_like(prob_expanded_nb1wh) < prob_expanded_nb1wh)
        logp_ber_nb = compute_logp_bernoulli(c=c_mcsamples_nb1wh.detach(),
                                             logit=unet_output.logit).sum(dim=(-1, -2, -3))
        with torch.no_grad():
            logp_dpp_nb = self.grid_dpp.log_prob(value=c_mcsamples_nb1wh.squeeze(-3).detach())
            baseline_b = logp_dpp_nb.mean(dim=0)  # average of the different MC samples
            d_ntb = (logp_dpp_nb - baseline_b)
            n_mean = self.grid_dpp.n_mean
            n_stdev = self.grid_dpp.n_stddev
            target_nobj_min = n_mean - n_stdev
            target_nobj_max = n_mean + n_stdev
            # prob_target = n_mean / torch.numel(unet_output.logit[-2:])
            # logit_target = torch.log(prob_target) - torch.log1p(-prob_target)
            # print("logit_target", logit_target)
        reinforce_ber = (logp_ber_nb * d_ntb.detach()).mean()

        # GECO (i.e. make the hyper-parameters dynamical)
        # if constraint < 0, parameter will be decreased
        # if constraint > 0, parameter will be increased

        # Preliminaries
        mse_too_large =  (mse_av > self.target_mse_max)
        mse_too_small =  (mse_av < self.target_mse_min)
        mse_in_range = (mse_av < self.target_mse_max) * (mse_av > self.target_mse_min)

        # Simple logic to decide the value of lmabda_fgfraction = lambda_fgfraction_max - lambda_fgfraction_min
        # Note that this dynamical parameter can be BOTH POSITIVE AND NEGATIVE
        fgfrac_hard_av = (mixing_fg_b1wh > 0.5).float().mean()
        fgfraction_in_range = (fgfrac_hard_av > self.target_fgfraction_min) * (fgfrac_hard_av < self.target_fgfraction_max)
        geco_fgfraction_min: GECO = self.geco_fgfraction_min.forward(constraint=1.0-2.0*(fgfrac_hard_av > self.target_fgfraction_min))
        geco_fgfraction_max: GECO = self.geco_fgfraction_max.forward(constraint=1.0-2.0*(fgfrac_hard_av < self.target_fgfraction_max))
        geco_fgfrac_hyperparam = geco_fgfraction_max.hyperparam - geco_fgfraction_min.hyperparam

        # REINFORCE
        # Reinforce starts at ONE and becomes ZERO at the end of the annealing procedure
        # Reducing geco_reinforce.hyperparam leads to fewer objects
        nobj_grid_av = c_grid_after_nms.sum(dim=(-1,-2,-3)).float().mean()
        nobj_grid_in_range = (nobj_grid_av > target_nobj_min) * (nobj_grid_av < target_nobj_max)
        nobj_grid_not_too_small = (nobj_grid_av > target_nobj_min)
        nobj_grid_too_small = (nobj_grid_av < target_nobj_min)

        decrease_reinforce = nobj_grid_not_too_small * fgfraction_in_range * mse_in_range
        increase_reinforce = nobj_grid_too_small
        geco_reinforce: GECO = self.geco_reinforce.forward(constraint=1.0*increase_reinforce - 1.0*decrease_reinforce)
        logit_kl_av = - entropy_ber + (geco_reinforce.hyperparam - 1.0) * reinforce_ber

        # ANNEALING
        # Annealing starts at ONE and becomes ZERO at the end of the annealing procedure
        # Reducing geco_annealing_factor.hyperparam force the model to learn to localize
        # Note that annealing can never be increased. It decrease monotonically
        decrease_annealing = mse_in_range * nobj_grid_in_range * fgfraction_in_range
        geco_annealing: GECO = self.geco_annealing_factor.forward(constraint=-1.0*decrease_annealing)

        # MSE
        decrease_lambda_mse = (mse_in_range * nobj_grid_in_range * fgfraction_in_range) or mse_too_small
        increase_lambda_mse = mse_too_large
        geco_mse: GECO = self.geco_mse_max.forward(constraint=1.0*increase_lambda_mse-1.0*decrease_lambda_mse)

        # Put all the losses together
        loss_geco = geco_annealing.loss +  geco_mse.loss + geco_reinforce.loss + \
                    geco_fgfraction_min.loss + geco_fgfraction_max.loss

        loss_vae = zinstance_kl_av + zbg_kl_av + logit_kl_av + \
                   geco_mse.hyperparam * (mse_av + bb_regression_cost + zwhere_kl_av +
                                          geco_fgfrac_hyperparam * mixing_p_detached_bk1wh.sum(dim=-4).mean() +
                                          mask_overlap_cost)

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
                              feature_map=unet_output.features.detach(),
                              delta_msefg_msebg=delta_msefg_msebg_bcwh.detach())

        similarity_l, similarity_w = self.grid_dpp.similiraty_kernel.get_l_w()

        metric = MetricMiniBatch(loss=loss_tot,
                                 # monitoring
                                 mse_av=mse_av.detach().item(),
                                 # number of objects
                                 mixing_fg_av=mixing_fg_b1wh.mean().detach().item(),
                                 fgfraction_av=fgfrac_hard_av.detach().item(),
                                 nobj_grid_av=nobj_grid_av.detach().item(),
                                 nobj_av=(prob_bk > 0.5).sum(dim=-1).float().mean().detach().item(),
                                 prob_av=prob_bk.sum(dim=-1).mean().detach().item(),
                                 prob_grid_av=unet_prob_b1wh.sum(dim=(-1,-2,-3)).mean().detach().item(),
                                 # terms in the loss function
                                 cost_mse=(geco_mse.hyperparam * mse_av).detach().item(),
                                 cost_mask_overlap_av=(geco_mse.hyperparam * mask_overlap_cost).detach().item(),
                                 cost_fgfraction=(geco_fgfrac_hyperparam * out_mask_bk1wh.mean()).detach().item(),
                                 #cost_nobj=(geco_nobj.hyperparam * torch.abs(unet_output.logit - logit_target).mean()).detach().item(),
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
                                 lambda_reinforce=geco_reinforce.hyperparam.detach().item(),
                                 entropy_ber=entropy_ber.detach().item(),
                                 reinforce_ber=reinforce_ber.detach().item(),
                                 # count accuracy
                                 count_prediction=(prob_bk > 0.5).int().sum(dim=-1).detach().cpu().numpy(),
                                 wrong_examples=-1 * numpy.ones(1),
                                 accuracy=-1.0)

        return inference, metric


#########      with torch.no_grad():
#########            # Convert bounding_boxes to the t_variable in [0,1]
#########            tt_ideal_bk = bb_to_tt(bb=bb_ideal_bk,
#########                                   rawimage_size=imgs_bcwh.shape[-2:],
#########                                   tgrid_size=t_grid.shape[-2:],
#########                                   min_box_size=self.min_box_size,
#########                                   max_box_size=self.max_box_size)
#########
#########            # I now know the K ideal values of t_variable.
#########            # I fill now a grid with the default and ideal value of t_variable
#########            bb_mask_grid = torch.zeros_like(t_grid)
#########            tgrid_ideal = torch.zeros_like(t_grid)
#########            tgrid_ideal[:2] = 0.5  # i.e. default center of the box is in the middle of cell
#########            tgrid_ideal[-2:] = 0.5  # i.e. default size of bounding boxes is average
#########            # tgrid_ideal[-2:] = 0.0 # i.e. default size of bounding boxes is minimal
#########            # tgrid_ideal[-2:] = 1.0 # i.e. default size of bounding boxes is maximal
#########
#########            # TODO: double check this
#########            assert (tt_ideal_bk.ix >= 0).all() and (tt_ideal_bk.ix < tgrid_ideal.shape[-2]).all()
#########            assert (tt_ideal_bk.iy >= 0).all() and (tt_ideal_bk.iy < tgrid_ideal.shape[-1]).all()
#########            assert torch.isfinite(tt_ideal_bk.tx).all()
#########            assert torch.isfinite(tt_ideal_bk.ty).all()
#########            assert torch.isfinite(tt_ideal_bk.tw).all()
#########            assert torch.isfinite(tt_ideal_bk.th).all()
#########
#########            b_index = torch.arange(tt_ideal_bk.ix.shape[0]).unsqueeze(-1).expand_as(tt_ideal_bk.ix)
#########            bb_mask_grid[b_index, :, tt_ideal_bk.ix, tt_ideal_bk.iy] = 1.0
#########            tgrid_ideal[b_index, 0, tt_ideal_bk.ix, tt_ideal_bk.iy] = tt_ideal_bk.tx
#########            tgrid_ideal[b_index, 1, tt_ideal_bk.ix, tt_ideal_bk.iy] = tt_ideal_bk.ty
#########            tgrid_ideal[b_index, 2, tt_ideal_bk.ix, tt_ideal_bk.iy] = tt_ideal_bk.tw
#########            tgrid_ideal[b_index, 3, tt_ideal_bk.ix, tt_ideal_bk.iy] = tt_ideal_bk.th
#######
######## Outside torch.no_grad() compute the bb_regression_cost
######## TODO: Compute regression for all or only some of the boxes?
######## bb_regression_cost = torch.mean(bb_mask_grid * (t_grid - tgrid_ideal.detach()).pow(2))
######## bb_regression_cost = (t_grid - tgrid_ideal.detach()).pow(2).mean()
