import torch
from typing import Union
import numpy

from .cropper_uncropper import Uncropper, Cropper
from .unet import UNet
from .encoders_decoders import EncoderConv, DecoderConv, Decoder1by1Linear, DecoderBackground
from .util import convert_to_box_list, invert_convert_to_box_list, compute_average_in_box, compute_ranking
from .util_ml import sample_and_kl_diagonal_normal, compute_entropy_bernoulli, compute_logp_bernoulli, Grid_DPP
from .namedtuple import Inference, NmsOutput, BB, UNEToutput, ZZ, DIST, MetricMiniBatch
from .non_max_suppression import NonMaxSuppression


def optimal_bb_and_bb_regression_penalty(mixing_k1wh: torch.Tensor,
                                         bounding_boxes_k: BB,
                                         pad_size: int,
                                         min_box_size: float,
                                         max_box_size: float) -> (BB, torch.Tensor):
    """ Given the mixing probabilities and the predicted bounding_boxes it computes the optimal bounding_boxes and
        the L2 cost between the predicted bounding_boxes and ideal bounding_boxes.

        Args:
            mixing_k1wh: torch.Tensor of shape :math:`(*, K, 1, W, H)`
            bounding_boxes_k: the bounding boxes predicted by the CNN of type :class:`BB` and shape :math:`(*, K)`
            pad_size: padding around the mask. If :attr:`pad_size` = 0 then the bounding box is body-fitted
            min_box_size: minimum allowed size for the bounding_box
            max_box_size: maximum allowed size for the bounding box

        Returns:
            The optimal bounding boxes in :class:`BB` of shape :math:`(*, K)` and
            the regression_penalty of shape :math:`(*, K)`

        Note:
            The optimal bounding_box  is body-fitted around :math:`mask=(mixing > 0.5)`
            with a padding of size `attr:pad_size` pixels. If the mask is small (or completely empty)
            the optimal bounding_box is a box of the minimum_allowed size.

        Note:
            It works with any number of leading dimensions. Each leading dimension is treated independently.
    """

    with torch.no_grad():

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
        full_x1_k = (torch.argmax(mask_kw * minus_w, dim=-1) - pad_size).clamp(min=0, max=mask_kw.shape[-1]).float()
        full_x3_k = (torch.argmax(mask_kw * plus_w,  dim=-1) + pad_size).clamp(min=0, max=mask_kw.shape[-1]).float()
        full_y1_k = (torch.argmax(mask_kh * minus_h, dim=-1) - pad_size).clamp(min=0, max=mask_kh.shape[-1]).float()
        full_y3_k = (torch.argmax(mask_kh * plus_h,  dim=-1) + pad_size).clamp(min=0, max=mask_kh.shape[-1]).float()

        # Find the coordinates of the empty bounding boxes
        # TODO: empty bounding box should be centered
        empty_x1_k = bounding_boxes_k.bx - 0.5 * min_box_size
        empty_x3_k = bounding_boxes_k.bx + 0.5 * min_box_size
        empty_y1_k = bounding_boxes_k.by - 0.5 * min_box_size
        empty_y3_k = bounding_boxes_k.by + 0.5 * min_box_size

        # Ideal_bb depends whether box is full or empty
        empty_k = (mask_k == 0)
        ideal_x1_k = torch.where(empty_k, empty_x1_k, full_x1_k)
        ideal_x3_k = torch.where(empty_k, empty_x3_k, full_x3_k)
        ideal_y1_k = torch.where(empty_k, empty_y1_k, full_y1_k)
        ideal_y3_k = torch.where(empty_k, empty_y3_k, full_y3_k)

        # Compute the box coordinates (note the clamping of bw and bh)
        ideal_bx_k = 0.5 * (ideal_x3_k + ideal_x1_k)
        ideal_by_k = 0.5 * (ideal_y3_k + ideal_y1_k)
        ideal_bw_k = (ideal_x3_k - ideal_x1_k).clamp(min=min_box_size, max=max_box_size)
        ideal_bh_k = (ideal_y3_k - ideal_y1_k).clamp(min=min_box_size, max=max_box_size)

    # Outside the torch.no_grad() compute the regression cost
    cost_bb_regression = torch.abs(ideal_bx_k - bounding_boxes_k.bx) + \
                         torch.abs(ideal_by_k - bounding_boxes_k.by) + \
                         torch.abs(ideal_bw_k - bounding_boxes_k.bw) + \
                         torch.abs(ideal_bh_k - bounding_boxes_k.bh)

    return BB(bx=ideal_bx_k, by=ideal_by_k, bw=ideal_bw_k, bh=ideal_bh_k), cost_bb_regression


def tgrid_to_bb(t_grid, width_input_image: int, height_input_image: int, min_box_size: float, max_box_size: float):
    """ 
    Convert the output of the zwhere decoder to a list of bounding boxes
    
    Args:
        t_grid: tensor of shape :math:`(B,4,w_grid,h_grid)` with values in (0,1)
        width_input_image: width of the input image
        height_input_image: height of the input image
        min_box_size: minimum allowed size for the bounding boxes
        max_box_size: maximum allowed size for the bounding boxes
        
    Returns:
        A container of type :class:`BB` with the bounding boxes of shape :math:`(B,N)`
        where :math:`N = w_grid * h_grid`. 
    """
    grid_width, grid_height = t_grid.shape[-2:]
    ix_grid = torch.arange(start=0, end=grid_width, dtype=t_grid.dtype,
                           device=t_grid.device).unsqueeze(-1)  # shape: grid_width, 1
    iy_grid = torch.arange(start=0, end=grid_height, dtype=t_grid.dtype,
                           device=t_grid.device).unsqueeze(-2)  # shape: 1, grid_height

    tx_grid, ty_grid, tw_grid, th_grid = torch.split(t_grid, 1, dim=-3)  # shapes: (b,1,grid_width,grid_height)

    bx_grid = width_input_image * (ix_grid + tx_grid) / grid_width    # values in (0,width_input_image)
    by_grid = height_input_image * (iy_grid + ty_grid) / grid_height  # values in (0,height_input_image)
    bw_grid = min_box_size + (max_box_size - min_box_size) * tw_grid  # values in (min_box_size, max_box_size)
    bh_grid = min_box_size + (max_box_size - min_box_size) * th_grid  # values in (min_box_size, max_box_size)
    return BB(bx=convert_to_box_list(bx_grid).squeeze(-1),
              by=convert_to_box_list(by_grid).squeeze(-1),
              bw=convert_to_box_list(bw_grid).squeeze(-1),
              bh=convert_to_box_list(bh_grid).squeeze(-1))


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


class InferenceAndGeneration(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        # variables
        self.bb_regression_strength = config["loss"]["bounding_box_regression_penalty_strength"]
        self.mask_overlap_strength = config["loss"]["mask_overlap_penalty_strength"]
        self.n_mc_samples = config["loss"]["n_mc_samples"]

        self.min_box_size = config["input_image"]["range_object_size"][0]
        self.max_box_size = config["input_image"]["range_object_size"][1]
        self.glimpse_size = config["architecture"]["glimpse_size"]
        self.pad_size_bb = config["loss"]["bounding_box_regression_padding"]

        # modules
        self.grid_dpp = Grid_DPP(length_scale=config["input_image"]["similarity_DPP_l"],
                                 length_scale_min_max=config["input_image"]["similarity_DPP_l_min_max"],
                                 weight=config["input_image"]["similarity_DPP_w"],
                                 weight_min_max=config["input_image"]["similarity_DPP_w_min_max"],
                                 learnable_params=False)

        self.unet: UNet = UNet(n_max_pool=config["architecture"]["n_max_pool_unet"],
                               level_zwhere_and_logit_output=config["architecture"]["level_zwherelogit_unet"],
                               level_background_output=config["architecture"]["n_max_pool_unet"],
                               n_ch_output_features=config["architecture"]["n_ch_output_features"],
                               ch_after_first_two_conv=config["architecture"]["n_ch_after_preprocessing"],
                               dim_zbg=config["architecture"]["dim_zbg"],
                               dim_zwhere=config["architecture"]["dim_zwhere"],
                               dim_logit=1,
                               ch_raw_image=config["architecture"]["n_ch_img"],
                               concatenate_raw_image_to_fmap=True,
                               grad_logit_max=config["loss"]["grad_logit_max"])

        # Encoder-Decoders
        self.decoder_zbg: DecoderBackground = DecoderBackground(dim_z=config["architecture"]["dim_zbg"],
                                                                ch_out=config["architecture"]["n_ch_img"])

        self.decoder_zwhere: Decoder1by1Linear = Decoder1by1Linear(dim_z=config["architecture"]["dim_zwhere"],
                                                                   ch_out=4,
                                                                   groups=4)

        self.decoder_logit: Decoder1by1Linear = Decoder1by1Linear(dim_z=1,
                                                                  ch_out=1)

        self.decoder_zinstance: DecoderConv = DecoderConv(size=config["architecture"]["glimpse_size"],
                                                          dim_z=config["architecture"]["dim_zinstance"],
                                                          ch_out=config["architecture"]["n_ch_img"] + 1)

        self.encoder_zinstance: EncoderConv = EncoderConv(size=config["architecture"]["glimpse_size"],
                                                          ch_in=config["architecture"]["n_ch_output_features"],
                                                          dim_z=config["architecture"]["dim_zinstance"])

        # Geco values
        self.sigma_fg = torch.nn.Parameter(data=torch.tensor(config["loss"]["geco_mse_target"],
                                                             dtype=torch.float)[..., None, None], requires_grad=False)
        self.sigma_bg = torch.nn.Parameter(data=torch.tensor(config["loss"]["geco_mse_target"],
                                                             dtype=torch.float)[..., None, None], requires_grad=False)
        # Geco MSE
        self.geco_target_mse_min = 0.0
        self.geco_target_mse_max = 1.0
        lambda_mse_min = abs(float(config["loss"]["geco_lambda_mse"][0]))
        lambda_mse_max = abs(float(config["loss"]["geco_lambda_mse"][1]))
        self.geco_rawlambda_mse_min = inverse_linear_exp_activation(lambda_mse_min)
        self.geco_rawlambda_mse_max = inverse_linear_exp_activation(lambda_mse_max)
        self.geco_rawlambda_mse = torch.nn.Parameter(data=torch.tensor(inverse_linear_exp_activation(100.0),
                                                                       dtype=torch.float), requires_grad=True)

        # Geco ncell
        self.geco_target_ncell_min = abs(float(config["loss"]["geco_ncell_target"][0]))
        self.geco_target_ncell_max = abs(float(config["loss"]["geco_ncell_target"][1]))
        lambda_ncell_min = abs(float(config["loss"]["geco_lambda_ncell"][0]))
        lambda_ncell_max = abs(float(config["loss"]["geco_lambda_ncell"][1]))
        self.geco_rawlambda_ncell_min = inverse_linear_exp_activation(lambda_ncell_min)
        self.geco_rawlambda_ncell_max = inverse_linear_exp_activation(lambda_ncell_max)
        self.geco_rawlambda_ncell = torch.nn.Parameter(data=torch.tensor(inverse_linear_exp_activation(1.0),
                                                                         dtype=torch.float), requires_grad=True)

        # Geco fg_fraction
        self.geco_target_fgfraction_min = abs(float(config["loss"]["geco_fgfraction_target"][0]))
        self.geco_target_fgfraction_max = abs(float(config["loss"]["geco_fgfraction_target"][1]))
        lambda_fgfraction_min = float(config["loss"]["geco_lambda_fgfraction"][0])
        lambda_fgfraction_max = float(config["loss"]["geco_lambda_fgfraction"][1])
        self.geco_rawlambda_fgfraction_min = inverse_linear_exp_activation(lambda_fgfraction_min)
        self.geco_rawlambda_fgfraction_max = inverse_linear_exp_activation(lambda_fgfraction_max)
        self.geco_rawlambda_fgfraction = torch.nn.Parameter(data=torch.tensor(inverse_linear_exp_activation(1.0),
                                                                              dtype=torch.float), requires_grad=True)

    def forward(self, imgs_bcwh: torch.Tensor,
                generate_synthetic_data: bool,
                prob_corr_factor: float,
                iom_threshold: float,
                k_objects_max: int,
                topk_only: bool,
                noisy_sampling: bool) -> (Inference, MetricMiniBatch):

        # 0. preparation
        batch_size, ch_raw_image, width_raw_image, height_raw_image = imgs_bcwh.shape

        # ---------------------------#
        # 1. UNET
        # ---------------------------#
        unet_output: UNEToutput = self.unet.forward(imgs_bcwh, verbose=False)
        unet_prob_b1wh = torch.sigmoid(unet_output.logit)

        # TODO: Replace the background block with a VQ-VAE
        # Compute the background
        zbg: DIST = sample_and_kl_diagonal_normal(posterior_mu=unet_output.zbg.mu,
                                                  posterior_std=unet_output.zbg.std,
                                                  prior_mu=torch.zeros_like(unet_output.zbg.mu),
                                                  prior_std=torch.ones_like(unet_output.zbg.std),
                                                  noisy_sampling=noisy_sampling,
                                                  sample_from_prior=generate_synthetic_data,
                                                  mc_samples=1 if generate_synthetic_data else self.n_mc_samples,
                                                  squeeze_mc=False)
        zbg_kl = zbg.kl.mean()  # mean over latent dimension and batch

        # TODO: remove sigmoid from here. Background is not limited to be in (0,1)
        out_background_mbcwh = torch.sigmoid(self.decoder_zbg(z=zbg.sample,
                                                              high_resolution=(imgs_bcwh.shape[-2],
                                                                               imgs_bcwh.shape[-1])))
        # Compute the bounding boxes
        zwhere_grid: DIST = sample_and_kl_diagonal_normal(posterior_mu=unet_output.zwhere.mu,
                                                          posterior_std=unet_output.zwhere.std,
                                                          prior_mu=torch.zeros_like(unet_output.zwhere.mu),
                                                          prior_std=torch.ones_like(unet_output.zwhere.std),
                                                          noisy_sampling=noisy_sampling,
                                                          sample_from_prior=generate_synthetic_data,
                                                          mc_samples=1 if generate_synthetic_data else self.n_mc_samples,
                                                          squeeze_mc=False)

        bounding_box_bn: BB = tgrid_to_bb(t_grid=torch.sigmoid(self.decoder_zwhere(zwhere_grid.sample)),
                                          width_input_image=width_raw_image,
                                          height_input_image=height_raw_image,
                                          min_box_size=self.min_box_size,
                                          max_box_size=self.max_box_size)

        # NMS + top-K operation
        with torch.no_grad():

            # Sample c_grid form either prior or posterior
            mc_samples = 1 if generate_synthetic_data else self.n_mc_samples
            squeeze_mc = False
            if generate_synthetic_data:
                # sample from dpp prior
                c_tmp = self.grid_dpp.sample(size=torch.Size([mc_samples] + list(unet_prob_b1wh.shape)))
            else:
                # sample from posterior
                prob_expanded = unet_prob_b1wh.expand([mc_samples] + list(unet_prob_b1wh.shape))
                c_tmp = torch.rand_like(prob_expanded) < prob_expanded if noisy_sampling else (0.5 < prob_expanded)
            c_grid_before_nms = c_tmp.squeeze(dim=0) if squeeze_mc else c_tmp

            # During pretraining I select the boxes according to: score = (1-a) * (c+p) + a * ranking
            if prob_corr_factor == 0:
                prob_from_ranking_grid = torch.zeros_like(unet_prob_b1wh)
            else:
                av_intensity_in_box_bn = compute_average_in_box(delta_imgs=(imgs_bcwh-out_background_mbcwh).abs(),
                                                                bounding_box=bounding_box_bn)
                ranking_bn = compute_ranking(av_intensity_in_box_bn)
                prob_from_ranking_bn = (ranking_bn + 1).float() / (ranking_bn.shape[-1] + 1)
                prob_from_ranking_grid = invert_convert_to_box_list(prob_from_ranking_bn.unsqueeze(dim=-1),
                                                                    original_width=unet_prob_b1wh.shape[-2],
                                                                    original_height=unet_prob_b1wh.shape[-1])

            score_grid = (1-prob_corr_factor) * (c_grid_before_nms + unet_prob_b1wh) + \
                         prob_corr_factor * prob_from_ranking_grid
            combined_topk_only = topk_only or generate_synthetic_data  # if generating from DPP do not do NMS
            nms_output: NmsOutput = NonMaxSuppression.compute_mask_and_index(score=convert_to_box_list(score_grid).squeeze(dim=-1),
                                                                             bounding_box=bounding_box_bn,
                                                                             iom_threshold=iom_threshold,
                                                                             k_objects_max=k_objects_max,
                                                                             topk_only=combined_topk_only)
            k_mask_grid = invert_convert_to_box_list(nms_output.k_mask_n.unsqueeze(dim=-1),
                                                     original_width=score_grid.shape[-2],
                                                     original_height=score_grid.shape[-1])
            c_grid_after_nms = c_grid_before_nms * k_mask_grid

        # Gather all relevant quantities from the selected boxes
        bounding_box_mbk: BB = BB(bx=torch.gather(bounding_box_bn.bx, dim=-1, index=nms_output.indices_k),
                                  by=torch.gather(bounding_box_bn.by, dim=-1, index=nms_output.indices_k),
                                  bw=torch.gather(bounding_box_bn.bw, dim=-1, index=nms_output.indices_k),
                                  bh=torch.gather(bounding_box_bn.bh, dim=-1, index=nms_output.indices_k))
        prob_mbk = torch.gather(convert_to_box_list(unet_prob_b1wh.expand_as(c_grid_before_nms)).squeeze(-1),
                                dim=-1, index=nms_output.indices_k)
        c_detached_mbk = torch.gather(convert_to_box_list(c_grid_after_nms).squeeze(-1),
                                      dim=-1, index=nms_output.indices_k)
        zwhere_kl_mbk = torch.gather(convert_to_box_list(zwhere_grid.kl).mean(dim=-1),
                                     dim=-1, index=nms_output.indices_k)

        # Crop the unet_features according to the selected boxes
        mc_samples, batch_size, k_boxes = bounding_box_mbk.bx.shape
        unet_features_expanded = unet_output.features.unsqueeze(-4).expand(mc_samples, batch_size, k_boxes, -1, -1, -1)
        cropped_feature_map = Cropper.crop(bounding_box=bounding_box_mbk,
                                           big_stuff=unet_features_expanded,
                                           width_small=self.glimpse_size,
                                           height_small=self.glimpse_size)

        # 6. Encode, sample z and decode to big images and big weights
        # Note that here mc_samples is always 1 and the dimension is squeezed
        zinstance_posterior: ZZ = self.encoder_zinstance.forward(cropped_feature_map)
        zinstance_few: DIST = sample_and_kl_diagonal_normal(posterior_mu=zinstance_posterior.mu,
                                                            posterior_std=zinstance_posterior.std,
                                                            prior_mu=torch.zeros_like(zinstance_posterior.mu),
                                                            prior_std=torch.ones_like(zinstance_posterior.std),
                                                            noisy_sampling=noisy_sampling,
                                                            sample_from_prior=generate_synthetic_data,
                                                            mc_samples=1,
                                                            squeeze_mc=True)
        zinstance_kl_mbk = zinstance_few.kl.mean(dim=-1)  # mean over latent dimension

        # Note that the last channel is a mask (i.e. there is a sigmoid non-linearity applied)
        # It is important that the sigmoid is applied before uncropping on a zero-canvas so that mask is zero everywhere
        # except inside the bounding boxes
        # TODO: Apply sigmoid to mask only
        small_stuff = torch.sigmoid(self.decoder_zinstance.forward(zinstance_few.sample))
        big_stuff = Uncropper.uncrop(bounding_box=bounding_box_mbk,
                                     small_stuff=small_stuff,
                                     width_big=width_raw_image,
                                     height_big=height_raw_image)  # shape: n_box, batch, ch, w, h
        out_img_mbkcwh, out_mask_mbk1wh = torch.split(big_stuff,
                                                      split_size_or_sections=(big_stuff.shape[-3] - 1, 1),
                                                      dim=-3)

        # Compute the mixing (using a softmax-like function)
        p_times_mask_mbk1wh = prob_mbk[..., None, None, None] * out_mask_mbk1wh
        mixing_mbk1wh = p_times_mask_mbk1wh / torch.sum(p_times_mask_mbk1wh, dim=-4, keepdim=True).clamp(min=1.0)
        mixing_fg_mb1wh = mixing_mbk1wh.sum(dim=-4)  # sum over k_boxes
        mixing_bg_mb1wh = torch.ones_like(mixing_fg_mb1wh) - mixing_fg_mb1wh

        # Compute mse
        mse_fg_mbkcwh = ((out_img_mbkcwh - imgs_bcwh.unsqueeze(-4)) / self.sigma_fg).pow(2)
        mse_bg_mbcwh = ((out_background_mbcwh - imgs_bcwh) / self.sigma_bg).pow(2)
        mse_av = torch.mean((mixing_mbk1wh * mse_fg_mbkcwh).sum(dim=-4) + mixing_bg_mb1wh * mse_bg_mbcwh)

        # Compute KL divergence between the DPP prior and the posterior:
        # KL(a,DPP) = \sum_c q(c|a) * [ log_q(c|a) - log_p(c|DPP) ]
        #           = - H_q(a) - \sum_c q(c|a) * log_p(c|DPP)
        # The first term is the negative entropy of the Bernoulli distribution. It can be computed analytically and its
        # minimization w.r.t. a lead to high entropy posteriors.
        # The derivative of the second term w.r.t. DPP can be estimated by simple MONTE CARLO samples and makes
        # DPP prior parameters adjust to the seen configurations.
        # The derivative of the second term w.r.t. a can be estimated by simple REINFORCE ESTIMATOR and makes
        # the posterior have more weight on configuration which are likely under the prior
        entropy_b = compute_entropy_bernoulli(logit=unet_output.logit).sum(dim=(-1, -2, -3))
        logp_ber_before_nms_mb = compute_logp_bernoulli(c=c_grid_before_nms.detach(),
                                                        logit=unet_output.logit).sum(dim=(-1, -2, -3))
        with torch.no_grad():
            logp_dpp_before_nms_mb = self.grid_dpp.log_prob(value=c_grid_before_nms.squeeze(-3).detach())
            baseline_b = logp_dpp_before_nms_mb.mean(dim=-2)
            d_mb = (logp_dpp_before_nms_mb - baseline_b)
            distance_from_reinforce_baseline = d_mb.abs().mean()  # for debug
        reinforce_mb = logp_ber_before_nms_mb * d_mb

        if self.grid_dpp.learnable_params:
            logp_dpp_after_nms_mb = self.grid_dpp.log_prob(value=c_grid_after_nms.squeeze(-3).detach())
        else:
            logp_dpp_after_nms_mb = torch.zeros_like(entropy_b)

        logit_kl = - (entropy_b + logp_dpp_after_nms_mb + reinforce_mb).mean()

        # KL is at full strength if the object is certain and lower strenght otherwise.
        # I clamp indicator_mbk to 0.1 to avoid numerical instabilities.
        indicator_mbk = prob_mbk.clamp(min=0.1).detach()
        zwhere_kl = (zwhere_kl_mbk * indicator_mbk).sum(dim=-1).mean()
        zinstance_kl = (zinstance_kl_mbk * indicator_mbk).sum(dim=-1).mean()

        # Loss for non-overlapping masks
        # TODO: I observe that this loss makes the mask shrink and the fg_fraction go down.
        #   The intended behavior is to avoid the overlaps not to drive down the fg_fraction.
        #   Maybe I should write this loss function in terms of mixing.
        mask_overlap_mb1wh = mixing_mbk1wh.sum(dim=-4).pow(2) - mixing_mbk1wh.pow(2).sum(dim=-4)
        loss_mask_overlap = self.mask_overlap_strength * torch.sum(mask_overlap_mb1wh, dim=(-1, -2, -3)).mean()

        # Loss to ideal bounding boxes
        with torch.no_grad():
            area_mask_mbk = mixing_mbk1wh.sum(dim=(-1, -2, -3))
            area_bb_mbk = bounding_box_mbk.bw * bounding_box_mbk.bh
            ratio_mbk = area_mask_mbk / area_bb_mbk
            is_active_mbk = ratio_mbk < 0.8  # this is a trick so that you do not start expanding empty squares
        bb_ideal_mbk, bb_regression_mbk = optimal_bb_and_bb_regression_penalty(mixing_k1wh=mixing_mbk1wh,
                                                                               bounding_boxes_k=bounding_box_mbk,
                                                                               pad_size=self.pad_size_bb,
                                                                               min_box_size=self.min_box_size,
                                                                               max_box_size=self.max_box_size)
        loss_bb_regression = self.bb_regression_strength * (prob_mbk.detach() * is_active_mbk *
                                                            bb_regression_mbk).sum(dim=-1).mean()

        # GECO
        with torch.no_grad():
            # MSE
            self.geco_rawlambda_mse.data.clamp_(min=self.geco_rawlambda_mse_min,
                                                max=self.geco_rawlambda_mse_max)
            lambda_mse = linear_exp_activation(self.geco_rawlambda_mse.data) * \
                         torch.sign(mse_av - self.geco_target_mse_min)
            mse_in_range = (mse_av > self.geco_target_mse_min) & \
                           (mse_av < self.geco_target_mse_max)
            g_mse = 2.0 * mse_in_range - 1.0

            # FG_FRACTION
            self.geco_rawlambda_fgfraction.data.clamp_(min=self.geco_rawlambda_fgfraction_min,
                                                       max=self.geco_rawlambda_fgfraction_max)
            fgfraction_av = mixing_fg_mb1wh.mean()
            lambda_fgfraction = linear_exp_activation(self.geco_rawlambda_fgfraction.data) * \
                                torch.sign(fgfraction_av - self.geco_target_fgfraction_min)
            fgfraction_in_range = (fgfraction_av > self.geco_target_fgfraction_min) & \
                                  (fgfraction_av < self.geco_target_fgfraction_max)
            g_fgfraction = 2.0 * fgfraction_in_range - 1.0

            # NCELL_AV
            self.geco_rawlambda_ncell.data.clamp_(min=self.geco_rawlambda_ncell_min,
                                                  max=self.geco_rawlambda_ncell_max)
            ncell_av = (prob_mbk > 0.5).float().sum(dim=-1).mean()
            lambda_ncell = linear_exp_activation(self.geco_rawlambda_ncell.data) * \
                           torch.sign(ncell_av - self.geco_target_ncell_min)
            ncell_in_range = (ncell_av > self.geco_target_ncell_min) & \
                             (ncell_av < self.geco_target_ncell_max)
            g_ncell = 2.0 * ncell_in_range - 1.0

        # Outside torch.no_grad()
        # TODO: insert loss_geco_ncell back
        loss_geco_mse = self.geco_rawlambda_mse * g_mse + lambda_mse.detach() * mse_av
        loss_geco_fgfraction = self.geco_rawlambda_fgfraction * g_fgfraction + \
                               lambda_fgfraction.detach() * mixing_fg_mb1wh.sum(dim=(-1, -2, -3)).mean()
        loss_geco_ncell = self.geco_rawlambda_ncell * g_ncell + \
                          lambda_ncell.detach() * unet_prob_b1wh.sum(dim=(-1, -2, -3)).mean()

        # Add all the losses together
        # TODO: additional losses should be multiplied by geco_lambda_mse
        loss = loss_mask_overlap + loss_bb_regression + \
               logit_kl + zbg_kl + zwhere_kl + zinstance_kl + \
               loss_geco_mse + loss_geco_fgfraction  # + loss_geco_ncell

        # Other stuff I want to monitor
        with torch.no_grad():
            area_mask_over_area_bb_av = (c_detached_mbk * ratio_mbk).sum() / c_detached_mbk.sum().clamp(min=1.0)
            similarity_l, similarity_w = self.grid_dpp.similiraty_kernel.get_l_w()

        # TODO: Remove a lot of stuff and keep only mixing_bk1wh without squeezing the mc_samples
        inference = Inference(logit_grid=unet_output.logit,
                              prob_from_ranking_grid=prob_from_ranking_grid,
                              background_cwh=out_background_mbcwh,
                              foreground_kcwh=out_img_mbkcwh,
                              sum_c_times_mask_1wh=torch.sum(c_detached_mbk[..., None, None, None] * out_mask_mbk1wh,
                                                             dim=-4),
                              mixing_k1wh=mixing_mbk1wh,
                              sample_c_grid_before_nms=c_grid_before_nms,
                              sample_c_grid_after_nms=c_grid_after_nms,
                              sample_c_k=c_detached_mbk,
                              sample_bb_k=bounding_box_mbk,
                              sample_bb_ideal_k=bb_ideal_mbk)

        metric = MetricMiniBatch(loss=loss,
                                 mse_av=mse_av.detach().item(),
                                 kl_logit=logit_kl.detach().item(),
                                 kl_zinstance=zinstance_kl.detach().item(),
                                 kl_zbg=zbg_kl.detach().item(),
                                 kl_zwhere=zwhere_kl.detach().item(),
                                 cost_mask_overlap_av=loss_mask_overlap.detach().item(),
                                 cost_bb_regression_av=loss_bb_regression.detach().item(),
                                 ncell_av=ncell_av.detach().item(),
                                 prob_av=prob_mbk.sum(dim=-1).mean().detach().item(),
                                 distance_from_reinforce_baseline=distance_from_reinforce_baseline.detach().item(),
                                 fgfraction_av=fgfraction_av.detach().item(),
                                 area_mask_over_area_bb_av=area_mask_over_area_bb_av.detach().item(),
                                 lambda_mse=lambda_mse.detach().item(),
                                 lambda_ncell=lambda_ncell.detach().item(),
                                 lambda_fgfraction=lambda_fgfraction.detach().item(),
                                 similarity_l=similarity_l.detach().item(),
                                 similarity_w=similarity_w.detach().item(),
                                 count_prediction=torch.sum(c_detached_mbk[0], dim=-1).detach().cpu().numpy(),
                                 wrong_examples=None,
                                 accuracy=None,
                                 grad_logit_min=None,
                                 grad_logit_mean=None,
                                 grad_logit_max=None)

        return inference, metric
