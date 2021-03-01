import torch
import torch.nn.functional as F
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

        # Find the coordinates of the bounding box
        ideal_x1_k = (torch.argmax(mask_kw * minus_w, dim=-1) - pad_size).clamp(min=0, max=mask_kw.shape[-1]).float()
        ideal_x3_k = (torch.argmax(mask_kw * plus_w,  dim=-1) + pad_size).clamp(min=0, max=mask_kw.shape[-1]).float()
        ideal_y1_k = (torch.argmax(mask_kh * minus_h, dim=-1) - pad_size).clamp(min=0, max=mask_kh.shape[-1]).float()
        ideal_y3_k = (torch.argmax(mask_kh * plus_h,  dim=-1) + pad_size).clamp(min=0, max=mask_kh.shape[-1]).float()

        # If the box is empty, do a special treatment, i.e. make them the smallest possible size
        empty_k = (mask_k == 0)
        ideal_x1_k[empty_k] = bounding_boxes_k.bx[empty_k] - 0.5 * min_box_size
        ideal_x3_k[empty_k] = bounding_boxes_k.bx[empty_k] + 0.5 * min_box_size
        ideal_y1_k[empty_k] = bounding_boxes_k.by[empty_k] - 0.5 * min_box_size
        ideal_y3_k[empty_k] = bounding_boxes_k.by[empty_k] + 0.5 * min_box_size

        # Compute the box coordinates (note the clamping of bw and bh)
        ideal_bx_k = 0.5 * (ideal_x3_k + ideal_x1_k)
        ideal_by_k = 0.5 * (ideal_y3_k + ideal_y1_k)
        ideal_bw_k = (ideal_x3_k - ideal_x1_k).clamp(min=min_box_size, max=max_box_size)
        ideal_bh_k = (ideal_y3_k - ideal_y1_k).clamp(min=min_box_size, max=max_box_size)

        # print("DEBUG min, max ->", min_box_size, max_box_size)
        # print("DEBUG input ->", bounding_boxes_kb.bx[0, 0], bounding_boxes_kb.by[0, 0],
        #       bounding_boxes_kb.bw[0, 0], bounding_boxes_kb.bh[0, 0], empty_kb[0, 0])
        # print("DEBUG ideal ->", ideal_bx_kb[0, 0], ideal_by_kb[0, 0],
        #       ideal_bw_kb[0, 0], ideal_bh_kb[0, 0], empty_kb[0, 0])

    # Outside the torch.no_grad() compute the regression cost
    cost_bb_regression = torch.abs(ideal_bx_k - bounding_boxes_k.bx) + \
                         torch.abs(ideal_by_k - bounding_boxes_k.by) + \
                         torch.abs(ideal_bw_k - bounding_boxes_k.bw) + \
                         torch.abs(ideal_bh_k - bounding_boxes_k.bh)

    return BB(bx=ideal_bx_k, by=ideal_by_k, bw=ideal_bw_k, bh=ideal_bh_k), cost_bb_regression


def tgrid_to_bb(t_grid, width_input_image: int, height_input_image: int, min_box_size: float, max_box_size: float):
    """ 
    Convert the output of the zwhere decoder to a list of boundinb boxes 
    
    Args:
        t_grid: tensor of shape :math:`(B,4,w_grid,h_grid)` with values in (0,1)
        width_input_image: width of the input image
        height_input_image: height of the input image
        min_box_size: minimum allowed size for the bounding boxes
        max_box_size: maximum allowed size for the bounding boxes
        
    Returns:
        A container of type :class:`BB` with the bounding boxes of shape :math:`(N,B)` 
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
                                 weight_min_max=config["input_image"]["similarity_DPP_w_min_max"])

        self.unet: UNet = UNet(n_max_pool=config["architecture"]["n_max_pool_unet"],
                               level_zwhere_and_logit_output=config["architecture"]["level_zwherelogit_unet"],
                               level_background_output=config["architecture"]["n_max_pool_unet"],
                               n_ch_output_features=config["architecture"]["n_ch_output_features"],
                               ch_after_first_two_conv=config["architecture"]["n_ch_after_preprocessing"],
                               dim_zbg=config["architecture"]["dim_zbg"],
                               dim_zwhere=config["architecture"]["dim_zwhere"],
                               dim_logit=1,
                               ch_raw_image=config["architecture"]["n_ch_img"],
                               concatenate_raw_image_to_fmap=True)

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

        self.geco_target_mse_min = 0.0
        self.geco_target_mse_max = 1.0
        self.geco_target_ncell_min = config["loss"]["geco_ncell_target"][0]
        self.geco_target_ncell_max = config["loss"]["geco_ncell_target"][1]
        self.geco_target_fgfraction_min = config["loss"]["geco_fgfraction_target"][0]
        self.geco_target_fgfraction_max = config["loss"]["geco_fgfraction_target"][1]

        self.geco_loglambda_mse_min = numpy.log(config["loss"]["geco_lambda_mse"][0])
        self.geco_loglambda_mse_max = numpy.log(config["loss"]["geco_lambda_mse"][1])
        self.geco_loglambda_fgfraction_min = numpy.log(config["loss"]["geco_lambda_fgfraction"][0])
        self.geco_loglambda_fgfraction_max = numpy.log(config["loss"]["geco_lambda_fgfraction"][1])
        self.geco_loglambda_ncell_min = numpy.log(config["loss"]["geco_lambda_ncell"][0])
        self.geco_loglambda_ncell_max = numpy.log(config["loss"]["geco_lambda_ncell"][1])

        self.geco_loglambda_fgfraction = torch.nn.Parameter(data=torch.tensor(0,  # self.geco_target_fgfraction_min,
                                                                              dtype=torch.float),
                                                            requires_grad=True)
        self.geco_loglambda_ncell = torch.nn.Parameter(data=torch.tensor(0,  # self.geco_loglambda_ncell_max,
                                                                         dtype=torch.float),
                                                       requires_grad=True)
        self.geco_loglambda_mse = torch.nn.Parameter(data=torch.tensor(self.geco_loglambda_mse_max,
                                                                       dtype=torch.float), requires_grad=True)

    def forward(self, imgs_bcwh: torch.Tensor,
                generate_synthetic_data: bool,
                prob_corr_factor: float,
                iom_threshold: float,
                k_objects_max: int,
                topk_only: bool,
                noisy_sampling: bool) -> (Inference, MetricMiniBatch):

        # compute the inference

        # 0. preparation
        batch_size, ch_raw_image, width_raw_image, height_raw_image = imgs_bcwh.shape

        # ---------------------------#
        # 1. UNET
        # ---------------------------#
        unet_output: UNEToutput = self.unet.forward(imgs_bcwh, verbose=False)
        unet_prob_b1wh = torch.sigmoid(unet_output.logit)
        # unet_output.logit.register_hook(lambda grad: print("grad before clipping:",
        #                                                   torch.min(grad), torch.mean(grad), torch.max(grad)))
        # unet_output.logit.register_hook(lambda grad: grad.clamp(min=-10, max=10))
        # unet_output.logit.register_hook(lambda grad: print("grad after clipping:",
        #                                                   torch.min(grad), torch.mean(grad), torch.max(grad)))

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

            # Do non-max-suppression
            score_grid = c_grid_before_nms + unet_prob_b1wh
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

        # Compute KL divergence between the DPP prior and the posterior:
        # KL(a,DPP) = \sum_c q(c|a) * [ log_q(c|a) - log_p(c|DPP) ]
        #           = - H_q(a) - \sum_c q(c|a) * log_p(c|DPP)
        # The first term is the negative entropy of the Bernoulli distribution. It can be computed analytically and its
        # minimization w.r.t. a lead to high entropy posteriors.
        # The derivative of the second term w.r.t. DPP can be estimated by simple MONTE CARLO samples and makes
        # DPP prior parameters adjust to the seen configurations.
        # The derivative of the second term w.r.t. a can be estimated by simple REINFORCE ESTIMATOR and makes
        # the posterior have more weight on configuration which are likely under the prior
        #
        # I am splitting the KL_logit into two terms, one will be always be active
        # the other will be turned on later during training
        entropy = compute_entropy_bernoulli(logit=unet_output.logit).sum(dim=(-1, -2, -3)).mean()
        logp_dpp_after_nms = self.grid_dpp.log_prob(value=c_grid_after_nms.squeeze(-3).detach()).mean()
        logit_kl_base = - entropy - logp_dpp_after_nms

        if prob_corr_factor < 1.0:
            logp_dpp_before_nms_mb = self.grid_dpp.log_prob(value=c_grid_before_nms.squeeze(-3).detach())
            logp_ber_before_nms_mb = compute_logp_bernoulli(c=c_grid_before_nms.detach(),
                                                            logit=unet_output.logit).sum(dim=(-1, -2, -3))
            baseline_b = logp_dpp_before_nms_mb.mean(dim=-2)
            d = (logp_dpp_before_nms_mb - baseline_b).detach()
            std_b = d.pow(2).mean(dim=-2).sqrt()
            distance_from_reinforce_baseline = d.abs().mean()
            reinforce_mb = logp_ber_before_nms_mb * torch.sign(logp_dpp_before_nms_mb - baseline_b).detach()
            logit_kl_additional = reinforce_mb.mean()
        else:
            distance_from_reinforce_baseline = torch.zeros_like(logit_kl_base)
            logit_kl_additional = torch.zeros_like(logit_kl_base)

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
        small_stuff = torch.sigmoid(self.decoder_zinstance.forward(zinstance_few.sample))
        big_stuff = Uncropper.uncrop(bounding_box=bounding_box_mbk,
                                     small_stuff=small_stuff,
                                     width_big=width_raw_image,
                                     height_big=height_raw_image)  # shape: n_box, batch, ch, w, h
        out_img_mbkcwh, out_mask_mbk1wh = torch.split(big_stuff,
                                                      split_size_or_sections=(big_stuff.shape[-3] - 1, 1),
                                                      dim=-3)

        # Compute the mixing
        # TODO: think hard about this part with Mehrtash
        #   This is a sort of softmax function......
        #   There are many ways to make c differentiable
        #   Should I detach the denominator?
        #   Should I introduce z_depth?
        #   c_smooth_mbk = prob_mbk + (c_detached_mbk.float() - prob_mbk).detach() THIS IS WRONG BECAUSE IT CAN BE EXACTLY ZERO
        #   c_smooth_mbk = prob_mbk - prob_mbk.detach() + torch.max(prob_mbk, c_detached_mbk.float()).detach()
        #   Should I detach the denominator? SHould the denominator be multiplied by c_detahced?
        # Ideally I would like to multiply by c. However if c=0 I can not learn anything and moreover
        # and kl_zwhere and kl_zinstnace diverge because they are not compensated by anything.
        # In the "GOLDEN CODE" in the spacetx I used: c+p-p.detached
        p_times_mask_mbk1wh = prob_mbk[..., None, None, None] * out_mask_mbk1wh
        mixing_mbk1wh = p_times_mask_mbk1wh / torch.sum(p_times_mask_mbk1wh, dim=-4, keepdim=True).clamp(min=1.0)
        mixing_fg_mb1wh = mixing_mbk1wh.sum(dim=-4)  # sum over k_boxes
        mixing_bg_mb1wh = torch.ones_like(mixing_fg_mb1wh) - mixing_fg_mb1wh
        fgfraction_av = mixing_fg_mb1wh.mean()

        # Compute mse
        mse_fg_mbkcwh = ((out_img_mbkcwh - imgs_bcwh.unsqueeze(-4)) / self.sigma_fg).pow(2)
        mse_bg_mbcwh = ((out_background_mbcwh - imgs_bcwh) / self.sigma_bg).pow(2)
        mse_av = torch.mean((mixing_mbk1wh * mse_fg_mbkcwh).sum(dim=-4) + mixing_bg_mb1wh * mse_bg_mbcwh)

        # GECO
        with torch.no_grad():
            # Clamp the log_lambda into the allowed regime
            self.geco_loglambda_mse.data.clamp_(min=self.geco_loglambda_mse_min,
                                                max=self.geco_loglambda_mse_max)
            self.geco_loglambda_ncell.data.clamp_(min=self.geco_loglambda_ncell_min,
                                                  max=self.geco_loglambda_ncell_max)
            self.geco_loglambda_fgfraction.data.clamp_(min=self.geco_loglambda_fgfraction_min,
                                                       max=self.geco_loglambda_fgfraction_max)
            # From log_lambda to lambda
            lambda_mse = self.geco_loglambda_mse.data.exp() * \
                         torch.sign(mse_av - self.geco_target_mse_min)

            ncell_av = c_detached_mbk.sum(dim=-1).float().mean()
            lambda_ncell = self.geco_loglambda_ncell.data.exp() * \
                           torch.sign(ncell_av - self.geco_target_ncell_min)

            lambda_fgfraction = self.geco_loglambda_fgfraction.data.exp() * \
                                torch.sign(fgfraction_av - self.geco_target_fgfraction_min)

            # Is the value in range?
            mse_in_range = (mse_av > self.geco_target_mse_min) & \
                           (mse_av < self.geco_target_mse_max)
            ncell_in_range = (ncell_av > self.geco_target_ncell_min) & \
                             (ncell_av < self.geco_target_ncell_max)
            fgfraction_in_range = (fgfraction_av > self.geco_target_fgfraction_min) & \
                                  (fgfraction_av < self.geco_target_fgfraction_max)
        # Outside torch.no_grad()
        loss_geco_mse = self.geco_loglambda_mse * (1.0 * mse_in_range - 1.0 * ~mse_in_range)
        loss_geco_fgfraction = self.geco_loglambda_fgfraction * (1.0 * fgfraction_in_range - 1.0 * ~fgfraction_in_range)
        loss_geco_ncell = self.geco_loglambda_ncell * (1.0 * ncell_in_range - 1.0 * ~ncell_in_range)

        # Loss for non-overlapping masks
        # TODO: I observe that this loss makes the mask shinkr and the fg_fraction go down.
        #   The intended behavior is to avoid the overlaps not to drive down the fg_fraction.
        #   Maybe I should write this loss function in terms of mixing.
        p_detached_times_mask_mbk1wh = prob_mbk[..., None, None, None].detach() * out_mask_mbk1wh
        mask_overlap_mb1wh = p_detached_times_mask_mbk1wh.sum(dim=-4).pow(2) - \
                             p_detached_times_mask_mbk1wh.pow(2).sum(dim=-4)
        loss_mask_overlap = self.mask_overlap_strength * torch.sum(mask_overlap_mb1wh, dim=(-1, -2, -3)).mean()

        # Loss to ideal bounding boxes
        with torch.no_grad():
            area_mask_mbk = mixing_mbk1wh.sum(dim=(-1, -2, -3))
            area_bb_mbk = bounding_box_mbk.bw * bounding_box_mbk.bh
            ratio_mbk = area_mask_mbk / area_bb_mbk
            is_active_mbk = ratio_mbk < 0.8  # this is a trick so that you do not start expanding empty squares
        # outside torch.no_grad()
        bb_ideal_mbk, bb_regression_mbk = optimal_bb_and_bb_regression_penalty(mixing_k1wh=mixing_mbk1wh,
                                                                               bounding_boxes_k=bounding_box_mbk,
                                                                               pad_size=self.pad_size_bb,
                                                                               min_box_size=self.min_box_size,
                                                                               max_box_size=self.max_box_size)
        loss_bb_regression = self.bb_regression_strength * (prob_mbk.detach() * is_active_mbk *
                                                            bb_regression_mbk).sum(dim=-1).mean()

        # KL should act at full strength on full boxes.
        # However, multiplying by c is very dangerous b/c if a box with c=0 receives any gradient (is this possible?)
        # then zwhere and zinstance will become unstable (b/c they are not regularized by the KL term)
        indicator_mbk = torch.max(prob_mbk, c_detached_mbk.float()).detach()
        zwhere_kl = (zwhere_kl_mbk * indicator_mbk).sum(dim=-1).mean()
        zinstance_kl = (zinstance_kl_mbk * indicator_mbk).sum(dim=-1).mean()

        # TODO: Decide what is the best thing to couple to lambda_ncell
        # coupled_to_ncell = unet_output.logit.clamp(min=-10).sum(dim=(-1, -2, -3)).mean() if lambda_ncell > 0 else \
        #    unet_output.logit.clamp(max=10).sum(dim=(-1, -2, -3)).mean()

        loss_base = logit_kl_base + zbg_kl + zwhere_kl + zinstance_kl + \
                    lambda_mse.detach() * mse_av + loss_geco_mse - loss_geco_mse.detach()

        loss_additional = loss_mask_overlap + loss_bb_regression + logit_kl_additional
                          # \
                          # lambda_ncell.detach() * unet_prob_b1wh.sum(dim=(-1, -2, -3)).mean()  # + \
                          # loss_geco_ncell - loss_geco_ncell.detach()
                          # +
                          # lambda_fgfraction.detach() * (prob_mbk[..., None, None, None].detach() *
                          #                               out_mask_mbk1wh).sum(dim=(-1, -2, -3, -4)).mean() + \
                          # loss_geco_fgfraction - loss_geco_fgfraction.detach() + logit_kl_additional \
                          # + lambda_ncell.detach() * com.sum(dim=(-1, -2, -3))
                          #

        loss = loss_base + (1.0 - prob_corr_factor) * loss_additional

        # Other stuff I want to monitor
        with torch.no_grad():
            area_mask_over_area_bb_av = (c_detached_mbk * ratio_mbk).sum() / c_detached_mbk.sum().clamp(min=1.0)
            similarity_l, similarity_w = self.grid_dpp.similiraty_kernel.get_l_w()
            # print(similarity_w.detach().item(), similarity_l.detach().item())

        # TODO: Remove a lot of stuff and keep only mixing_bk1wh without squeezing the mc_samples
        inference = Inference(logit_grid=unet_output.logit,
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
                                 kl_logit_base=logit_kl_base.detach().item(),
                                 kl_logit_additional=logit_kl_additional.detach().item(),
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
