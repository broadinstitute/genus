import torch
import torch.nn.functional as F
import numpy

from .cropper_uncropper import Uncropper, Cropper
from .unet import UNet
from .encoders_decoders import EncoderConv, DecoderConv, Decoder1by1Linear, DecoderBackground
from .util import convert_to_box_list, invert_convert_to_box_list
from .util import compute_ranking, compute_average_in_box
from .util_ml import sample_and_kl_diagonal_normal, sample_c_grid
from .util_ml import compute_logp_dpp, compute_logp_bernoulli, SimilarityKernel
from .namedtuple import Inference, NmsOutput, BB, UNEToutput, ZZ, DIST, MetricMiniBatch
from .non_max_suppression import NonMaxSuppression


def optimal_bb_and_bb_regression_penalty(mixing_kb1wh: torch.Tensor,
                                         bounding_boxes_kb: BB,
                                         pad_size: int,
                                         min_box_size: float,
                                         max_box_size: float) -> (BB, torch.Tensor):
    """ Given the mixing probabilities and the predicted bounding_boxes it computes the optimal bounding_boxes and
        the L2 cost between the predicted bounding_boxes and ideal bounding_boxes.

        Note:
            The optimal bounding_box  is body-fitted around :math:`mask=(mixing > 0.5)`
            with a padding of size `attr:pad_size` pixels. If the mask is small (or completely empty)
            the optimal bounding_box is a box of the minimum_allowed size.

        Args:
            mixing_kb1wh: torch.Tensor of shape :math:`(K, B, 1, W, H)`
            bounding_boxes_kb: the bounding boxes predicted by the CNN of type :class:`BB` and shape :math:`(K,B)`
            pad_size: padding around the mask. If :attr:`pad_size` = 0 then the bounding box is body-fitted
            min_box_size: minimum allowed size for the bounding_box
            max_box_size: maximum allowed size for the bounding box

        Returns:
            The optimal bounding boxes in :class:`BB` of shape :math:`(K, B)` and
            the regression_penalty of shape :math:`(K, B)`
    """

    with torch.no_grad():

        # Compute the ideal Bounding boxes
        mask_kbwh = (mixing_kb1wh.squeeze(-3) > 0.5).int()
        mask_kbh = torch.max(mask_kbwh, dim=-2)[0]
        mask_kbw = torch.max(mask_kbwh, dim=-1)[0]
        mask_kb = torch.max(mask_kbw, dim=-1)[0]  # 0 if empty, 1 if non-empty

        plus_h = torch.arange(start=0, end=mask_kbh.shape[-1], step=1,
                              dtype=torch.float, device=mixing_kb1wh.device) + 1
        plus_w = torch.arange(start=0, end=mask_kbw.shape[-1], step=1,
                              dtype=torch.float, device=mixing_kb1wh.device) + 1
        minus_h = plus_h[-1] - plus_h + 1
        minus_w = plus_w[-1] - plus_w + 1

        # Find the coordinates of the bounding box
        ideal_x1_kb = (torch.argmax(mask_kbw * minus_w, dim=-1) - pad_size).clamp(min=0, max=mask_kbw.shape[-1]).float()
        ideal_x3_kb = (torch.argmax(mask_kbw * plus_w,  dim=-1) + pad_size).clamp(min=0, max=mask_kbw.shape[-1]).float()
        ideal_y1_kb = (torch.argmax(mask_kbh * minus_h, dim=-1) - pad_size).clamp(min=0, max=mask_kbh.shape[-1]).float()
        ideal_y3_kb = (torch.argmax(mask_kbh * plus_h,  dim=-1) + pad_size).clamp(min=0, max=mask_kbh.shape[-1]).float()

        # If the box is empty, do a special treatment, i.e. make them the smallest possible size
        empty_kb = (mask_kb == 0)
        ideal_x1_kb[empty_kb] = bounding_boxes_kb.bx[empty_kb] - 0.5 * min_box_size
        ideal_x3_kb[empty_kb] = bounding_boxes_kb.bx[empty_kb] + 0.5 * min_box_size
        ideal_y1_kb[empty_kb] = bounding_boxes_kb.by[empty_kb] - 0.5 * min_box_size
        ideal_y3_kb[empty_kb] = bounding_boxes_kb.by[empty_kb] + 0.5 * min_box_size

        # Compute the box coordinates (note the clamping of bw and bh)
        ideal_bx_kb = 0.5 * (ideal_x3_kb + ideal_x1_kb)
        ideal_by_kb = 0.5 * (ideal_y3_kb + ideal_y1_kb)
        ideal_bw_kb = (ideal_x3_kb - ideal_x1_kb).clamp(min=min_box_size, max=max_box_size)
        ideal_bh_kb = (ideal_y3_kb - ideal_y1_kb).clamp(min=min_box_size, max=max_box_size)

        # print("DEBUG min, max ->", min_box_size, max_box_size)
        # print("DEBUG input ->", bounding_boxes_kb.bx[0, 0], bounding_boxes_kb.by[0, 0],
        #       bounding_boxes_kb.bw[0, 0], bounding_boxes_kb.bh[0, 0], empty_kb[0, 0])
        # print("DEBUG ideal ->", ideal_bx_kb[0, 0], ideal_by_kb[0, 0],
        #       ideal_bw_kb[0, 0], ideal_bh_kb[0, 0], empty_kb[0, 0])

        # Outside the torch.no_grad() compute the regression cost
    cost_bb_regression = torch.abs(ideal_bx_kb - bounding_boxes_kb.bx) + \
                         torch.abs(ideal_by_kb - bounding_boxes_kb.by) + \
                         torch.abs(ideal_bw_kb - bounding_boxes_kb.bw) + \
                         torch.abs(ideal_bh_kb - bounding_boxes_kb.bh)

    return BB(bx=ideal_bx_kb, by=ideal_by_kb, bw=ideal_bw_kb, bh=ideal_bh_kb), cost_bb_regression


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
        self.mask_overlap_type = config["loss"]["mask_overlap_penalty_type"]

        self.min_box_size = config["input_image"]["range_object_size"][0]
        self.max_box_size = config["input_image"]["range_object_size"][1]
        self.glimpse_size = config["architecture"]["glimpse_size"]
        self.pad_size_bb = config["loss"]["bounding_box_regression_padding"]

        # modules
        self.similarity_kernel_dpp = SimilarityKernel(length_scale=config["input_image"]["similarity_DPP_l"],
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

        self.geco_loglambda_fgfraction = torch.nn.Parameter(data=torch.tensor(0.0, dtype=torch.float),
                                                            requires_grad=True)
        self.geco_loglambda_ncell = torch.nn.Parameter(data=torch.tensor(0.0, dtype=torch.float),
                                                       requires_grad=True)
        self.geco_loglambda_mse = torch.nn.Parameter(data=torch.tensor(self.geco_loglambda_ncell_max,
                                                                       dtype=torch.float), requires_grad=True)

##    @staticmethod
##    def _compute_logit_corrected(logit_praw: torch.Tensor,
##                                 p_correction: torch.Tensor,
##                                 prob_corr_factor: float):
##        """
##        In log space computes the probability correction:
##        1. p = (1-a) * p_raw + a * p_corr
##        2. (1-p) = (1-a) * (1-p_raw) + a * (1-p_corr)
##        3. logit_p = log_p - log_1_m_p
##        It returns the logit of the corrected probability
##        """
##
##        log_praw = F.logsigmoid(logit_praw)
##        log_1_m_praw = F.logsigmoid(-logit_praw)
##        log_a = torch.tensor(prob_corr_factor, device=logit_praw.device, dtype=logit_praw.dtype).log()
##        log_1_m_a = torch.tensor(1 - prob_corr_factor, device=logit_praw.device, dtype=logit_praw.dtype).log()
##        log_p_corrected = torch.logaddexp(log_praw + log_1_m_a, torch.log(p_correction) + log_a)
##        log_1_m_p_corrected = torch.logaddexp(log_1_m_praw + log_1_m_a, torch.log1p(-p_correction) + log_a)
##        logit_corrected = log_p_corrected - log_1_m_p_corrected
##        return logit_corrected

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

        # TODO: Replace the background block with a VQ-VAE
        # Compute the background
        zbg: DIST = sample_and_kl_diagonal_normal(posterior_mu=unet_output.zbg.mu,
                                                  posterior_std=unet_output.zbg.std,
                                                  prior_mu=torch.zeros_like(unet_output.zbg.mu),
                                                  prior_std=torch.ones_like(unet_output.zbg.std),
                                                  noisy_sampling=noisy_sampling,
                                                  sample_from_prior=generate_synthetic_data)
        out_background_bcwh = torch.sigmoid(self.decoder_zbg(z=zbg.sample, high_resolution=(imgs_bcwh.shape[-2],
                                                                                            imgs_bcwh.shape[-1])))

        # bounding boxes
        zwhere_grid: DIST = sample_and_kl_diagonal_normal(posterior_mu=unet_output.zwhere.mu,
                                                          posterior_std=unet_output.zwhere.std,
                                                          prior_mu=torch.zeros_like(unet_output.zwhere.mu),
                                                          prior_std=torch.ones_like(unet_output.zwhere.std),
                                                          noisy_sampling=noisy_sampling,
                                                          sample_from_prior=generate_synthetic_data)

        decoded_zwhere = torch.sigmoid(self.decoder_zwhere(zwhere_grid.sample))
        bounding_box_nb: BB = tgrid_to_bb(t_grid=decoded_zwhere,
                                          width_input_image=width_raw_image,
                                          height_input_image=height_raw_image,
                                          min_box_size=self.min_box_size,
                                          max_box_size=self.max_box_size)

        # Correct probability if necessary
        with torch.no_grad():
            if (prob_corr_factor > 0) and (prob_corr_factor <= 1.0):

                # Compute the ranking
                stupid_bb_nb: BB = tgrid_to_bb(t_grid=0.5*torch.ones_like(decoded_zwhere),
                                               width_input_image=width_raw_image,
                                               height_input_image=height_raw_image,
                                               min_box_size=self.min_box_size,
                                               max_box_size=self.max_box_size)

                av_intensity_nb = compute_average_in_box((imgs_bcwh - out_background_bcwh).abs(), stupid_bb_nb)
                ranking_nb = compute_ranking(av_intensity_nb)  # It is in [0,n-1]
                unit_ranking_nb = (ranking_nb + 1).float() / (ranking_nb.shape[-2]+1)  # strictly inside (0,1) range
                unit_ranking_b1wh = invert_convert_to_box_list(unit_ranking_nb.unsqueeze(-1),
                                                               original_width=unet_output.logit.shape[-2],
                                                               original_height=unet_output.logit.shape[-1])

                # Extract the local-maxima from the ranking
                tmp_pooled_b1wh = F.max_pool2d(unit_ranking_b1wh, kernel_size=3, stride=1,
                                               padding=1, return_indices=False)
                local_maxima_mask_b1wh = (tmp_pooled_b1wh == unit_ranking_b1wh)

                # Now select the top k local maxima
                score_b1wh = unit_ranking_b1wh * local_maxima_mask_b1wh + 0.001 * torch.rand_like(unit_ranking_b1wh)
                score_nb = convert_to_box_list(score_b1wh).squeeze(-1)
                values_kb, index_kb = torch.topk(score_nb, k=k_objects_max, dim=-2, largest=True, sorted=True)
                k_local_maxima_mask_nb = torch.zeros_like(score_nb).scatter(dim=-2,
                                                                            index=index_kb,
                                                                            src=torch.ones_like(score_nb))
                # For debug
                # tmp_kb = torch.sort(score_nb*k_local_maxima_mask_nb, dim=-2, descending=True)[0][:k_objects_max]
                # print(tmp_kb.shape, values_kb.shape)
                # print(tmp_kb[:,0])
                # print(values_kb[:,0])

                k_local_maxima_mask_b1wh = invert_convert_to_box_list(k_local_maxima_mask_nb.unsqueeze(-1),
                                                                      original_width=unet_output.logit.shape[-2],
                                                                      original_height=unet_output.logit.shape[-1])

                # Now compute the target_probability
                p_target_b1wh = (k_local_maxima_mask_b1wh * unit_ranking_b1wh).clamp(min=0.0001, max=0.9999)
                logit_target_b1wh = torch.log(p_target_b1wh) - torch.log1p(-p_target_b1wh)
            elif prob_corr_factor == 0:
                unit_ranking_b1wh = 0.5 * torch.ones_like(unet_output.logit)
                p_target_b1wh = 0.5 * torch.ones_like(unet_output.logit)
                logit_target_b1wh = unet_output.logit
            else:
                raise Exception("prob_corr_factor has an invalid value", prob_corr_factor)

        # Outside torch.no_grad()
        # Note that the logit_warming_loss will keep the unet_output.logit close to
        logit_warming_loss = prob_corr_factor * (logit_target_b1wh.detach() - unet_output.logit).pow(2).sum()
        unet_prob_b1wh = torch.sigmoid(unet_output.logit)
        logit_min = torch.min(unet_output.logit).detach()
        logit_max = torch.max(unet_output.logit).detach()
        print("DEBUG logit_grid min,max -->", logit_min.item(), logit_max.item())
        #if torch.isnan(logit_min) or torch.isnan(logit_max):
        #    raise Exception("logit_unet_is_nan")

        # Sample the probability map from prior or posterior
        similarity_kernel = self.similarity_kernel_dpp.forward(n_width=unet_output.logit.shape[-2],
                                                               n_height=unet_output.logit.shape[-1])
        # NMS + top-K operation
        with torch.no_grad():
            c_grid_before_nms_b1wh = sample_c_grid(prob_grid=unet_prob_b1wh,
                                                   similarity_matrix=similarity_kernel,
                                                   noisy_sampling=noisy_sampling,
                                                   sample_from_prior=generate_synthetic_data)

            score_nb = convert_to_box_list(c_grid_before_nms_b1wh + unet_prob_b1wh).squeeze(-1)
            combined_topk_only = topk_only or generate_synthetic_data  # if generating from DPP do not do NMS
            nms_output: NmsOutput = NonMaxSuppression.compute_mask_and_index(score_nb=score_nb,
                                                                             bounding_box_nb=bounding_box_nb,
                                                                             iom_threshold=iom_threshold,
                                                                             k_objects_max=k_objects_max,
                                                                             topk_only=combined_topk_only)
            # Mask with all zero except 1s in the locations specified by the indices
            mask_nb = torch.zeros_like(score_nb).scatter(dim=0,
                                                         index=nms_output.indices_kb,
                                                         src=torch.ones_like(score_nb))
            mask_grid_b1wh = invert_convert_to_box_list(mask_nb.unsqueeze(-1),
                                                        original_width=c_grid_before_nms_b1wh.shape[-2],
                                                        original_height=c_grid_before_nms_b1wh.shape[-1])
            c_grid_after_nms_b1wh = c_grid_before_nms_b1wh * mask_grid_b1wh

        # Compute KL divergence between the DPP prior and the posterior: KL = logp(c|logit) - logp(c|similarity)
        # The effect of this term should be:
        # 1. DECREASE logit where c=1, INCREASE logit where c=0 (i.e. make the posterior distribution more entropic)
        # 2. Make the DPP parameters adjust to the seen configurations
        c_grid_logp_prior_b = compute_logp_dpp(c_grid=c_grid_after_nms_b1wh.detach(),
                                               similarity_matrix=similarity_kernel)
        c_grid_logp_posterior_b = compute_logp_bernoulli(c_grid=c_grid_after_nms_b1wh.detach(),
                                                         logit_grid=unet_output.logit)
        kl_logit_b = c_grid_logp_posterior_b - c_grid_logp_prior_b

        # Gather all relevant quantities from the selected boxes
        bounding_box_kb: BB = BB(bx=torch.gather(bounding_box_nb.bx, dim=0, index=nms_output.indices_kb),
                                 by=torch.gather(bounding_box_nb.by, dim=0, index=nms_output.indices_kb),
                                 bw=torch.gather(bounding_box_nb.bw, dim=0, index=nms_output.indices_kb),
                                 bh=torch.gather(bounding_box_nb.bh, dim=0, index=nms_output.indices_kb))

        prob_kb = torch.gather(convert_to_box_list(unet_prob_b1wh).squeeze(-1),
                               dim=0, index=nms_output.indices_kb)
        c_detached_kb = torch.gather(convert_to_box_list(c_grid_after_nms_b1wh).squeeze(-1),
                                     dim=0, index=nms_output.indices_kb)

        zwhere_kl_nbz = convert_to_box_list(zwhere_grid.kl)
        indices_kbz = nms_output.indices_kb.unsqueeze(-1).expand(-1, -1, zwhere_kl_nbz.shape[-1])
        zwhere_kl_kbz = torch.gather(zwhere_kl_nbz, dim=0, index=indices_kbz)

        # 5. Crop the unet_features according to the selected boxes
        n_boxes, batch_size = bounding_box_kb.bx.shape
        unet_features_expanded = unet_output.features.expand(n_boxes, batch_size, -1, -1, -1)
        cropped_feature_map = Cropper.crop(bounding_box=bounding_box_kb,
                                           big_stuff=unet_features_expanded,
                                           width_small=self.glimpse_size,
                                           height_small=self.glimpse_size)

        # 6. Encode, sample z and decode to big images and big weights
        zinstance_posterior: ZZ = self.encoder_zinstance.forward(cropped_feature_map)
        zinstance_few: DIST = sample_and_kl_diagonal_normal(posterior_mu=zinstance_posterior.mu,
                                                            posterior_std=zinstance_posterior.std,
                                                            prior_mu=torch.zeros_like(zinstance_posterior.mu),
                                                            prior_std=torch.ones_like(zinstance_posterior.std),
                                                            noisy_sampling=noisy_sampling,
                                                            sample_from_prior=generate_synthetic_data)

        # Note that the last channel is a mask (i.e. there is a sigmoid non-linearity applied)
        # It is important that the sigmoid is applied before uncropping on a zero-canvas so that mask is zero everywhere
        # except inside the bounding boxes
        small_stuff = torch.sigmoid(self.decoder_zinstance.forward(zinstance_few.sample))
        big_stuff = Uncropper.uncrop(bounding_box=bounding_box_kb,
                                     small_stuff=small_stuff,
                                     width_big=width_raw_image,
                                     height_big=height_raw_image)  # shape: n_box, batch, ch, w, h
        out_img_kbcwh, out_mask_kb1wh = torch.split(big_stuff,
                                                    split_size_or_sections=(big_stuff.shape[-3] - 1, 1),
                                                    dim=-3)

        # 7. Compute the mixing
        # TODO: think hard about this part with Mehrtash
        #   There are many ways to make c differentiable
        #   Should I detach the denominator?
        #   Should I introduce z_depth?
        # Ideally I would like to multiply by c. However if c=0 I can not learn anything. Therefore use max(p,c).
        # c_differentiable_kb = prob_kb - prob_kb.detach() + torch.max(prob_kb, c_detached_kb).detach()
        # c_detached_times_mask_kb1wh = c_detached_kb[..., None, None, None] * out_mask_kb1wh
        c_differentiable_kb = prob_kb + (c_detached_kb - prob_kb).detach()
        c_differentiable_times_mask_kb1wh = c_differentiable_kb[..., None, None, None] * out_mask_kb1wh
        mixing_kb1wh = c_differentiable_times_mask_kb1wh / c_differentiable_times_mask_kb1wh.sum(dim=-5).clamp(min=1.0)
        mixing_fg_b1wh = mixing_kb1wh.sum(dim=-5)  # sum over boxes
        mixing_bg_b1wh = torch.ones_like(mixing_fg_b1wh) - mixing_fg_b1wh

        # c_or_p_attached_kb = prob_kb - prob_kb.detach() + torch.max(prob_kb, c_detached_kb).detach()
        # c_or_p_times_mask_kb1wh = out_mask_kb1wh * c_or_p_attached_kb[..., None, None, None]
        # mixing_kb1wh = c_or_p_times_mask_kb1wh / out_mask_kb1wh.sum(dim=-5).clamp(min=1.0)  # softplus-like
        # mixing_kb1wh = c_or_p_times_mask_kb1wh / c_or_p_times_mask_kb1wh.sum(dim=-5).clamp(min=1.0)  # softplus-like
        # mixing_fg_b1wh = mixing_kb1wh.sum(dim=-5)  # sum over boxes
        # mixing_bg_b1wh = torch.ones_like(mixing_fg_b1wh) - mixing_fg_b1wh

        # 8. Compute the ideal bounding boxes
        bb_ideal_kb, bb_regression_kb = optimal_bb_and_bb_regression_penalty(mixing_kb1wh=mixing_kb1wh,
                                                                             bounding_boxes_kb=bounding_box_kb,
                                                                             pad_size=self.pad_size_bb,
                                                                             min_box_size=self.min_box_size,
                                                                             max_box_size=self.max_box_size)
        # TODO: should I multiply this by c_detached_kb
        #   if I multiply bu c_attached then the nounding boxes with smaller bb_regression cost are favored
        cost_bb_regression = self.bb_regression_strength * torch.sum(c_detached_kb * bb_regression_kb)/batch_size

        # 9. Compute the mask overlap penalty using c_detached so that this penalty changes
        #  the mask but not the probabilities
        # TODO: detach c when computing the overlap? Probably yes.
        if self.mask_overlap_type == 1:
            # APPROACH 1: Compute mask overlap penalty
            mask_overlap_v1 = c_differentiable_times_mask_kb1wh.sum(dim=-5).pow(2) - \
                              c_differentiable_times_mask_kb1wh.pow(2).sum(dim=-5)
            # print("DEBUG mask_overlap1 min,max", torch.min(mask_overlap_v1), torch.max(mask_overlap_v1))
            cost_overlap_tmp_v1 = torch.sum(mask_overlap_v1, dim=(-1, -2, -3))  # sum over ch, w, h
            cost_overlap_v1 = self.mask_overlap_strength * cost_overlap_tmp_v1.mean()  # mean over batch
            cost_overlap = cost_overlap_v1
        elif self.mask_overlap_type == 2:
            # APPROACH 2: Compute mask overlap
            mask_overlap_v2 = torch.sum(mixing_kb1wh * (torch.ones_like(mixing_kb1wh) - mixing_kb1wh), dim=-5)
            cost_overlap_tmp_v2 = torch.sum(mask_overlap_v2, dim=(-1, -2, -3))
            cost_overlap_v2 = self.mask_overlap_strength * cost_overlap_tmp_v2.mean()
            cost_overlap = cost_overlap_v2
        else:
            raise Exception("self.mask_overlap_type not valid")

        inference = Inference(logit_grid=unet_output.logit,
                              logit_grid_unet=unet_output.logit,
                              prob_grid_target=p_target_b1wh,
                              prob_unit_ranking=unit_ranking_b1wh,
                              background_bcwh=out_background_bcwh,
                              foreground_kbcwh=out_img_kbcwh,
                              sum_c_times_mask_b1wh=c_differentiable_times_mask_kb1wh.sum(dim=-5),
                              mixing_kb1wh=mixing_kb1wh,
                              sample_c_grid_before_nms=c_grid_before_nms_b1wh,
                              sample_c_grid_after_nms=c_grid_after_nms_b1wh,
                              sample_c_kb=c_detached_kb,
                              sample_bb_kb=bounding_box_kb,
                              sample_bb_ideal_kb=bb_ideal_kb)

        # ---------------------------------- #
        # Compute the metrics

        # 1. Observation model
        mse_fg_kbcwh = ((out_img_kbcwh - imgs_bcwh) / self.sigma_fg).pow(2)
        mse_bg_bcwh = ((out_background_bcwh - imgs_bcwh) / self.sigma_bg).pow(2)
        mse_av = ((mixing_kb1wh * mse_fg_kbcwh).sum(dim=-5) + mixing_bg_b1wh * mse_bg_bcwh).mean()

        # 2. KL divergence
        # Note that divide by batch_size only.
        # I could divide by latent dimension.
        # I can not divide by number of object b/c that could become zero
        c_detached_kb1 = c_detached_kb[..., None]
        kl_zbg = torch.sum(zbg.kl) / batch_size
        kl_zinstance = torch.sum(zinstance_few.kl * c_detached_kb1) / batch_size
        kl_zwhere = torch.sum(zwhere_kl_kbz * c_detached_kb1) / batch_size
        kl_logit = torch.mean(kl_logit_b) / batch_size
        kl_av = kl_zbg + kl_zinstance + kl_zwhere + kl_logit

        with torch.no_grad():
            ncell_av = torch.sum(c_detached_kb) / batch_size
            fgfraction_av = torch.mean(mixing_fg_b1wh)

            mse_in_range = (mse_av > self.geco_target_mse_min) & (mse_av < self.geco_target_mse_max)
            ncell_in_range = (ncell_av > self.geco_target_ncell_min) & (ncell_av < self.geco_target_ncell_max)
            fgfraction_in_range = (fgfraction_av > self.geco_target_fgfraction_min) & \
                                  (fgfraction_av < self.geco_target_fgfraction_max)

            # Clamp the log_lambda into the allowed regime
            self.geco_loglambda_mse.data.clamp_(min=self.geco_loglambda_mse_min,
                                                max=self.geco_loglambda_mse_max)
            self.geco_loglambda_ncell.data.clamp_(min=self.geco_loglambda_ncell_min,
                                                  max=self.geco_loglambda_ncell_max)
            self.geco_loglambda_fgfraction.data.clamp_(min=self.geco_loglambda_fgfraction_min,
                                                       max=self.geco_loglambda_fgfraction_max)

            # From log_lambda to lambda
            lambda_mse = self.geco_loglambda_mse.data.exp() * torch.sign(mse_av - self.geco_target_mse_min)
            lambda_ncell = self.geco_loglambda_ncell.data.exp() * torch.sign(ncell_av - self.geco_target_ncell_min)
            lambda_fgfraction = self.geco_loglambda_fgfraction.data.exp() * torch.sign(fgfraction_av -
                                                                                       self.geco_target_fgfraction_min)

        # Loss geco (i.e. makes loglambda increase or decrease)
        # I want to implement: increase slowly, decrease rapidly
        loss_geco = logit_warming_loss + \
                    self.geco_loglambda_mse * (1.0 * mse_in_range - 1.0 * ~mse_in_range) + \
                    self.geco_loglambda_fgfraction * (1.0 * fgfraction_in_range - 1.0 * ~fgfraction_in_range) + \
                    self.geco_loglambda_ncell * (1.0 * ncell_in_range - 1.0 * ~ncell_in_range)

        # TODO: should lambda_ncell act on all probabilities or only the selected ones?
        #  what about acting on the underlying logits
        #  should lambda_fgfraction act on small_mask or large_mask?
        #  should lambda_ncell act on logit or probabilities?
        loss_vae = cost_overlap + cost_bb_regression + kl_av + \
                   lambda_mse.detach() * mse_av + \
                   lambda_ncell.detach() * torch.mean(unet_output.logit) + \
                   lambda_fgfraction.detach() * torch.mean(out_mask_kb1wh)

        loss = loss_vae + loss_geco - loss_geco.detach()

        similarity_l, similarity_w = self.similarity_kernel_dpp.get_l_w()

        # add everything you want as long as there is one loss
        metric = MetricMiniBatch(loss=loss,
                                 logit_warming_loss=logit_warming_loss.detach().item(),
                                 mse_av=mse_av.detach().item(),
                                 kl_av=kl_av.detach().item(),
                                 kl_logit=kl_logit.detach().item(),
                                 kl_zinstance=kl_zinstance.detach().item(),
                                 kl_zbg=kl_zbg.detach().item(),
                                 kl_zwhere=kl_zwhere.detach().item(),
                                 cost_mask_overlap_av=cost_overlap.detach().item(),
                                 cost_bb_regression_av=cost_bb_regression.detach().item(),
                                 ncell_av=ncell_av.detach().item(),
                                 fgfraction_av=fgfraction_av.detach().item(),
                                 lambda_mse=lambda_mse.detach().item(),
                                 lambda_ncell=lambda_ncell.detach().item(),
                                 lambda_fgfraction=lambda_fgfraction.detach().item(),
                                 count_prediction=torch.sum(c_detached_kb, dim=0).detach().cpu().numpy(),
                                 wrong_examples=-1 * numpy.ones(1),
                                 accuracy=-1.0,
                                 similarity_l=similarity_l.detach().item(),
                                 similarity_w=similarity_w.detach().item())

        return inference, metric
