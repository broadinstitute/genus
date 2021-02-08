import torch
import torch.nn.functional as F
import numpy

from .cropper_uncropper import Uncropper, Cropper
from .unet import UNet
from .encoders_decoders import EncoderConv, DecoderConv, Decoder1by1Linear, DecoderBackground
from .util import convert_to_box_list, invert_convert_to_box_list
from .util import compute_ranking, compute_average_in_box
from .util_ml import sample_and_kl_diagonal_normal, sample_c_map
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

        plus_h = torch.arange(start=0, end=mask_kbh.shape[-1], step=1, dtype=torch.float, device=mixing_kb1wh.device) + 1
        plus_w = torch.arange(start=0, end=mask_kbw.shape[-1], step=1, dtype=torch.float, device=mixing_kb1wh.device) + 1
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

        # Now compute the regression cost for bw and bh (bx,by are NOT subject to regression cost)
        # TODO: Make a regression cost for bx and by.
        #  The problem is that depending on the value of bx, by different grid cell are responsible
        x1_tmp_kb = ideal_bx_kb - 0.5 * ideal_bw_kb
        x3_tmp_kb = ideal_bx_kb + 0.5 * ideal_bw_kb
        y1_tmp_kb = ideal_by_kb - 0.5 * ideal_bh_kb
        y3_tmp_kb = ideal_by_kb + 0.5 * ideal_bh_kb

        bw_target_kb = torch.max(x3_tmp_kb - bounding_boxes_kb.bx,
                                 bounding_boxes_kb.bx - x1_tmp_kb).clamp(min=min_box_size, max=max_box_size)
        bh_target_kb = torch.max(y3_tmp_kb - bounding_boxes_kb.bw,
                                 bounding_boxes_kb.bw - y1_tmp_kb).clamp(min=min_box_size, max=max_box_size)

##        print("DEBUG min, max ->", min_box_size, max_box_size)
##        print("DEBUG input ->", bounding_boxes_kb.bx[0, 0], bounding_boxes_kb.by[0, 0],
##              bounding_boxes_kb.bw[0, 0], bounding_boxes_kb.bh[0, 0], empty_kb[0, 0])
##        print("DEBUG ideal ->", ideal_bx_kb[0, 0], ideal_by_kb[0, 0],
##              ideal_bw_kb[0, 0], ideal_bh_kb[0, 0], empty_kb[0, 0])

    # this is the only part outside the torch.no_grad()
    cost_bb_regression = ((bw_target_kb - bounding_boxes_kb.bw)/min_box_size).pow(2) + \
                         ((bh_target_kb - bounding_boxes_kb.bh)/min_box_size).pow(2)

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


def from_w_to_pi(weight: torch.Tensor, dim: int):
    """ Compute the interacting and non-interacting mixing probabilities
        Make sure that when summing over dim=dim the mask sum to zero or one
        mask_j = fg_mask * partitioning_j
        where fg_mask = tanh ( sum_i w_i) and partitioning_j = w_j / (sum_i w_i)
    """
    assert len(weight.shape) == 5
    sum_weight = torch.sum(weight, dim=dim, keepdim=True)
    fg_mask = torch.tanh(sum_weight)
    partitioning = weight / torch.clamp(sum_weight, min=1E-6)
    return fg_mask * partitioning


class InferenceAndGeneration(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.geco_target_mse_min = 0.25
        self.geco_target_mse_max = 0.75
        self.geco_target_ncell_min = 1.0
        self.geco_target_ncell_max = 3.0
        self.geco_target_fgfraction_min = 0.02
        self.geco_target_fgfraction_max = 0.10

        # variables
        self.bb_regression_strength = config["loss"]["bounding_box_regression_penalty_strength"]
        self.mask_overlap_strength = config["loss"]["mask_overlap_penalty_strength"]
        self.mask_overlap_type = config["loss"]["mask_overlap_penalty_type"]

        self.size_min = config["input_image"]["range_object_size"][0]
        self.size_max = config["input_image"]["range_object_size"][1]
        self.cropped_size = config["architecture"]["glimpse_size"]
        self.pad_size_bb = config["loss"]["bounding_box_regression_padding"]

        # modules
        self.similarity_kernel_dpp = SimilarityKernel(n_kernels=1)
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

        # Decoders
        self.decoder_zbg: DecoderBackground = DecoderBackground(dim_z=config["architecture"]["dim_zbg"],
                                                                ch_out=config["architecture"]["n_ch_img"])

        self.decoder_zwhere: Decoder1by1Linear = Decoder1by1Linear(dim_z=config["architecture"]["dim_zwhere"],
                                                                   ch_out=4)

        self.decoder_logit: Decoder1by1Linear = Decoder1by1Linear(dim_z=1,
                                                                  ch_out=1)

        self.decoder_zinstance: DecoderConv = DecoderConv(size=config["architecture"]["glimpse_size"],
                                                          dim_z=config["architecture"]["dim_zinstance"],
                                                          ch_out=config["architecture"]["n_ch_img"] + 1)

        # Encoders
        self.encoder_zinstance: EncoderConv = EncoderConv(size=config["architecture"]["glimpse_size"],
                                                          ch_in=config["architecture"]["n_ch_output_features"],
                                                          dim_z=config["architecture"]["dim_zinstance"])

        # Raw image parameters
        self.sigma_fg = torch.nn.Parameter(data=torch.tensor(0.1, dtype=torch.float)[..., None, None], requires_grad=False)
        self.sigma_bg = torch.nn.Parameter(data=torch.tensor(0.1, dtype=torch.float)[..., None, None], requires_grad=False)

        self.geco_dict = config["loss"]

        self.geco_fgfraction = torch.nn.Parameter(data=torch.tensor(1.0, dtype=torch.float), requires_grad=True)
        self.geco_ncell = torch.nn.Parameter(data=torch.tensor(1.0, dtype=torch.float), requires_grad=True)
        self.geco_mse = torch.nn.Parameter(data=torch.tensor(0.8, dtype=torch.float), requires_grad=True)

        self.running_avarage_kl_logit = torch.nn.Parameter(data=4 * torch.ones(1, dtype=torch.float),
                                                           requires_grad=True)

    @staticmethod
    def NLL_MSE(output: torch.tensor, target: torch.tensor, sigma: torch.tensor) -> torch.Tensor:
        return ((output - target) / sigma).pow(2)

    def forward(self, imgs_bcwh: torch.Tensor,
                generate_synthetic_data: bool,
                prob_corr_factor: float,
                iom_threshold: float,
                k_objects_max: int,
                topk_only: bool,
                noisy_sampling: bool,
                quantize_prob: bool = False,
                quantize_prob_value: float = 0.5) -> (Inference, MetricMiniBatch):

        # compute the inference

        # 0. preparation
        batch_size, ch_raw_image, width_raw_image, height_raw_image = imgs_bcwh.shape

        # ---------------------------#
        # 1. UNET
        # ---------------------------#
        unet_output: UNEToutput = self.unet.forward(imgs_bcwh, verbose=False)

        # background
        zbg: DIST = sample_and_kl_diagonal_normal(posterior_mu=unet_output.zbg.mu,
                                                  posterior_std=unet_output.zbg.std,
                                                  prior_mu=torch.zeros_like(unet_output.zbg.mu),
                                                  prior_std=torch.ones_like(unet_output.zbg.std),
                                                  noisy_sampling=noisy_sampling,
                                                  sample_from_prior=generate_synthetic_data)

        out_background_bcwh = torch.sigmoid(self.decoder_zbg(z=zbg.sample,
                                                             high_resolution=(imgs_bcwh.shape[-2],
                                                                              imgs_bcwh.shape[-1])))

        # bounbding boxes
        zwhere_map: DIST = sample_and_kl_diagonal_normal(posterior_mu=unet_output.zwhere.mu,
                                                         posterior_std=unet_output.zwhere.std,
                                                         prior_mu=torch.zeros_like(unet_output.zwhere.mu),
                                                         prior_std=torch.ones_like(unet_output.zwhere.std),
                                                         noisy_sampling=noisy_sampling,
                                                         sample_from_prior=generate_synthetic_data)

        bounding_box_nb: BB = tgrid_to_bb(t_grid=torch.sigmoid(self.decoder_zwhere(zwhere_map.sample)),
                                          width_input_image=width_raw_image,
                                          height_input_image=height_raw_image,
                                          min_box_size=self.size_min,
                                          max_box_size=self.size_max)

        with torch.no_grad():

            # Correct probability if necessary
            if (prob_corr_factor > 0) and (prob_corr_factor <= 1.0):
                av_intensity_nb = compute_average_in_box((imgs_bcwh - out_background_bcwh).abs(), bounding_box_nb)
                assert len(av_intensity_nb.shape) == 2
                n_boxes_all, batch_size = av_intensity_nb.shape
                ranking_nb = compute_ranking(av_intensity_nb)  # n_boxes_all, batch. It is in [0,n_box_all-1]
                tmp = (ranking_nb + 1).float() / (n_boxes_all + 1)  # less or equal to 1
                q_approx = tmp.pow(10)  # suppress most probabilities but keep few close to 1.
                p_map_delta = invert_convert_to_box_list(q_approx.unsqueeze(-1),
                                                         original_width=unet_output.logit.shape[-2],
                                                         original_height=unet_output.logit.shape[-1])

        # Now I have p, log(p), log(1-p)
        if (prob_corr_factor > 0) and (prob_corr_factor <= 1.0):
            p_map = ((1 - prob_corr_factor) * torch.sigmoid(unet_output.logit) +
                     prob_corr_factor * p_map_delta).clamp(min=1E-4, max=1 - 1E-4)
            log_p_map = torch.log(p_map)
            log_one_minus_p_map = torch.log1p(-p_map)
        else:
            p_map = torch.sigmoid(unet_output.logit)
            log_p_map = F.logsigmoid(unet_output.logit)
            log_one_minus_p_map = F.logsigmoid(-unet_output.logit)

        # Sample the probability map from prior or posterior
        similarity_kernel = self.similarity_kernel_dpp.forward(n_width=unet_output.logit.shape[-2],
                                                               n_height=unet_output.logit.shape[-1])
        if quantize_prob:
            # print("I am quantizing the probability")
            # print(p_map[0,0].sum(), (p_map > quantize_prob_value).float()[0,0].sum())
            c_map_before_nms = sample_c_map(p_map=(p_map > quantize_prob_value).float(),
                                            similarity_kernel=similarity_kernel,
                                            noisy_sampling=noisy_sampling,
                                            sample_from_prior=generate_synthetic_data)
        else:
            c_map_before_nms = sample_c_map(p_map=p_map,
                                            similarity_kernel=similarity_kernel,
                                            noisy_sampling=noisy_sampling,
                                            sample_from_prior=generate_synthetic_data)

        # NMS + top-K operation
        with torch.no_grad():
            score = convert_to_box_list(c_map_before_nms + p_map).squeeze(-1)  # shape: n_box_all, batch_size
            combined_topk_only = topk_only or generate_synthetic_data  # if generating from DPP do not do NMS
            nms_output: NmsOutput = NonMaxSuppression.compute_mask_and_index(score_nb=score,
                                                                             bounding_box_nb=bounding_box_nb,
                                                                             iom_threshold=iom_threshold,
                                                                             k_objects_max=k_objects_max,
                                                                             topk_only=combined_topk_only)
            # Mask with all zero except 1s where the box where selected
            mask = torch.zeros_like(score).scatter(dim=0,
                                                   index=nms_output.indices_kb,
                                                   src=torch.ones_like(score))  # shape: n_box_all, batch_size
            mask_map = invert_convert_to_box_list(mask.unsqueeze(-1),
                                                  original_width=c_map_before_nms.shape[-2],
                                                  original_height=c_map_before_nms.shape[-1])  # shape: batch_size, 1, w, h

        # TODO: check if I can use the c_map_after_nms in both places.....
        c_map_after_nms = c_map_before_nms * mask_map
        logp_logit_prior = compute_logp_dpp(c_grid=c_map_after_nms.detach(),
                                            similarity_matrix=similarity_kernel)
        logp_logit_posterior = compute_logp_bernoulli(c_grid=c_map_after_nms.detach(),
                                                      logit_grid=log_p_map - log_one_minus_p_map)
        kl_logit = logp_logit_posterior - logp_logit_prior  # this will make adjust DPP and keep entropy of posterior

        c_few = torch.gather(convert_to_box_list(c_map_before_nms).squeeze(-1), dim=0, index=nms_output.indices_kb)

        bounding_box_kb: BB = BB(bx=torch.gather(bounding_box_nb.bx, dim=0, index=nms_output.indices_kb),
                                 by=torch.gather(bounding_box_nb.by, dim=0, index=nms_output.indices_kb),
                                 bw=torch.gather(bounding_box_nb.bw, dim=0, index=nms_output.indices_kb),
                                 bh=torch.gather(bounding_box_nb.bh, dim=0, index=nms_output.indices_kb))

        zwhere_sample_all = convert_to_box_list(zwhere_map.sample)  # shape: nbox_all, batch_size, ch
        zwhere_kl_all = convert_to_box_list(zwhere_map.kl)  # shape: nbox_all, batch_size, ch
        new_index = nms_output.indices_kb.unsqueeze(-1).expand(-1, -1, zwhere_kl_all.shape[
            -1])  # shape: nbox_few, batch_size, ch
        zwhere_kl_few = torch.gather(zwhere_kl_all, dim=0, index=new_index)  # shape (nbox_few, batch_size, ch)
        zwhere_sample_few = torch.gather(zwhere_sample_all, dim=0, index=new_index)

        # ------------------------------------------------------------------#
        # 5. Crop the unet_features according to the selected boxes
        # ------------------------------------------------------------------#
        n_boxes, batch_size = bounding_box_kb.bx.shape
        unet_features_expanded = unet_output.features.unsqueeze(0).expand(n_boxes, batch_size, -1, -1, -1)
        cropped_feature_map: torch.Tensor = Cropper.crop(bounding_box=bounding_box_kb,
                                                         big_stuff=unet_features_expanded,
                                                         width_small=self.cropped_size,
                                                         height_small=self.cropped_size)

        # ------------------------------------------------------------------#
        # 6. Encode, sample z and decode to big images and big weights
        # ------------------------------------------------------------------#
        zinstance_posterior: ZZ = self.encoder_zinstance.forward(cropped_feature_map)
        zinstance_few: DIST = sample_and_kl_diagonal_normal(posterior_mu=zinstance_posterior.mu,
                                                            posterior_std=zinstance_posterior.std,
                                                            prior_mu=torch.zeros_like(zinstance_posterior.mu),
                                                            prior_std=torch.ones_like(zinstance_posterior.std),
                                                            noisy_sampling=noisy_sampling,
                                                            sample_from_prior=generate_synthetic_data)

        small_stuff = torch.sigmoid(self.decoder_zinstance.forward(zinstance_few.sample))  # stuff between 0 and 1
        big_stuff = Uncropper.uncrop(bounding_box=bounding_box_kb,
                                     small_stuff=small_stuff,
                                     width_big=width_raw_image,
                                     height_big=height_raw_image)  # shape: n_box, batch, ch, w, h
        out_mask_kb1wh, out_img_kbcwh = torch.split(big_stuff, split_size_or_sections=(1, big_stuff.shape[-3] - 1), dim=-3)
        c_times_mask_kb1wh = out_mask_kb1wh * c_few[..., None, None, None]  # this is strictly smaller than 1
        sum_c_times_mask_b1wh = c_times_mask_kb1wh.sum(dim=-5)
        sum_c_times_mask_squared_b1wh = c_times_mask_kb1wh.pow(2).sum(dim=-5)
        mixing_kb1wh = c_times_mask_kb1wh / sum_c_times_mask_b1wh.clamp_(min=1.0)  # softplus-like function

        # Compute the ideal bounding boxes
        bb_ideal_kb, bb_regression_kb = optimal_bb_and_bb_regression_penalty(mixing_kb1wh=mixing_kb1wh,
                                                                             bounding_boxes_kb=bounding_box_kb,
                                                                             pad_size=self.pad_size_bb,
                                                                             min_box_size=self.size_min,
                                                                             max_box_size=self.size_max)
        cost_bb_regression = self.bb_regression_strength * bb_regression_kb.mean()

        # APPROACH 1: Compute mask overlap penalty
        mask_overlap_v1 = 0.5 * (sum_c_times_mask_b1wh.pow(2) - sum_c_times_mask_squared_b1wh).clamp(min=0)
        cost_overlap_tmp_v1 = torch.sum(mask_overlap_v1, dim=(-1, -2, -3))  # sum over ch, w, h
        cost_overlap_v1 = self.mask_overlap_strength * cost_overlap_tmp_v1.mean()  # mean over batch

        # APPROACH 2: Compute mask overlap
        mask_overlap_v2 = torch.sum(mixing_kb1wh * (torch.ones_like(mixing_kb1wh) - mixing_kb1wh), dim=-5)  # sum boxes
        cost_overlap_tmp_v2 = 0.01 * torch.sum(mask_overlap_v2, dim=(-1, -2, -3))
        cost_overlap_v2 = self.mask_overlap_strength * cost_overlap_tmp_v2.mean()

        if self.mask_overlap_type == 1:
            cost_overlap = cost_overlap_v1
        elif self.mask_overlap_type == 2:
            cost_overlap = cost_overlap_v2
        else:
            raise Exception("self.mask_overlap_type not valid")

        inference = Inference(logit_grid=log_p_map-log_one_minus_p_map,
                              logit_grid_unet=unet_output.logit,
                              background_bcwh=out_background_bcwh,
                              foreground_kbcwh=out_img_kbcwh,
                              mixing_kb1wh=mixing_kb1wh,
                              sample_c_grid_before_nms=c_map_before_nms,
                              sample_c_grid_after_nms=c_map_after_nms,
                              sample_c_kb=c_few,
                              sample_bb_kb=bounding_box_kb,
                              sample_bb_ideal_kb=bb_ideal_kb)

        # ---------------------------------- #
        # Compute the metrics
        n_box_few, batch_size = c_few.shape

        # 1. Observation model
        mixing_fg = torch.sum(mixing_kb1wh, dim=-5)  # sum over boxes
        mixing_bg = torch.ones_like(mixing_fg) - mixing_fg
        mse = InferenceAndGeneration.NLL_MSE(output=out_img_kbcwh,
                                             target=imgs_bcwh,
                                             sigma=self.sigma_fg)  # boxes, batch_size, ch, w, h
        mse_bg = InferenceAndGeneration.NLL_MSE(output=out_background_bcwh,
                                                target=imgs_bcwh,
                                                sigma=self.sigma_bg)  # batch_size, ch, w, h
        mse_av = ((mixing_kb1wh * mse).sum(dim=-5) + mixing_bg * mse_bg).mean()  # mean over batch_size, ch, w, h

        # TODO: put htis stuff inside torch.no_grad()
        with torch.no_grad():
            g_mse = (self.geco_target_mse_min - mse_av).clamp(min=0) + \
                    (self.geco_target_mse_max - mse_av).clamp(max=0)

        # 2. Sparsity should encourage:
        # 1. few object
        # 2. tight bounding boxes
        # 3. tight masks
        # The three terms take care of all these requirement.
        # Note:
        # 1) All the terms contain c=Bernoulli(p). It is actually the same b/c during back prop c=p
        # 2) fg_fraction is based on the selected quantities
        # 3) sparsity n_cell is based on c_map so that the entire matrix becomes sparse.
        with torch.no_grad():
            x_sparsity_av = torch.mean(mixing_fg)
            x_sparsity_max = self.geco_target_fgfraction_max
            x_sparsity_min = self.geco_target_fgfraction_min
            g_sparsity = torch.min(x_sparsity_av - x_sparsity_min,
                                   x_sparsity_max - x_sparsity_av)  # positive if in range
        # TODO remove c_times_area_few from the sparsity term
        c_times_area_few = c_few * bounding_box_kb.bw * bounding_box_kb.bh
        x_sparsity = 0.5 * (torch.sum(mixing_fg) + torch.sum(c_times_area_few)) / torch.numel(mixing_fg)
        f_sparsity = x_sparsity * torch.sign(x_sparsity_av - x_sparsity_min).detach()

        with torch.no_grad():
            x_cell_av = torch.sum(c_map_after_nms) / batch_size
            x_cell_max = self.geco_target_ncell_max
            x_cell_min = self.geco_target_ncell_min
            g_cell = torch.min(x_cell_av - x_cell_min,
                               x_cell_max - x_cell_av) / n_box_few  # positive if in range, negative otherwise
        x_cell = torch.sum(c_map_before_nms) / (batch_size * n_box_few)
        f_cell = x_cell * torch.sign(x_cell_av - x_cell_min).detach()

        # 3. compute KL
        # Note that I compute the mean over batch, latent_dimensions and n_object.
        # This means that latent_dim can effectively control the complexity of the reconstruction,
        # i.e. more latent more capacity.
        kl_zbg = torch.mean(zbg.kl)  # mean over: batch, latent_dim
        kl_zinstance = torch.mean(zinstance_few.kl)  # mean over: n_boxes, batch, latent_dim
        kl_zwhere = torch.mean(zwhere_kl_few)  # mean over: n_boxes, batch, latent_dim
        kl_logit = torch.mean(kl_logit)  # mean over: batch

        kl_av = kl_zbg + kl_zinstance + kl_zwhere + \
                torch.exp(-self.running_avarage_kl_logit) * kl_logit + \
                self.running_avarage_kl_logit - self.running_avarage_kl_logit.detach()

        # 6. Note that I clamp in_place
        geco_mse_detached = self.geco_mse.data.clamp_(min=0.1, max=0.9).detach()
        geco_ncell_detached = self.geco_ncell.data.clamp_(min=0.1, max=20.0).detach()
        geco_fgfraction_detached = self.geco_fgfraction.data.clamp_(min=0.1, max=20.0).detach()
        one_minus_geco_mse_detached = torch.ones_like(geco_mse_detached) - geco_mse_detached

        reg_av = cost_overlap + cost_bb_regression
        sparsity_av = geco_fgfraction_detached * f_sparsity + geco_ncell_detached * f_cell
        loss_vae = sparsity_av + geco_mse_detached * (mse_av + reg_av) + one_minus_geco_mse_detached * kl_av
        loss_geco = self.geco_fgfraction * g_sparsity.detach() + \
                    self.geco_ncell * g_cell.detach() + \
                    self.geco_mse * g_mse.detach()
        loss = loss_vae + loss_geco - loss_geco.detach()

        similarity_l, similarity_w = self.similarity_kernel_dpp.get_l_w()

        # add everything you want as long as there is one loss
        metric = MetricMiniBatch(loss=loss,
                                 mse_av=mse_av.detach().item(),
                                 kl_av=kl_av.detach().item(),
                                 cost_mask_overlap_av=cost_overlap.detach().item(),
                                 cost_bb_regression_av=cost_bb_regression.detach().item(),
                                 ncell_av=x_cell_av.detach().item(),
                                 fgfraction_av=torch.mean(mixing_fg).detach().item(),
                                 lambda_mse=geco_mse_detached.detach().item(),
                                 lambda_ncell=geco_ncell_detached.detach().item(),
                                 lambda_fgfraction=geco_fgfraction_detached.detach().item(),
                                 count_prediction=torch.sum(c_few, dim=0).detach().cpu().numpy(),
                                 wrong_examples=-1 * numpy.ones(1),
                                 accuracy=-1.0,
                                 similarity_l=similarity_l.detach().item(),
                                 similarity_w=similarity_w.detach().item(),
                                 kl_logit_av=self.running_avarage_kl_logit.exp().detach().item())

        return inference, metric
