import numpy
import torch
import torch.nn.functional as F

from .non_max_suppression import NonMaxSuppression
from .cropper_uncropper import Uncropper, Cropper
from .unet import UNet, PreProcessor
from .encoders_decoders import EncoderInstance, DecoderInstance, DecoderBackground, DecoderWhere
from .util import convert_to_box_list, invert_convert_to_box_list, compute_ranking, compute_average_in_box
from .util_ml import sample_and_kl_diagonal_normal, sample_c_grid, compute_logp_dpp, compute_logp_bernoulli, SimilarityKernel
from .namedtuple import Inference, BB, UNEToutput, ZZ, DIST, MetricMiniBatch, NmsOutput


def mixing_to_ideal_bb(mixing_kb1wh: torch.Tensor, pad_size: int):
    """ Given the mixing probabilities, it finds the :math:`mask=(mixing > 0.5)` and compute the coordinates of a
        bounding box which fits around the mask with a padding of size `attr:pad_size` pixels.

        Args:
            mixing_kb1wh: torch.Tensor of shape :math:`(K, B, 1, W, H)`
            pad_size: padding around the mask. If :attr:`pad_size` = 0 then the bounding box is body fitted

        Returns:
            The ideal bounding boxes in :class:`BB` of shape :math:`(K, B)`
    """

    # Compute the ideal Bounding boxes
    n_width, n_height = mixing_kb1wh.shape[-2:]
    ix_w1 = torch.arange(start=0,
                         end=n_width,
                         dtype=torch.int,
                         device=mixing_kb1wh.device).unsqueeze(-1)
    iy_1h = torch.arange(start=0,
                         end=n_height,
                         dtype=torch.int,
                         device=mixing_kb1wh.device).unsqueeze(-2)

    mask_kb1wh = (mixing_kb1wh > 0.5).int()
    # compute ideal x1,x3,y1,y3 of shape: n_box_few, batch_size
    ideal_x3_kb = torch.max(torch.flatten(mask_kb1wh * ix_w1, start_dim=-3), dim=-1)[0]
    ideal_y3_kb = torch.max(torch.flatten(mask_kb1wh * iy_1h, start_dim=-3), dim=-1)[0]
    ideal_x1_kb = n_width - torch.max(torch.flatten(mask_kb1wh * (n_width - ix_w1), start_dim=-3), dim=-1)[0]
    ideal_y1_kb = n_height - torch.max(torch.flatten(mask_kb1wh * (n_height - iy_1h), start_dim=-3), dim=-1)[0]
    ideal_x1_kb = (ideal_x1_kb - pad_size).clamp(min=0, max=n_width)
    ideal_y1_kb = (ideal_y1_kb - pad_size).clamp(min=0, max=n_height)
    ideal_x3_kb = (ideal_x3_kb + pad_size).clamp(min=0, max=n_width)
    ideal_y3_kb = (ideal_y3_kb + pad_size).clamp(min=0, max=n_height)

    # print("ideal_y3_kb.shape ->", ideal_y3_kb.shape)
    return BB(bx=0.5*(ideal_x1_kb+ideal_x3_kb),
              by=0.5*(ideal_y1_kb+ideal_y3_kb),
              bw=ideal_x3_kb + ideal_x1_kb,
              bh=ideal_y1_kb + ideal_x1_kb)


######    # assuming that bx and bw are fixed. What should bw and bh be?
######    size_obj_min = self.input_img_dict["size_object_min"]
######    size_obj_max = self.input_img_dict["size_object_max"]
######    bw_target = torch.max(ideal_x3 - inference.sample_bb.bx,
######                          inference.sample_bb.bx - ideal_x1).clamp(min=size_obj_min, max=size_obj_max)


class InferenceAndGeneration(torch.nn.Module):

    def __init__(self, params):
        super().__init__()

        # variables
        self.size_min = params["input_image"]["range_object_size"][0]
        self.size_max = params["input_image"]["range_object_size"][1]
        self.glimpse_size = params["architecture"]["glimpse_size"]
        self.mask_overlap_penalty_strength = params["loss"]["mask_overlap_penalty_strength"]
        self.pad_size_bb = params["loss"]["bounding_box_regression_padding"]
        self.bb_regression_penalty_strength = params["loss"]["bounding_box_regression_penalty_strength"]

        # modules
        self.similarity_kernel_dpp = SimilarityKernel(length_scale=params["input_image"]["similarity_DPP_l"],
                                                      weight=params["input_image"]["similarity_DPP_w"],
                                                      length_scale_min_max=params["input_image"]["similarity_DPP_l_min_max"],
                                                      weight_min_max=params["input_image"]["similarity_DPP_w_min_max"])

        self.unet: UNet = UNet(params)
        self.preprocessor: PreProcessor = PreProcessor(params)

        # Decoders
        self.decoder_zbg: DecoderBackground = DecoderBackground(dim_z=params["architecture"]["dim_zbg"],
                                                                ch_out=params["architecture"]["n_ch_img"])

        self.decoder_zwhere: DecoderWhere = DecoderWhere(dim_z=params["architecture"]["dim_zwhere"])

        self.decoder_zinstance: DecoderInstance = DecoderInstance(size=params["architecture"]["glimpse_size"],
                                                                  dim_z=params["architecture"]["dim_zinstance"],
                                                                  ch_out=params["architecture"]["n_ch_img"] + 1)
        # Encoders
        self.encoder_zinstance: EncoderInstance = EncoderInstance(size=params["architecture"]["glimpse_size"],
                                                                  ch_in=params["architecture"]["n_ch_output_features"],
                                                                  dim_z=params["architecture"]["dim_zinstance"])

        # Parameters
        self.loss_dict = params["loss"]
        one = torch.ones(1, dtype=torch.float)

        self.sigma_mse_bg = torch.nn.Parameter(data=self.loss_dict["geco_mse_target"]*one[..., None, None],
                                               requires_grad=False)
        self.sigma_mse_fg = torch.nn.Parameter(data=self.loss_dict["geco_mse_target"]*one[..., None, None],
                                               requires_grad=False)
        self.running_avarage_kl_logit = torch.nn.Parameter(data=one, requires_grad=True)

        self.geco_loglambda_fgfraction = torch.nn.Parameter(data=2*one, requires_grad=True)
        self.geco_loglambda_ncell = torch.nn.Parameter(data=one, requires_grad=True)
        self.geco_loglambda_mse = torch.nn.Parameter(data=one, requires_grad=True)

        self.max_loglambda_mse = numpy.log(self.loss_dict["geco_lambda_mse_max"])
        self.max_loglambda_fgfraction = numpy.log(self.loss_dict["geco_lambda_fgfraction_max"])
        self.max_loglambda_ncell = numpy.log(self.loss_dict["geco_lambda_ncell_max"])

        self.target_fgfraction_min = min(self.loss_dict["geco_fgfraction_target"])
        self.target_fgfraction_max = max(self.loss_dict["geco_fgfraction_target"])
        self.target_ncell_min = min(self.loss_dict["geco_ncell_target"])
        self.target_ncell_max = max(self.loss_dict["geco_ncell_target"])
        self.target_mse_min = 0.0
        self.target_mse_max = 1.0

    @staticmethod
    def _compute_logit_corrected(logit_praw: torch.Tensor,
                                 p_corr: torch.Tensor,
                                 a: float):
        """ In log space computes the probability correction p = (1-a) * p_raw + a * p_corr
            It returns the logit of the corrected probability
        """
        log_praw = F.logsigmoid(logit_praw)
        log_1_m_praw = F.logsigmoid(-logit_praw)
        log_a = torch.tensor(a, device=logit_praw.device, dtype=logit_praw.dtype).log()
        log_1_m_a = torch.tensor(1-a, device=logit_praw.device, dtype=logit_praw.dtype).log()
        log_p_corrected = torch.logaddexp(log_praw + log_1_m_a, torch.log(p_corr) + log_a)
        log_1_m_p_corrected = torch.logaddexp(log_1_m_praw + log_1_m_a, torch.log(1-p_corr) + log_a)
        logit_corrected = log_p_corrected - log_1_m_p_corrected
        return logit_corrected

    def forward(self, imgs_bcwh: torch.Tensor,
                generate_synthetic_data: bool,
                prob_corr_factor: float,
                iom_threshold: float,
                k_objects_max: int,
                topk_only: bool,
                noisy_sampling: bool) -> (Inference, MetricMiniBatch):

        # 1. UNET
        imgs_preprocessed_bcwh = self.preprocessor.forward(imgs_bcwh, verbose=False)
        unet_output: UNEToutput = self.unet.forward(imgs_preprocessed_bcwh, verbose=False)

        # background
        # Todo: replace the background block with a VQ-VAE.
        # Note that I will not care about generating background from scratch. I only care about encoding the background
        # in a low dimensional representation of shape: size x size x K
        # http://ameroyer.github.io/projects/2019/08/20/VQVAE.html
        zbg: DIST = sample_and_kl_diagonal_normal(posterior_mu=unet_output.zbg.mu,
                                                  posterior_std=unet_output.zbg.std,
                                                  prior_mu=torch.zeros_like(unet_output.zbg.mu),
                                                  prior_std=torch.ones_like(unet_output.zbg.std),
                                                  noisy_sampling=noisy_sampling,
                                                  sample_from_prior=generate_synthetic_data)
        out_background_bcwh = self.decoder_zbg(z=zbg.sample, high_resolution=(imgs_bcwh.shape[-2], imgs_bcwh.shape[-1]))


        # bounding boxes
        zwhere_grid: DIST = sample_and_kl_diagonal_normal(posterior_mu=unet_output.zwhere.mu,
                                                          posterior_std=unet_output.zwhere.std,
                                                          prior_mu=torch.zeros_like(unet_output.zwhere.mu),
                                                          prior_std=torch.ones_like(unet_output.zwhere.std),
                                                          noisy_sampling=noisy_sampling,
                                                          sample_from_prior=generate_synthetic_data)

        bounding_box_nb: BB = self.decoder_zwhere(z=zwhere_grid.sample,
                                                  width_raw_image=imgs_bcwh.shape[-2],
                                                  height_raw_image=imgs_bcwh.shape[-1],
                                                  min_box_size=self.size_min,
                                                  max_box_size=self.size_max)
        # print("bounding_box_nb.bx.shape ->", bounding_box_nb.bx.shape)

        # Correct probability if necessary
        with torch.no_grad():
            if prob_corr_factor > 0:
                av_intensity_nb = compute_average_in_box((imgs_bcwh - out_background_bcwh).abs(), bounding_box_nb)
                ranking_nb = compute_ranking(av_intensity_nb)  # It is in [0,n-1]
                tmp_nb = (ranking_nb + 1).float() / (ranking_nb.shape[-2]+1)  # strictly inside (0,1) range
                p_corr_b1wh = invert_convert_to_box_list(tmp_nb.pow(10).unsqueeze(-1),
                                                         original_width=unet_output.logit.shape[-2],
                                                         original_height=unet_output.logit.shape[-1])
            else:
                p_corr_b1wh = 0.5*torch.ones_like(unet_output.logit)
        # End of torch.no_grad
        logit_grid_corrected = self._compute_logit_corrected(logit_praw=unet_output.logit,
                                                             p_corr=p_corr_b1wh,
                                                             a=prob_corr_factor)

        # Sample the probability grid from prior or posterior
        similarity_kernel = self.similarity_kernel_dpp.forward(n_width=unet_output.logit.shape[-2],
                                                               n_height=unet_output.logit.shape[-1])

        c_grid_before_nms_b1wh = sample_c_grid(logit_grid=logit_grid_corrected,
                                               similarity_matrix=similarity_kernel,
                                               noisy_sampling=noisy_sampling,
                                               sample_from_prior=generate_synthetic_data)

        # NMS + top-K operation
        with torch.no_grad():
            score_nb = convert_to_box_list(c_grid_before_nms_b1wh + torch.sigmoid(logit_grid_corrected)).squeeze(-1)
            combined_topk_only = topk_only or generate_synthetic_data  # if generating from DPP do not do NMS
            nms_output: NmsOutput = NonMaxSuppression.compute_indices(score_nb=score_nb,
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

        # Compute the KL divergence between the DPP prior and the posterior.
        # Note that I use: c_grid_after_nms = c_grid_before_nms * mask_grid
        # This is because I am learning the prior. If I feed c_grid_before_nms which can have nearby points on,
        # then I can learn very bad parameters for my prior.
        c_grid_after_nms_b1wh = c_grid_before_nms_b1wh * mask_grid_b1wh
        c_grid_logp_prior_b = compute_logp_dpp(c_grid=c_grid_after_nms_b1wh.detach(),
                                               similarity_matrix=similarity_kernel)
        c_grid_logp_posterior_b = compute_logp_bernoulli(c_grid=c_grid_after_nms_b1wh.detach(),
                                                         logit_grid=logit_grid_corrected)

        # this will make adjust DPP and keep entropy of posterior
        kl_logit_b = c_grid_logp_posterior_b - c_grid_logp_prior_b

        c_kb = torch.gather(convert_to_box_list(c_grid_before_nms_b1wh).squeeze(-1), dim=0, index=nms_output.indices_kb)
        bounding_box_kb: BB = BB(bx=torch.gather(bounding_box_nb.bx, dim=0, index=nms_output.indices_kb),
                                 by=torch.gather(bounding_box_nb.by, dim=0, index=nms_output.indices_kb),
                                 bw=torch.gather(bounding_box_nb.bw, dim=0, index=nms_output.indices_kb),
                                 bh=torch.gather(bounding_box_nb.bh, dim=0, index=nms_output.indices_kb))

        zwhere_kl_nbz = convert_to_box_list(zwhere_grid.kl)
        indices_kbz = nms_output.indices_kb.unsqueeze(-1).expand(-1, -1, zwhere_kl_nbz.shape[-1])
        zwhere_kl_kbz = torch.gather(zwhere_kl_nbz, dim=0, index=indices_kbz)

        # Crop the unet_features according to the selected boxes
        unet_features_kbcwh = unet_output.features.unsqueeze(0).expand(nms_output.indices_kb.shape[0], -1, -1, -1, -1)
        cropped_feature_kbcwh = Cropper.crop(bounding_box=bounding_box_kb,
                                             big_stuff=unet_features_kbcwh,
                                             width_small=self.glimpse_size,
                                             height_small=self.glimpse_size)
        # print("cropped_feature_kbcwh.shape -->", cropped_feature_kbcwh.shape)

        # Encode, sample zinstance and decode to big images and big weights
        zinstance_posterior: ZZ = self.encoder_zinstance.forward(cropped_feature_kbcwh)
        zinstance_kbz: DIST = sample_and_kl_diagonal_normal(posterior_mu=zinstance_posterior.mu,
                                                            posterior_std=zinstance_posterior.std,
                                                            prior_mu=torch.zeros_like(zinstance_posterior.mu),
                                                            prior_std=torch.ones_like(zinstance_posterior.std),
                                                            noisy_sampling=noisy_sampling,
                                                            sample_from_prior=generate_synthetic_data)
        cropped_stuff_kbcwh = self.decoder_zinstance.forward(zinstance_kbz.sample)  # stuff between 0 and 1
        uncropped_stuff_kbcwh = Uncropper.uncrop(bounding_box=bounding_box_kb,
                                                 small_stuff=cropped_stuff_kbcwh,
                                                 width_big=imgs_bcwh.shape[-2],
                                                 height_big=imgs_bcwh.shape[-1])
        out_weights_kb1wh, out_img_kbcwh = torch.split(uncropped_stuff_kbcwh,
                                                       split_size_or_sections=(1, uncropped_stuff_kbcwh.shape[-3]-1),
                                                       dim=-3)

        # Compute the mixing
        out_mask_kb1wh = torch.sigmoid(out_weights_kb1wh)
        c_times_mask_kb1wh = out_mask_kb1wh * c_kb[..., None, None, None]  # this is strictly smaller than 1
        mixing_kb1wh = c_times_mask_kb1wh / c_times_mask_kb1wh.sum(dim=-5).clamp(min=1.0)  # softplus-like function

        # Compute the mask_overlap
        # A = (x1+x2+x3)^2 = x1^2 + x2^2 + x3^2 + 2 x1*x2 + 2 x1*x3 + 2 x2*x3
        # Therefore sum_{i \ne j} x_i x_j = x1*x2 + x1*x3 + x2*x3 = 0.5 * [(sum xi)^2 - (sum xi^2)]
        sum_x = c_times_mask_kb1wh.sum(dim=-5)  # sum over boxes first
        sum_x2 = c_times_mask_kb1wh.pow(2).sum(dim=-5)  # square first and sum over boxes later
        mask_overlap = 0.5 * (sum_x.pow(2) - sum_x2).clamp(min=0).sum()
        cost_mask_overlap = self.mask_overlap_penalty_strength * mask_overlap

        # Compute ideal box
        # TODO: compute the L1 or L2 loss between the ideal and actual bounding box
        with torch.no_grad():
            bb_ideal_kb = mixing_to_ideal_bb(mixing_kb1wh, pad_size=self.pad_size_bb)
        ######    bh_target = torch.max(ideal_y3 - inference.sample_bb.by,
        ######                          inference.sample_bb.by - ideal_y1).clamp(min=size_obj_min, max=size_obj_max)
        cost_bb_regression = self.bb_regression_penalty_strength * torch.zeros_like(cost_mask_overlap)

        # Compute MSE
        mixing_fg_b1wh = torch.sum(mixing_kb1wh, dim=-5)
        mixing_bg_b1wh = torch.ones_like(mixing_fg_b1wh) - mixing_fg_b1wh
        mse_bg_bcwh = ((out_background_bcwh - imgs_bcwh)/self.sigma_mse_bg).pow(2)
        mse_fg_kbcwh = ((out_img_kbcwh - imgs_bcwh)/self.sigma_mse_fg).pow(2)
        mse_av = ((mixing_kb1wh * mse_fg_kbcwh).sum(dim=-5) + mixing_bg_b1wh * mse_bg_bcwh).mean()

        # Compute KL (mean over batch, latent_dim, sum over n_boxes)
        c_detach_kb1 = c_kb.unsqueeze(-1).detach()
        kl_background = torch.mean(zbg.kl)  # mean over batch, latent_dim
        kl_instance = torch.mean(c_detach_kb1 * zinstance_kbz.kl) * c_detach_kb1.shape[0]
        kl_where = torch.mean(c_detach_kb1 * zwhere_kl_kbz) * c_detach_kb1.shape[0]
        kl_logit = torch.mean(kl_logit_b)
        kl_av = kl_background + kl_instance + kl_where + \
                torch.exp(-self.running_avarage_kl_logit) * kl_logit + \
                self.running_avarage_kl_logit - self.running_avarage_kl_logit.detach()

        # Clamp log lambda in place if necessary
        self.geco_loglambda_mse.data.clamp_(max=self.max_loglambda_mse)
        self.geco_loglambda_ncell.data.clamp_(max=self.max_loglambda_ncell)
        self.geco_loglambda_fgfraction.data.clamp_(max=self.max_loglambda_fgfraction)

        with torch.no_grad():
            batch_size = c_grid_after_nms_b1wh.shape[0]
            # If in range log_lambda should decrease if out of range it should increase
            ncell_av = c_grid_after_nms_b1wh.sum() / batch_size
            v_ncell = min(ncell_av - self.target_ncell_min, self.target_ncell_max - ncell_av)

            fgfraction_av = mixing_fg_b1wh.mean()
            v_fgfraction = min(fgfraction_av - self.target_fgfraction_min, self.target_fgfraction_max - fgfraction_av)

            v_mse = min(mse_av - self.target_mse_min, self.target_mse_max - mse_av)

            # Get lambda from log_lambda
            lambda_mse = self.geco_loglambda_mse.exp() * torch.sign(mse_av - self.target_mse_min)
            lambda_fgfraction = self.geco_loglambda_fgfraction.exp() * torch.sign(fgfraction_av - self.target_mse_min)
            lambda_ncell = self.geco_loglambda_ncell.exp() * torch.sign(ncell_av - self.target_ncell_min)

        loss_vae = kl_av + \
                   lambda_ncell * torch.sum(c_grid_before_nms_b1wh) / batch_size + \
                   lambda_fgfraction * torch.sum(mixing_fg_b1wh) / batch_size + \
                   lambda_mse * (mse_av + cost_bb_regression + cost_mask_overlap)

        loss_geco = v_mse * self.geco_loglambda_mse + \
                    v_fgfraction * self.geco_loglambda_fgfraction + \
                    v_ncell * self.geco_loglambda_ncell

        # All metrics of interest to monitor the behaviour of the model during training
        # One special element is loss
        # add everything you want as long as there is one loss
        similarity_l, similarity_w = self.similarity_kernel_dpp.get_l_w()
        metrics = MetricMiniBatch(loss=loss_geco + loss_vae,
                                  mse_av=mse_av.detach().item(),
                                  kl_av=kl_av.detach().item(),
                                  cost_mask_overlap_av=cost_mask_overlap.detach().item(),
                                  cost_bb_regression_av=cost_bb_regression.detach().item(),
                                  ncell_av=ncell_av.detach().item(),
                                  fgfraction_av=fgfraction_av.detach().item(),
                                  # geco
                                  lambda_mse=lambda_mse.detach().item(),
                                  lambda_ncell=lambda_ncell.detach().item(),
                                  lambda_fgfraction=lambda_fgfraction.detach().item(),
                                  # conting accuracy
                                  count_prediction=c_kb.sum(dim=0).detach().cpu().numpy(),
                                  wrong_examples=-1*numpy.ones(1),
                                  accuracy=-1.0,
                                  # similarity
                                  similarity_l=similarity_l.detach().item(),
                                  similarity_w=similarity_w.detach().item(),
                                  kl_logit_av=self.running_avarage_kl_logit.detach().item())

        inference = Inference(logit_grid=logit_grid_corrected.detach(),
                              logit_grid_unet=unet_output.logit.detach(),
                              background_bcwh=out_background_bcwh.detach(),
                              mixing_kb1wh=mixing_kb1wh.detach(),
                              foreground_kbcwh=out_img_kbcwh.detach(),
                              # the sample of the 4 latent variables
                              sample_c_grid_before_nms=c_grid_before_nms_b1wh.detach(),
                              sample_c_grid_after_nms=c_grid_after_nms_b1wh.detach(),
                              sample_c_kb=c_kb.detach(),
                              sample_bb_kb=bounding_box_kb,
                              sample_bb_ideal_kb=bb_ideal_kb)

        return inference, metrics
