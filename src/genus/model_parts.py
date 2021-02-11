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


def optimal_bb_and_bb_regression_penalty(mixing_nb1wh: torch.Tensor,
                                         bounding_boxes_nb: BB,
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
        mask_nbwh = (mixing_nb1wh.squeeze(-3) > 0.5).int()
        mask_nbh = torch.max(mask_nbwh, dim=-2)[0]
        mask_nbw = torch.max(mask_nbwh, dim=-1)[0]
        mask_nb = torch.max(mask_nbw, dim=-1)[0]  # 0 if empty, 1 if non-empty

        plus_h = torch.arange(start=0, end=mask_nbh.shape[-1], step=1,
                              dtype=torch.float, device=mixing_nb1wh.device) + 1
        plus_w = torch.arange(start=0, end=mask_nbw.shape[-1], step=1,
                              dtype=torch.float, device=mixing_nb1wh.device) + 1
        minus_h = plus_h[-1] - plus_h + 1
        minus_w = plus_w[-1] - plus_w + 1

        # Find the coordinates of the bounding box
        ideal_x1_nb = (torch.argmax(mask_nbw * minus_w, dim=-1) - pad_size).clamp(min=0, max=mask_nbw.shape[-1]).float()
        ideal_x3_nb = (torch.argmax(mask_nbw * plus_w,  dim=-1) + pad_size).clamp(min=0, max=mask_nbw.shape[-1]).float()
        ideal_y1_nb = (torch.argmax(mask_nbh * minus_h, dim=-1) - pad_size).clamp(min=0, max=mask_nbh.shape[-1]).float()
        ideal_y3_nb = (torch.argmax(mask_nbh * plus_h,  dim=-1) + pad_size).clamp(min=0, max=mask_nbh.shape[-1]).float()

        # If the box is empty, do a special treatment, i.e. make them the smallest possible size
        empty_nb = (mask_nb == 0)
        ideal_x1_nb[empty_nb] = bounding_boxes_nb.bx[empty_nb] - 0.5 * min_box_size
        ideal_x3_nb[empty_nb] = bounding_boxes_nb.bx[empty_nb] + 0.5 * min_box_size
        ideal_y1_nb[empty_nb] = bounding_boxes_nb.by[empty_nb] - 0.5 * min_box_size
        ideal_y3_nb[empty_nb] = bounding_boxes_nb.by[empty_nb] + 0.5 * min_box_size

        # Compute the box coordinates (note the clamping of bw and bh)
        ideal_bx_nb = 0.5 * (ideal_x3_nb + ideal_x1_nb)
        ideal_by_nb = 0.5 * (ideal_y3_nb + ideal_y1_nb)
        ideal_bw_nb = (ideal_x3_nb - ideal_x1_nb).clamp(min=min_box_size, max=max_box_size)
        ideal_bh_nb = (ideal_y3_nb - ideal_y1_nb).clamp(min=min_box_size, max=max_box_size)

        # print("DEBUG min, max ->", min_box_size, max_box_size)
        # print("DEBUG input ->", bounding_boxes_nb.bx[0, 0], bounding_boxes_nb.by[0, 0],
        #       bounding_boxes_nb.bw[0, 0], bounding_boxes_nb.bh[0, 0], empty_nb[0, 0])
        # print("DEBUG ideal ->", ideal_bx_nb[0, 0], ideal_by_nb[0, 0],
        #       ideal_bw_nb[0, 0], ideal_bh_nb[0, 0], empty_nb[0, 0])

    # Outside the torch.no_grad() compute the regression cost
    cost_bb_regression = torch.abs(ideal_bx_nb - bounding_boxes_nb.bx) + \
                         torch.abs(ideal_by_nb - bounding_boxes_nb.by) + \
                         torch.abs(ideal_bw_nb - bounding_boxes_nb.bw) + \
                         torch.abs(ideal_bh_nb - bounding_boxes_nb.bh)

    return BB(bx=ideal_bx_nb, by=ideal_by_nb, bw=ideal_bw_nb, bh=ideal_bh_nb), cost_bb_regression


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

        # Variables
        self.min_box_size = config["input_image"]["range_object_size"][0]
        self.max_box_size = config["input_image"]["range_object_size"][1]
        self.glimpse_size = config["architecture"]["glimpse_size"]

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


        # Extra penalties
        self.bb_regression_strength = config["loss"]["bounding_box_regression_penalty_strength"]
        self.mask_overlap_strength_base = config["loss"]["mask_overlap_penalty_strength"]
        self.mask_overlap_type = config["loss"]["mask_overlap_penalty_type"]
        self.pad_size_bb = config["loss"]["bounding_box_regression_padding"]

        # Geco values
        self.sigma_fg = torch.nn.Parameter(data=torch.tensor(config["loss"]["geco_mse_target"],
                                                             dtype=torch.float)[..., None, None], requires_grad=False)
        self.sigma_bg = torch.nn.Parameter(data=torch.tensor(config["loss"]["geco_mse_target"],
                                                             dtype=torch.float)[..., None, None], requires_grad=False)

        self.geco_target_mse_min = 0.0
        self.geco_target_mse_max = 1.0
        self.geco_target_ncell_min_base = config["loss"]["geco_ncell_target"][0]
        self.geco_target_ncell_max_base = config["loss"]["geco_ncell_target"][1]
        self.geco_target_fgfraction_min_base = config["loss"]["geco_fgfraction_target"][0]
        self.geco_target_fgfraction_max_base = config["loss"]["geco_fgfraction_target"][1]

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
        self.geco_loglambda_mse = torch.nn.Parameter(data=torch.tensor(0.0, dtype=torch.float),
                                                     requires_grad=True)

        # The some values are changed during pretraining
        self.mask_overlap_strength = None
        self.geco_target_ncell_min = None
        self.geco_target_ncell_max = None
        self.geco_target_fgfraction_min = None
        self.geco_target_fgfraction_max = None

        # Raw image parameters
        self.running_avarage_kl_logit = torch.nn.Parameter(data=4 * torch.ones(1, dtype=torch.float),
                                                           requires_grad=True)



    @staticmethod
    def _compute_logit_corrected(logit_praw: torch.Tensor,
                                 p_correction: torch.Tensor,
                                 prob_corr_factor: float):
        """
        In log space computes the probability correction:
        1. p = (1-a) * p_raw + a * p_corr
        2. (1-p) = (1-a) * (1-p_raw) + a * (1-p_corr)
        3. logit_p = log_p - log_1_m_p
        It returns the logit of the corrected probability
        """

        log_praw = F.logsigmoid(logit_praw)
        log_1_m_praw = F.logsigmoid(-logit_praw)
        log_a = torch.tensor(prob_corr_factor, device=logit_praw.device, dtype=logit_praw.dtype).log()
        log_1_m_a = torch.tensor(1 - prob_corr_factor, device=logit_praw.device, dtype=logit_praw.dtype).log()
        log_p_corrected = torch.logaddexp(log_praw + log_1_m_a, torch.log(p_correction) + log_a)
        log_1_m_p_corrected = torch.logaddexp(log_1_m_praw + log_1_m_a, torch.log1p(-p_correction) + log_a)
        logit_corrected = log_p_corrected - log_1_m_p_corrected
        return logit_corrected

    @staticmethod
    def mse(output: torch.tensor, target: torch.tensor, sigma: torch.tensor) -> torch.Tensor:
        return ((output - target) / sigma).pow(2)

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

        bounding_box_nb: BB = tgrid_to_bb(t_grid=torch.sigmoid(self.decoder_zwhere(zwhere_grid.sample)),
                                          width_input_image=width_raw_image,
                                          height_input_image=height_raw_image,
                                          min_box_size=self.min_box_size,
                                          max_box_size=self.max_box_size)

        # During warm-up phase:
        # 1. the target for n_cell and fg_fraction is higher
        # 2. the mask overlap penalty is set to zero
        if (prob_corr_factor > 0) and (prob_corr_factor <= 1.0):
            # If in the warm-up phase
            n_all = torch.numel(unet_output.logit[0, 0])
            self.geco_target_ncell_min = prob_corr_factor * (0.5 * n_all) + \
                                         (1 - prob_corr_factor) * self.geco_target_ncell_min_base
            self.geco_target_ncell_max = prob_corr_factor * (0.75 * n_all) + \
                                         (1 - prob_corr_factor) * self.geco_target_ncell_max_base
            self.geco_target_fgfraction_min = prob_corr_factor * 0.5 + \
                                              (1 - prob_corr_factor) * self.geco_target_fgfraction_min_base
            self.geco_target_fgfraction_max = prob_corr_factor * 0.75 + \
                                              (1 - prob_corr_factor) * self.geco_target_fgfraction_max_base
            self.mask_overlap_strength = prob_corr_factor * 0.0 + \
                                         (1- prob_corr_factor) * self.mask_overlap_strength_base
        else:
            # If NOT in the warm-up phase
            self.geco_target_ncell_min = self.geco_target_ncell_min_base
            self.geco_target_ncell_max = self.geco_target_ncell_max_base
            self.geco_target_fgfraction_min = self.geco_target_fgfraction_min_base
            self.geco_target_fgfraction_max = self.geco_target_fgfraction_max_base
            self.mask_overlap_strength = self.mask_overlap_strength_base
        #print("prob_corr_factor ---------->", prob_corr_factor)
        #print("target ncell ---------->", self.geco_target_ncell_min, self.geco_target_ncell_max)
        #print("target fgfraction ----->", self.geco_target_fgfraction_min, self.geco_target_fgfraction_max)
        #print("mask_overlap_strength ->", self.mask_overlap_strength)

        logit_warming_loss = torch.zeros(1, dtype=imgs_bcwh.dtype, device=imgs_bcwh.device)


#            with torch.no_grad():
#                av_intensity_nb = compute_average_in_box((imgs_bcwh - out_background_bcwh).abs(), bounding_box_nb)
#                ranking_nb = compute_ranking(av_intensity_nb)  # It is in [0,n-1]
#                tmp_nb = (ranking_nb + 1).float() / (ranking_nb.shape[-2]+1)  # strictly inside (0,1) range
#                p_corr_b1wh = invert_convert_to_box_list(tmp_nb.pow(10).unsqueeze(-1),
#                                                         original_width=unet_output.logit.shape[-2],
#                                                         original_height=unet_output.logit.shape[-1])
#            # outside torch.no_grad
#            logit_grid_corrected = InferenceAndGeneration._compute_logit_corrected(logit_praw=unet_output.logit,
#                                                                                   p_correction=p_corr_b1wh,
#                                                                                   prob_corr_factor=prob_corr_factor)
#            prob_grid_corrected = torch.sigmoid(logit_grid_corrected)
#            logit_warming_loss = torch.zeros(1, dtype=imgs_bcwh.dtype, device=imgs_bcwh.device)
#            #logit_warming_loss = (logit_grid_corrected.detach() - unet_output.logit).pow(2).sum()
#        elif prob_corr_factor == 0:
#            p_corr_b1wh = 0.5 * torch.ones_like(unet_output.logit)
#            logit_grid_corrected = unet_output.logit
#            logit_warming_loss = torch.zeros(1, dtype=imgs_bcwh.dtype, device=imgs_bcwh.device)
#            prob_grid_corrected = torch.sigmoid(logit_grid_corrected)
#        else:
#            raise Exception("prob_corr_factor has an invalid value", prob_corr_factor)
#        print("logit_grid shape min,max -->", unet_output.logit.shape, torch.min(unet_output.logit).detach().item(),
#              torch.max(unet_output.logit).detach().item())

        logit_grid_corrected = unet_output.logit
        prob_grid_corrected = torch.sigmoid(logit_grid_corrected)

        # Sample the probability map from prior or posterior
        similarity_kernel = self.similarity_kernel_dpp.forward(n_width=logit_grid_corrected.shape[-2],
                                                               n_height=logit_grid_corrected.shape[-1])

        with torch.no_grad():
            c_grid_detached_b1wh = sample_c_grid(logit_grid=logit_grid_corrected,
                                                 similarity_matrix=similarity_kernel,
                                                 noisy_sampling=noisy_sampling,
                                                 sample_from_prior=generate_synthetic_data)

        # Compute KL divergence between the DPP prior and the posterior: KL = logp(c|logit) - logp(c|similarity)
        # The effect of this term should be:
        # 1. DECREASE logit where c=1, INCREASE logit where c=0 (i.e. make the posterior distribution more entropic)
        # 2. Make the DPP parameters adjust to the seen configurations
        c_grid_logp_prior_b = compute_logp_dpp(c_grid=c_grid_detached_b1wh.detach(),
                                               similarity_matrix=similarity_kernel)
        c_grid_logp_posterior_b = compute_logp_bernoulli(c_grid=c_grid_detached_b1wh.detach(),
                                                         logit_grid=unet_output.logit)
        prob_nb1 = convert_to_box_list(prob_grid_corrected)
        c_detached_nb1 = convert_to_box_list(c_grid_detached_b1wh)
        kl_logit_b = c_grid_logp_posterior_b - c_grid_logp_prior_b
        zwhere_kl_nbz = convert_to_box_list(zwhere_grid.kl)

        # 5. Crop the unet_features according to the selected boxes
        n_all, batch_size = bounding_box_nb.bx.shape
        unet_features_expanded = unet_output.features.expand(n_all, batch_size, -1, -1, -1)
        cropped_feature_map: torch.Tensor = Cropper.crop(bounding_box=bounding_box_nb,
                                                         big_stuff=unet_features_expanded,
                                                         width_small=self.glimpse_size,
                                                         height_small=self.glimpse_size)

        # 6. Encode, sample z and decode to big images and big weights
        zinstance_posterior: ZZ = self.encoder_zinstance.forward(cropped_feature_map)
        zinstance_dist: DIST = sample_and_kl_diagonal_normal(posterior_mu=zinstance_posterior.mu,
                                                             posterior_std=zinstance_posterior.std,
                                                             prior_mu=torch.zeros_like(zinstance_posterior.mu),
                                                             prior_std=torch.ones_like(zinstance_posterior.std),
                                                             noisy_sampling=noisy_sampling,
                                                             sample_from_prior=generate_synthetic_data)

        # It is important that the sigmoid is applied before uncropping on a zero-canvas so that mask is zero everywhere
        # except inside the bounding boxes
        small_stuff = torch.sigmoid(self.decoder_zinstance.forward(zinstance_dist.sample))
        big_stuff = Uncropper.uncrop(bounding_box=bounding_box_nb,
                                     small_stuff=small_stuff,
                                     width_big=width_raw_image,
                                     height_big=height_raw_image)  # shape: n_box, batch, ch, w, h
        out_img_nbcwh, out_mask_nb1wh = torch.split(big_stuff,
                                                    split_size_or_sections=(big_stuff.shape[-3] - 1, 1),
                                                    dim=-3)

        # 7. Compute the mixing
        # TODO: think about this with Mehrtash
        #   Note that mixing are multiplied by the max between max(p, c).
        #   Ideally I would like to multiply by c. However if c=0 I can not learn anything. Therefore use max(p,c).
        # c_differentiable_nb1 = prob_nb1 + (c_detached_nb1 - prob_nb1).detach()
        # c_differentiable_nb1 = prob_nb1
        c_differentiable_nb1 = prob_nb1 - prob_nb1.detach() + torch.max(prob_nb1, c_detached_nb1).detach()

        c_differentiable_times_mask_nb1wh = c_differentiable_nb1[..., None, None] * out_mask_nb1wh
        mixing_nb1wh = c_differentiable_times_mask_nb1wh / c_differentiable_times_mask_nb1wh.sum(dim=-5).clamp(min=1.0)
        mixing_fg_b1wh = mixing_nb1wh.sum(dim=-5)  # sum over boxes
        mixing_bg_b1wh = torch.ones_like(mixing_fg_b1wh) - mixing_fg_b1wh

        # 8. Compute the ideal bounding boxes
        bb_ideal_nb, bb_regression_nb = optimal_bb_and_bb_regression_penalty(mixing_nb1wh=mixing_nb1wh,
                                                                             bounding_boxes_nb=bounding_box_nb,
                                                                             pad_size=self.pad_size_bb,
                                                                             min_box_size=self.min_box_size,
                                                                             max_box_size=self.max_box_size)
        # TODO: should I multiply this by c_detached_nb or c_attached_nb?
        #   If I multiply by c_differentiable, the boxes which are centered on the object will be favored b/c lower
        #   bb_regression cost
        cost_bb_regression = self.bb_regression_strength * torch.sum(c_differentiable_nb1.squeeze(-1) *
                                                                     bb_regression_nb) / batch_size

        # 9. Compute the mask overlap penalty using c_detached so that this penalty changes
        #  the mask but not the probabilities
        # TODO: detach c when computing the overlap? Probably yes.
        if self.mask_overlap_type == 1:
            # APPROACH 1: Compute mask overlap penalty
            mask_overlap_v1 = c_differentiable_times_mask_nb1wh.sum(dim=-5).pow(2) - \
                              c_differentiable_times_mask_nb1wh.pow(2).sum(dim=-5)
            # print("DEBUG mask_overlap1 min,max", torch.min(mask_overlap_v1), torch.max(mask_overlap_v1))
            cost_overlap_tmp_v1 = torch.sum(mask_overlap_v1, dim=(-1, -2, -3))  # sum over ch, w, h
            cost_overlap_v1 = self.mask_overlap_strength * cost_overlap_tmp_v1.mean()  # mean over batch
            cost_overlap = cost_overlap_v1
        elif self.mask_overlap_type == 2:
            # APPROACH 2: Compute mask overlap
            mask_overlap_v2 = torch.sum(mixing_nb1wh * (torch.ones_like(mixing_nb1wh) - mixing_nb1wh), dim=-5)
            cost_overlap_tmp_v2 = torch.sum(mask_overlap_v2, dim=(-1, -2, -3))
            cost_overlap_v2 = self.mask_overlap_strength * cost_overlap_tmp_v2.mean()
            cost_overlap = cost_overlap_v2
        else:
            raise Exception("self.mask_overlap_type not valid")

        inference = Inference(logit_grid=logit_grid_corrected,
                              logit_grid_unet=unet_output.logit,
                              logit_grid_correction=unet_output.logit,
                              background_bcwh=out_background_bcwh,
                              foreground_kbcwh=out_img_nbcwh,
                              sum_c_times_mask_b1wh=c_differentiable_times_mask_nb1wh.sum(dim=-5),
                              mixing_kb1wh=mixing_nb1wh,
                              sample_c_grid_before_nms=c_grid_detached_b1wh,
                              sample_c_grid_after_nms=c_grid_detached_b1wh,
                              sample_c_kb=c_detached_nb1.squeeze(-1),
                              sample_bb_kb=bounding_box_nb,
                              sample_bb_ideal_kb=bb_ideal_nb)

        # ---------------------------------- #
        # Compute the metrics

        # 1. Observation model
        mse_fg_nbcwh = ((out_img_nbcwh - imgs_bcwh) / self.sigma_fg).pow(2)
        mse_bg_bcwh = ((out_background_bcwh - imgs_bcwh) / self.sigma_bg).pow(2)
        mse_av = ((mixing_nb1wh * mse_fg_nbcwh).sum(dim=-5) + mixing_bg_b1wh * mse_bg_bcwh).mean()

        # 2. KL divergence
        # Note that I compute the mean over batch, latent_dimensions and n_object.
        # This means that latent_dim can effectively control the complexity of the reconstruction,
        # i.e. more latent more capacity.
        # TODO: kl should act only on filled boxes, i.e. c=1
        kl_zbg = torch.mean(zbg.kl)  # mean over: batch, latent_dim
        kl_zinstance = torch.sum(zinstance_dist.kl * c_detached_nb1) / (c_detached_nb1.sum() * zinstance_dist.kl.shape[-3])
        kl_zwhere = torch.sum(zwhere_kl_nbz * c_detached_nb1) / (c_detached_nb1.sum() * zwhere_kl_nbz.shape[-3])
        kl_logit = torch.zeros_like(kl_zbg)  #  torch.mean(kl_logit_b)  # mean over: batch
        kl_av = kl_zbg + kl_zinstance + kl_zwhere
        #+ \
        #        torch.exp(-self.running_avarage_kl_logit) * kl_logit + \
        #        self.running_avarage_kl_logit - self.running_avarage_kl_logit.detach()

        with torch.no_grad():
            ncell_av = torch.sum(c_detached_nb1) / batch_size
            fgfraction_av = torch.mean(mixing_fg_b1wh)

            mse_in_range = float((mse_av > self.geco_target_mse_min) &
                                 (mse_av < self.geco_target_mse_max))
            ncell_in_range = float((ncell_av > self.geco_target_ncell_min) &
                                   (ncell_av < self.geco_target_ncell_max))
            fgfraction_in_range = float((fgfraction_av > self.geco_target_fgfraction_min) &
                                        (fgfraction_av < self.geco_target_fgfraction_max))
            # print("debug in_range, mse, ncell, fgfraction", mse_in_range, ncell_in_range, fgfraction_in_range)
            # print("debug loglambda, mse, ncell, fgfraction", self.geco_loglambda_mse,
            #      self.geco_loglambda_ncell, self.geco_loglambda_fgfraction)

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
        loss_geco = self.geco_loglambda_mse * (2.0 * mse_in_range - 1.0) + \
                    self.geco_loglambda_fgfraction * (2.0 * fgfraction_in_range - 1.0) + \
                    self.geco_loglambda_ncell * (2.0 * ncell_in_range - 1.0)

        # TODO:
        #  should lambda_ncell act on all probabilities or only the selected ones? -> ALL
        #  what about acting on the underlying logits? -> GOOD IDEA
        #  should lambda_fgfraction act on small_mask or large_mask? LARGE otherwise it can not reach desired fg_fraction
        #  change prior to keep the probabilities close to 0.5?
        loss_vae = cost_overlap + cost_bb_regression + kl_av + \
                   lambda_mse.detach() * mse_av + \
                   lambda_ncell.detach() * torch.mean(logit_grid_corrected) + \
                   lambda_fgfraction.detach() * torch.sum(out_mask_nb1wh) / (batch_size * n_all)

        loss = loss_vae + loss_geco - loss_geco.detach()

        similarity_l, similarity_w = self.similarity_kernel_dpp.get_l_w()

        # add everything you want as long as there is one loss
        metric = MetricMiniBatch(loss=loss,
                                 logit_warming_loss=logit_warming_loss.detach().item(),
                                 mse_av=mse_av.detach().item(),
                                 kl_av=kl_av.detach().item(),
                                 cost_mask_overlap_av=cost_overlap.detach().item(),
                                 cost_bb_regression_av=cost_bb_regression.detach().item(),
                                 ncell_av=ncell_av.detach().item(),
                                 fgfraction_av=fgfraction_av.detach().item(),
                                 lambda_mse=lambda_mse.detach().item(),
                                 lambda_ncell=lambda_ncell.detach().item(),
                                 lambda_fgfraction=lambda_fgfraction.detach().item(),
                                 count_prediction=torch.sum(c_detached_nb1.squeeze(-1), dim=0).detach().cpu().numpy(),
                                 wrong_examples=-1 * numpy.ones(1),
                                 accuracy=-1.0,
                                 similarity_l=similarity_l.detach().item(),
                                 similarity_w=similarity_w.detach().item(),
                                 kl_logit_av=self.running_avarage_kl_logit.exp().detach().item())

        return inference, metric
