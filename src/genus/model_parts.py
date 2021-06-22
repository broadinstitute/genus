import torch
from typing import Union, Tuple
import numpy
import torch.nn.functional as F
from .cropper_uncropper import Uncropper, Cropper
from .unet import UnetSPACE
from .conv import DecoderBg, EncoderInstanceSPACE, DecoderInstanceSPACE
#from .unet import UNet, UNetNew, UnetSPACE
#from .conv import EncoderInstance, DecoderInstance, DecoderBg, DecoderWhere, EncoderInstanceSPACE, DecoderInstanceSPACE
from .util import convert_to_box_list, invert_convert_to_box_list, compute_average_in_box, compute_ranking
from .util_ml import compute_entropy_bernoulli, compute_logp_bernoulli, Grid_DPP, sample_and_kl_diagonal_normal, MovingAverageCalculator
from .namedtuple import Inference, NmsOutput, BB, UNEToutput, MetricMiniBatch, DIST, TT, GECO
from .non_max_suppression import NonMaxSuppression

def softmax_with_indicator(indicator: torch.Tensor, weight: torch.Tensor, dim: int, add_one_to_denominator: bool):
    """
    If add_one_to_denominator==True adds one to the denominator. Values will then sum to less than one.
    This is useful if you want to get the complementary (i.e. the background mixing probability in our case)

    Args:
          indicator: torch.Tensor in the range [0,1]
          weight: torch.Tensor in (-infty, infty)
          dim: int, dimension along which the input need to be normalized
          add_one_to_denominator: bool, whether to add 1 to the denominator

    Returns:
        the softmax, i.e. y_k = indicator_k * weight_k.exp() / [1 + sum_j indicator_j * weight_j.exp() ]

    Note:
        For numerical stability it multiplies both numerator and denominator by torch.exp(-A):
        y_k = indicator_k * (weight_k-A).exp() / [torch.exp(-A) + sum_j indicator_j * (weight_j-A).exp() ]
        where A = max(weight_k)
    """

    assert indicator.shape == weight.shape

    # Usual trick as in softmax to avoid exponentiating to a very large number (i.e. subtract the largest weight)
    weight_max = torch.max(weight).detach()
    num = indicator * torch.exp(weight - weight_max)
    if add_one_to_denominator:
        den = torch.sum(num, dim, keepdim=True) + torch.exp(-weight_max)
    else:
        den = torch.sum(num, dim, keepdim=True)
    return num / den


def compute_iou(bounding_boxes_a: BB, bounding_boxes_b: BB):
    """
    Compute the Intersection Over Union between pairs of bounding boxes (one from each set).

    Args:
        bounding_boxes_a: first set of bounding box
        bounding_boxes_b: second set of bounding box

    Returns:
        A tensor with the intersection over union between boxes
    """
    area_a = bounding_boxes_a.bw * bounding_boxes_a.bh
    x1_a = bounding_boxes_a.bx - 0.5*bounding_boxes_a.bw
    x3_a = bounding_boxes_a.bx + 0.5*bounding_boxes_a.bw
    y1_a = bounding_boxes_a.by - 0.5*bounding_boxes_a.bh
    y3_a = bounding_boxes_a.by + 0.5*bounding_boxes_a.bh

    area_b = bounding_boxes_b.bw * bounding_boxes_b.bh
    x1_b = bounding_boxes_b.bx - 0.5*bounding_boxes_b.bw
    x3_b = bounding_boxes_b.bx + 0.5*bounding_boxes_b.bw
    y1_b = bounding_boxes_b.by - 0.5*bounding_boxes_b.bh
    y3_b = bounding_boxes_b.by + 0.5*bounding_boxes_b.bh

    x1 = torch.max(x1_a, x1_b)
    x3 = torch.min(x3_a, x3_b)
    y1 = torch.max(y1_a, y1_b)
    y3 = torch.min(y3_a, y3_b)
    intersection = torch.clamp(x3 - x1, min=0) * torch.clamp(y3 - y1, min=0)

    iou = intersection / (area_a + area_b - intersection)
    return iou


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


def tt_to_bb(tt: TT,
             rawimage_size: Tuple[int, int],
             tgrid_size: Tuple[int, int],
             min_box_size: float,
             max_box_size: float) -> BB:
    """
    Transformation from :class:'TT' to :class:'BB'.

    Note:
        It is very important that MAX_DISPALCEMENT is larger than 0.5 meaning that multiple voxel can predict the same
        bounding box. This ensure a degree of redundancy and robustness
    """

    # Logic:
    # 1) ix + 0.5 is the center of the voxel
    # 2) dx is the displacement from the center of the voxel
    # 3) finally I convert to the large image size

    MAX_DISPLACEMENT = 1.5
    dx = MAX_DISPLACEMENT * 2.0 * (tt.tx - 0.5)  # value in [-MAX_DISPLACEMENT, MAX_DISPLACEMENT]
    dy = MAX_DISPLACEMENT * 2.0 * (tt.ty - 0.5)  # value in [-MAX_DISPLACEMENT, MAX_DISPLACEMENT]

    # values in (0.5 - MAX_DISPLACEMENT, rawimage_size - 0.5 + MAX_DISPLACEMENT)
    bx = (tt.ix + 0.5 + dx) * float(rawimage_size[0]) / tgrid_size[0]
    by = (tt.iy + 0.5 + dy) * float(rawimage_size[1]) / tgrid_size[1]

    # Bounding boxes in the range (min_box_size, max_box_size)
    bw = min_box_size + (max_box_size - min_box_size) * tt.tw
    bh = min_box_size + (max_box_size - min_box_size) * tt.th
    return BB(bx=bx, by=by, bw=bw, bh=bh)


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
    """ Dynamical parameter with value in [min_value, max_value]. """
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
            assert max_value >= min_value, "max_value {0} should be >= min_value {1}".format(max_value, min_value)

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
    def set(self, value: float):
        """ Helper function used to manually update the value of the hyperparamter. This should not be used.
            The hyperparameter should be updates by the optimizer via minimization of the associated loss function.
        """
        raw_value = inverse_linear_exp_activation(value) if self.linear_exp else value
        self.geco_raw_lambda.data.fill_(raw_value)
        self.geco_raw_lambda.clamp_(min=self.raw_min_value, max=self.raw_max_value)

    @torch.no_grad()
    def get(self):
        """ Helper function. Clamp the hyperparameter in the allowed range and returns a detached copy of it"""
        self.geco_raw_lambda.data.clamp_(min=self.raw_min_value, max=self.raw_max_value)
        if self.linear_exp:
            transformed_lambda = linear_exp_activation(self.geco_raw_lambda.data)
        else:
            transformed_lambda = self.geco_raw_lambda.data
        return transformed_lambda.detach()

    def forward(self, constraint) -> GECO:
        """
        Given a constraint it returns the current value of the hyperparameter and the loss term which when minimized
        leads to the desired change in the value of the hyperparameter

        Args:
            constraint: torch.Tensor with the value of the constraint.

        Returns:
            A container of type :class:'GECO'

        Note:
            If constraint < 0 (i.e. constraint is satisfied) the minimization of the returned loss leads to a
            decrease of the hyperparameter will be decreased
            If constraint > 0 (i.e. constraint is violated) the opposite is true
            If constraint == 0, the hyperparameter will not be changed
        """

        # This is first since it clamps lambda in the allowed range
        transformed_lambda = self.get()

        # Minimization of the loss function leads to:
        # Constraint satisfied -----> constraint < 0 ---> reduce the parameter
        # Constraint is violated ---> constraint > 0 ---> increase the parameter
        loss_geco = - self.geco_raw_lambda * constraint.detach()

        return GECO(loss=loss_geco, hyperparam=transformed_lambda)


class InferenceAndGeneration(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        # Variable related to multi-objective optimization
        self.multi_objective_optimization = config["loss"]["multi_objective_optimization"]
        self.moo_approximation = config["loss"]["multi_objective_approximation"]

        # variables
        self.GMM = config["loss"]["GMM_observation_model"]
        self.n_mc_samples = config["loss"]["n_mc_samples"]
        self.indicator_type = config["loss"]["indicator_type"]
        self.is_zero_background = config["input_image"]["is_zero_background"]

        self.mask_overlap_strength = config["loss"]["mask_overlap_penalty_strength"]
        self.box_overlap_strength = config["loss"]["box_overlap_penalty_strength"]

        self.min_box_size = config["input_image"]["range_object_size"][0]
        self.max_box_size = config["input_image"]["range_object_size"][1]
        self.glimpse_size = config["architecture"]["glimpse_size"]
        self.pad_size_bb = config["loss"]["ideal_box_padding"]

        # modules
        self.grid_dpp = Grid_DPP(length_scale=config["input_image"]["DPP_length"],
                                 weight=config["input_image"]["DPP_weight"],
                                 learnable_params=config["input_image"]["DPP_learnable_parameters"])

        self.unet: UnetSPACE = UnetSPACE(ch_in=config["input_image"]["ch_in"],
                                         ch_out=config["architecture"]["unet_ch_feature_map"],
                                         dim_zbg=config["architecture"]["zbg_dim"],
                                         dim_zwhere=config["architecture"]["zwhere_dim"],
                                         dim_logit=1)

        self.encoder_zinstance: EncoderInstanceSPACE = EncoderInstanceSPACE(
            glimpse_size=config["architecture"]["glimpse_size"],
            ch_in=config["architecture"]["unet_ch_feature_map"],
            dim_z=config["architecture"]["zinstance_dim"])

        self.decoder_zinstance: DecoderInstanceSPACE = DecoderInstanceSPACE(
            glimpse_size=config["architecture"]["glimpse_size"],
            dim_z=config["architecture"]["zinstance_dim"],
            ch_out=config["input_image"]["ch_in"] + 1)

        self.decoder_zwhere: torch.nn.Module = torch.nn.Conv2d(in_channels=config["architecture"]["zwhere_dim"],
                                                               out_channels=4, kernel_size=1, groups=4)

        self.zbg_dim = config["architecture"]["zbg_dim"]
        if not self.is_zero_background:
            raise NotImplementedError
            self.decoder_zbg: DecoderBg = DecoderBg(ch_in=config["architecture"]["zbg_dim"],
                                                    ch_out=config["input_image"]["ch_in"],
                                                    scale_factor=config["architecture"]["unet_scale_factor_background"])



        #######        if config["architecture"]["space_inspired"]:
        #######        elif config["architecture"]["pretrained_unet_from_mateuszbuda"]:
        #######            self.unet: UNetNew = UNetNew(pre_processor=None,
        #######                                         scale_factor_boundingboxes=config["architecture"]["unet_scale_factor_boundingboxes"],
#######                                         ch_in=config["input_image"]["ch_in"],
#######                                         ch_out=config["architecture"]["unet_ch_feature_map"],
#######                                         dim_zbg=config["architecture"]["zbg_dim"],
#######                                         dim_zwhere=config["architecture"]["zwhere_dim"],
#######                                         dim_logit=1,
#######                                         pretrained=True,
#######                                         partially_frozen=config["architecture"]["partially_frozen_unet"])
#######        else:
#######            self.unet: UNet = UNet(scale_factor_initial_layer=config["architecture"]["unet_scale_factor_initial_layer"],
#######                                   scale_factor_background=config["architecture"]["unet_scale_factor_background"],
#######                                   scale_factor_boundingboxes=config["architecture"]["unet_scale_factor_boundingboxes"],
#######                                   ch_in=config["input_image"]["ch_in"],
#######                                   ch_out=config["architecture"]["unet_ch_feature_map"],
#######                                   ch_before_first_maxpool=config["architecture"]["unet_ch_before_first_maxpool"],
#######                                   dim_zbg=config["architecture"]["zbg_dim"],
#######                                   dim_zwhere=config["architecture"]["zwhere_dim"],
#######                                   dim_logit=1)

####3        # Encoder-Decoders
####3        if config["architecture"]["space_inspired"]:
####3        else:
####3            self.encoder_zinstance: EncoderInstance = EncoderInstance(glimpse_size=config["architecture"]["glimpse_size"],
####3                                                                      ch_in=config["architecture"]["unet_ch_feature_map"],
####3                                                                      dim_z=config["architecture"]["zinstance_dim"])
####3            self.decoder_zinstance: DecoderInstance = DecoderInstance(glimpse_size=config["architecture"]["glimpse_size"],
####3                                                                      dim_z=config["architecture"]["zinstance_dim"],
####3                                                                      ch_out=config["input_image"]["ch_in"] + 1)


        # Quantities to compute the moving averages
        self.moving_average_calculator = MovingAverageCalculator(beta=config["optimizer"]["beta_moving_averages"],
                                                                 n_features=3)

        # Observation model
        self.sigma_mse = torch.nn.Parameter(data=torch.tensor(config["input_image"]["sigma_mse"],
                                                             dtype=torch.float)[..., None, None], requires_grad=False)

        # Dynamical parameter controlled by GECO are adjusted according to the following targets
        self.target_fgfraction_min = config["input_image"]["target_fgfraction_min_max"][0]
        self.target_fgfraction_max = config["input_image"]["target_fgfraction_min_max"][1]
        self.geco_fgfraction_min = GecoParameter(initial_value=1.0,
                                                 min_value=0.0,
                                                 max_value=config["loss"]["lambda_fgfraction_max"],
                                                 linear_exp=True)
        self.geco_fgfraction_max = GecoParameter(initial_value=1.0,
                                                 min_value=0.0,
                                                 max_value=config["loss"]["lambda_fgfraction_max"],
                                                 linear_exp=True)

        self.target_nobj_av_per_patch_min = config["input_image"]["target_nobj_av_per_patch_min_max"][0]
        self.target_nobj_av_per_patch_max = config["input_image"]["target_nobj_av_per_patch_min_max"][1]
        self.geco_nobj_min = GecoParameter(initial_value=1.0,
                                           min_value=0.0,
                                           max_value=config["loss"]["lambda_nobj_max"],
                                           linear_exp=True)
        self.geco_nobj_max = GecoParameter(initial_value=1.0,
                                           min_value=0.0,
                                           max_value=config["loss"]["lambda_nobj_max"],
                                           linear_exp=True)

        self.target_mse_for_annealing = config["input_image"]["target_mse_for_annealing"]
        self.geco_annealing = GecoParameter(initial_value=1.0, min_value=0.0, max_value=1.0, linear_exp=False)

        self.target_mse_fg = config["input_image"]["target_mse_fg"]
        self.geco_kl_fg = GecoParameter(initial_value=config["loss"]["lambda_kl_fg_min_max"][0],
                                        min_value=config["loss"]["lambda_kl_fg_min_max"][0],
                                        max_value=config["loss"]["lambda_kl_fg_min_max"][1],
                                        linear_exp=True)

        self.target_mse_bg = config["input_image"]["target_mse_bg"]
        self.geco_kl_bg = GecoParameter(initial_value=config["loss"]["lambda_kl_bg_min_max"][0],
                                        min_value=config["loss"]["lambda_kl_bg_min_max"][0],
                                        max_value=config["loss"]["lambda_kl_bg_min_max"][1],
                                        linear_exp=True)

        self.target_IoU_min = config["input_image"]["target_IoU_min_max"][0]
        self.target_IoU_max = config["input_image"]["target_IoU_min_max"][1]
        self.geco_kl_box = GecoParameter(initial_value=config["loss"]["lambda_kl_box_min_max"][0],
                                         min_value=config["loss"]["lambda_kl_box_min_max"][0],
                                         max_value=config["loss"]["lambda_kl_box_min_max"][1],
                                         linear_exp=True)



    def forward(self, imgs_bcwh: torch.Tensor,
                generate_synthetic_data: bool,
                iom_threshold: float,
                k_objects_max: int,
                topk_only: bool,
                noisy_sampling: bool,
                backbone_no_grad: bool) -> (Inference, MetricMiniBatch):

        # 0. preparation
        batch_size, ch_raw_image, width_raw_image, height_raw_image = imgs_bcwh.shape

        # 1. UNET
        unet_output: UNEToutput = self.unet.forward(imgs_bcwh, backbone_no_grad=backbone_no_grad, verbose=False)
        unet_prob_b1wh = torch.sigmoid(unet_output.logit)

        # 2. Background decoder
        if self.is_zero_background:
            out_background_bcwh = torch.zeros_like(imgs_bcwh)
            zbg_kl_av = torch.zeros(size=[1], dtype=imgs_bcwh.dtype, device=imgs_bcwh.device).mean()
            # out_background_bcwh = torch.zeros_like(imgs_bcwh) if self.is_zero_background else \
            #     F.interpolate(self.decoder_zbg(zbg.value), size=imgs_bcwh.shape[-2:], mode='bilinear', align_corners=False)
        else:
            zbg_mu, zbg_std = torch.split(unet_output.zbg,
                                          split_size_or_sections=unet_output.zbg.shape[-3]//2,
                                          dim=-3)
            zbg: DIST = sample_and_kl_diagonal_normal(posterior_mu=zbg_mu,
                                                      posterior_std=F.softplus(zbg_std)+1E-3,
                                                      noisy_sampling=noisy_sampling,
                                                      sample_from_prior=generate_synthetic_data)
            out_background_bcwh = self.decoder_zbg(zbg.value)
            zbg_kl_av = zbg.kl.sum() / batch_size

        # 3. Bounding-Box decoding
        #print("unet_output.zwhere.shape", unet_output.zwhere.shape)
        zwhere_mu, zwhere_std = unet_output.zwhere.chunk(chunks=2, dim=-3)
        zwhere: DIST = sample_and_kl_diagonal_normal(posterior_mu=zwhere_mu,
                                                     posterior_std=F.softplus(zwhere_std)+1E-3,
                                                     noisy_sampling=noisy_sampling,
                                                     sample_from_prior=generate_synthetic_data)
        t_grid = torch.sigmoid(self.decoder_zwhere(zwhere.value))  # shape B, 4, small_w, small_h
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
            annealing_factor = self.geco_annealing.get()
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
            c_grid_after_nms = c_grid_before_nms * nms_mask

            assert nms_mask.shape == c_grid_before_nms.shape == c_grid_after_nms.shape == unet_prob_b1wh.shape

            c_detached_bk = torch.gather(convert_to_box_list(c_grid_after_nms).squeeze(-1),
                                         dim=-1,
                                         index=nms_output.indices_k)

        # 6. Gather the probability and bounding_boxes which survived the NMS+TOP-K operation
        bounding_box_bk: BB = BB(bx=torch.gather(bounding_box_bn.bx, dim=-1, index=nms_output.indices_k),
                                 by=torch.gather(bounding_box_bn.by, dim=-1, index=nms_output.indices_k),
                                 bw=torch.gather(bounding_box_bn.bw, dim=-1, index=nms_output.indices_k),
                                 bh=torch.gather(bounding_box_bn.bh, dim=-1, index=nms_output.indices_k))
        zwhere_kl_bk = torch.gather(convert_to_box_list(zwhere.kl).sum(dim=-1), dim=-1, index=nms_output.indices_k)
        prob_bk = torch.gather(convert_to_box_list(unet_prob_b1wh).squeeze(-1), dim=-1, index=nms_output.indices_k)

        # 7. Crop the unet_features according to the selected boxes
        batch_size, k_boxes = bounding_box_bk.bx.shape
        unet_features_expanded = unet_output.features.unsqueeze(-4).expand(batch_size, k_boxes, -1, -1, -1)
        cropped_feature_map = Cropper.crop(bounding_box=bounding_box_bk,
                                           big_stuff=unet_features_expanded,
                                           width_small=self.glimpse_size,
                                           height_small=self.glimpse_size)

        # 8. Encode, Sample, Decode zinstance
        zinstance_mu_and_std: torch.Tensor = self.encoder_zinstance.forward(cropped_feature_map)
        zinstance_mu, zinstance_std = zinstance_mu_and_std.chunk(chunks=2, dim=-1)
        zinstance: DIST = sample_and_kl_diagonal_normal(posterior_mu=zinstance_mu,
                                                        posterior_std=F.softplus(zinstance_std)+1E-3,
                                                        noisy_sampling=noisy_sampling,
                                                        sample_from_prior=generate_synthetic_data)
        zinstance_kl_bk = zinstance.kl.sum(dim=-1)  # average over the latent dimension

        # 9. Dec(z_instance) -> y_k, w_k
        small_imgs_and_weights_out = self.decoder_zinstance.forward(zinstance.value)
        big_stuff = Uncropper.uncrop(bounding_box=bounding_box_bk,
                                     small_stuff=torch.cat((small_imgs_and_weights_out,
                                                            torch.ones_like(small_imgs_and_weights_out[..., :1, :, :])), dim=-3),
                                     width_big=width_raw_image,
                                     height_big=height_raw_image)  # shape: batch, n_box, ch, w, h
        out_img_bkcwh,  out_weight_bk1wh, out_square_bk1wh = big_stuff.split(split_size=(ch_raw_image, 1, 1), dim=-3)

        # 10. Compute the mixing (using a softmax-like function).
        c_attached_bk = (c_detached_bk.float() - prob_bk).detach() + prob_bk
        mixing_bk1wh = softmax_with_indicator(indicator=c_attached_bk[..., None, None, None] * out_square_bk1wh,
                                              weight=out_weight_bk1wh,
                                              dim=-4,
                                              add_one_to_denominator=True)
        mixing_fg_b1wh = mixing_bk1wh.sum(dim=-4)
        mixing_bg_b1wh = torch.ones_like(mixing_fg_b1wh) - mixing_fg_b1wh

        # 12. Observation model
        if self.GMM:
            y_bcwh = mixing_bg_b1wh * out_background_bcwh + (mixing_bk1wh * out_img_bkcwh).sum(dim=-4)
            mse_bcwh = ((y_bcwh - imgs_bcwh) / self.sigma_mse).pow(2)
            mse_av = mse_bcwh.mean()
            with torch.no_grad():
                mask_bg_b1wh = (mixing_bg_b1wh > 0.5)
                mse_bg_av = (mask_bg_b1wh * mse_bcwh).sum() / mask_bg_b1wh.sum()
                mse_fg_av = (~mask_bg_b1wh * mse_bcwh).sum() / ~mask_bg_b1wh.sum()
        else:
            mse_fg_bkcwh = ((out_img_bkcwh - imgs_bcwh.unsqueeze(-4)) / self.sigma_mse).pow(2)
            mse_bg_bcwh = ((out_background_bcwh - imgs_bcwh) / self.sigma_mse).pow(2)
            mse_bcwh = mixing_bg_b1wh * mse_bg_bcwh + torch.sum(mixing_bk1wh * mse_fg_bkcwh, dim=-4)
            mse_av = mse_bcwh.mean()
            with torch.no_grad():
                mask_fg_bk1wh = (mixing_bk1wh > 0.5)
                mse_fg_av = (mask_fg_bk1wh * mse_fg_bkcwh).sum() / mask_fg_bk1wh.sum()
                mask_bg_b1wh = (mixing_bg_b1wh > 0.5)
                mse_bg_av = (mask_bg_b1wh * mse_bg_bcwh).sum() / mask_bg_b1wh.sum()


        # Mask overlap. Note that only active masks contribute (b/c c_detached_bk)
        # Worst case scenario is when two masks are 0.5 for the same pixel -> cost = 0.5
        # Best case scenario is when one mask is 1.0 and other are zeros -> cost = 0.0
        #tmp_mixing_bk1wh, _ = torch.split(softmax_with_indicator(indicator=c_attached_bk[..., None, None, None].detach() * out_square_bk1wh,
        #                                                         weight=out_weight_bk1wh,
        #                                                         dim=-4,
        #                                                         append_uniform_zero_weight=True),
        #                                  split_size_or_sections=(out_square_bk1wh.shape[-4], 1), dim=-4)
        #mask_overlap_b1wh = tmp_mixing_bk1wh.sum(dim=-4).pow(2) - tmp_mixing_bk1wh.pow(2).sum(dim=-4)
        mask_overlap_b1wh = torch.zeros_like(imgs_bcwh[:,:1,:,:])

        # Bounding box overlap. Note that only active boxes contribute (b/c d_detached_bk)
        #box_bk1wh = c_detached_bk[..., None, None, None].detach() * out_square_bk1wh
        #box_overlap_b1wh = box_bk1wh.sum(dim=-4).pow(2) - box_bk1wh.pow(2).sum(dim=-4)
        box_overlap_b1wh = torch.zeros_like(imgs_bcwh[:,:1,:,:])

        # 13. Compute the ideal boxes and the IoU between inferred and ideal boxes
        bb_ideal_bk: BB = optimal_bb(mixing_k1wh=mixing_bk1wh,
                                     bounding_boxes_k=bounding_box_bk,
                                     pad_size=self.pad_size_bb,
                                     min_box_size=self.min_box_size,
                                     max_box_size=self.max_box_size)
        iou_bk = compute_iou(bounding_boxes_a=bounding_box_bk, bounding_boxes_b=bb_ideal_bk)
        iou_av = (iou_bk * c_detached_bk).sum() / c_detached_bk.sum().clamp(min=1.0)


        # 13. Compute the KL divergence between bernoulli posterior and DPP prior
        # Compute KL divergence between the DPP prior and the posterior:
        # KL(a,DPP) = \sum_c q(c|a) * [ log_q(c|a) - log_p(c|DPP) ] = -H[q] - sum_c q(c|a) log_p(c|DPP)
        # The first term is the negative entropy of the Bernoulli distribution.
        # It can be computed analytically and its minimization w.r.t. "a" leads to high entropy posteriors.
        # The second term can be estimated by REINFORCE ESTIMATOR and makes
        # the posterior have more weight on configuration which are likely under the prior

        # A. Analytical expression for entropy
        entropy_ber = compute_entropy_bernoulli(logit=unet_output.logit).sum(dim=(-1, -2, -3)).mean()

        # B. Reinforce estimator for gradients w.r.t. logit
        prob_expanded_nb1wh = unet_prob_b1wh.expand([self.n_mc_samples, -1, -1, -1, -1])  # keep: batch,ch,w,h
        c_mcsamples_nb1wh = (torch.rand_like(prob_expanded_nb1wh) < prob_expanded_nb1wh)
        logp_ber_nb = compute_logp_bernoulli(c=c_mcsamples_nb1wh.detach(),
                                             logit=unet_output.logit).sum(dim=(-1, -2, -3))
        with torch.no_grad():
            # These 3 lines are the reinforce estimator
            logp_dpp_nb = self.grid_dpp.log_prob(value=c_mcsamples_nb1wh.squeeze(dim=-3).detach())
            baseline_b = logp_dpp_nb.mean(dim=0)  # average of the different MC samples
            d_nb = (logp_dpp_nb - baseline_b)
        reinforce_ber = (logp_ber_nb * d_nb.detach()).mean()

        # C. Simple MC estimator to make DPP adjust to the configuration after NMS.
        # This has gradients w.r.t DPP parameters
        log_dpp_after_nms = self.grid_dpp.log_prob(value=c_grid_after_nms.squeeze(-3).detach()).mean() \
            if self.grid_dpp.learnable_params else torch.zeros_like(entropy_ber)

        # D. Put together the expression for both the evaluation of the gradients and the evaluation of the value
        logit_kl_for_gradient_av = - entropy_ber - reinforce_ber - log_dpp_after_nms
        logit_kl_for_value_av = (logp_ber_nb - logp_dpp_nb).mean().detach()
        logit_kl_av = (logit_kl_for_value_av - logit_kl_for_gradient_av).detach() + logit_kl_for_gradient_av

        # The KL for instance and boxes
        # indicator_bk = c_detached_bk
        indicator_bk = torch.ones_like(c_detached_bk)
        zinstance_kl_av = (zinstance_kl_bk * indicator_bk).sum() / batch_size
        zwhere_kl_av = (zwhere_kl_bk * indicator_bk).sum() / batch_size

        # GECO (i.e. make the hyper-parameters dynamical)
        # if constraint < 0, parameter will be decreased
        # if constraint > 0, parameter will be increased
        # if constraint = 0, parameter will stay the same
        with torch.no_grad():

            # NOBJ: Count object using c_detached and couple to unet_prob_b1wh (force n_obj in range)
            nav_selected_smooth = prob_bk.sum(dim=-1).mean()
            nav_selected_hard = c_detached_bk.sum(dim=-1).float().mean()
            nav_selected = torch.max(nav_selected_hard, nav_selected_smooth)
            nobj_in_range = (nav_selected > self.target_nobj_av_per_patch_min) * \
                            (nav_selected < self.target_nobj_av_per_patch_max)
            constraint_nobj_max = nav_selected - self.target_nobj_av_per_patch_max  # positive if nobj > target_max
            constraint_nobj_min = self.target_nobj_av_per_patch_min - nav_selected  # positive if nobj < target_min

            # FGFRACTION: Count and couple based on mixing_fg (force fg_fraction in range)
            fgfraction_av_hard = (mixing_fg_b1wh > 0.5).float().mean()
            fgfraction_av_smooth = mixing_fg_b1wh.mean()
            fgfraction_av = torch.min(fgfraction_av_hard, fgfraction_av_smooth)
            fgfraction_in_range = (fgfraction_av < self.target_fgfraction_max) * \
                                  (fgfraction_av > self.target_fgfraction_min)
            constraint_fgfraction_min = self.target_fgfraction_min - fgfraction_av  # positive if fgfraction < target_min
            constraint_fgfraction_max = fgfraction_av - self.target_fgfraction_max  # positive if fgfraction > target_max

            # ANNEALING: Reduce if mse_av < self.target_mse_for_annealing and other conditions are satisfied
            decrease_annealing = (mse_av < self.target_mse_for_annealing) * nobj_in_range * fgfraction_in_range
            constraint_annealing = - 1.0 * decrease_annealing

            # MSE_FG AND MSE_BG need to be equal to target. I change kl strength to satisfy this condition
            # if target > mse increases factor in front of kl and viceverse
            constraint_kl_fg = (self.target_mse_fg - mse_fg_av) * (annealing_factor == 0.0)
            constraint_kl_bg = (self.target_mse_bg - mse_bg_av) * (annealing_factor == 0.0)

            # IoU need to be in range.
            increase_kl_box = (iou_av - self.target_IoU_max).clamp(min=0) # if IoU_av > target_max increases kl strength
            decrease_kl_box = (self.target_IoU_min - iou_av).clamp(min=0) # if IoU_av < target_min decreases kl strength
            constraint_kl_box = increase_kl_box - decrease_kl_box

        # Produce both the loss and the hyperparameters
        geco_fgfraction_min: GECO = self.geco_fgfraction_min.forward(constraint=constraint_fgfraction_min)
        geco_fgfraction_max: GECO = self.geco_fgfraction_max.forward(constraint=constraint_fgfraction_max)

        geco_nobj_min: GECO = self.geco_nobj_min.forward(constraint=constraint_nobj_min)
        geco_nobj_max: GECO = self.geco_nobj_max.forward(constraint=constraint_nobj_max)

        geco_annealing: GECO = self.geco_annealing.forward(constraint=constraint_annealing)

        geco_kl_fg: GECO = self.geco_kl_fg.forward(constraint=constraint_kl_fg)
        geco_kl_bg: GECO = self.geco_kl_bg.forward(constraint=constraint_kl_bg)
        geco_kl_box: GECO = self.geco_kl_box.forward(constraint=constraint_kl_box)


####        # KL_BG: If self.target_mse_bg < mse_bg then decrease kl_background and viceversa
####        constraint_kl_bg = (self.target_mse_bg - mse_bg_av) * fgfraction_in_range * (annealing_factor == 0.0)
####
####        # MSE.
####        # If mse_av > self.target_mse Increase mse_hyperparameter and viceversa
####        constraint_mse = (mse_av - self.target_mse)
####
####        #####            # SPARSITY:
####        #####            # If the reconstruction is good enough and n_obj > n_min increase sparsity.
####        #####            # If n_obj < n_min decrease sparsity
####        #####            # If n_obj in range and reconstruction is bad do nothing
####        #####            increase_sparsity = (nav_selected > self.target_nobj_av_per_patch_min) * (mse_av < self.target_mse)
####        #####            decrease_sparsity = (nav_selected < self.target_nobj_av_per_patch_min)
####        #####            constraint_sparsity = 1.0 * increase_sparsity - 1.0 * decrease_sparsity

        # Confining potential which keeps logit into intermediate regime
        logit_max = 8.0
        logit_min = -8.0
        all_logit_in_range = 100 * torch.sum( (unet_output.logit - logit_max).clamp(min=0.0).pow(2) +
                                            (- unet_output.logit + logit_min).clamp(min=0.0).pow(2) ) / batch_size

        # Linear potential which can be used to control the number of object automatically.
        lambda_nobj = (geco_nobj_max.hyperparam - geco_nobj_min.hyperparam)
        nobj_coupling = unet_output.logit.clamp(min=logit_min, max=logit_max).sum() / batch_size

        # Couple to mixing to push the fgfraction up/down
        lambda_fgfraction = (geco_fgfraction_max.hyperparam - geco_fgfraction_min.hyperparam)
        fgfraction_coupling = mixing_fg_b1wh.mean()

        # Penalize overlapping masks
        mask_overlap_cost = self.mask_overlap_strength * mask_overlap_b1wh.mean()

        # Penalize overlapping boxes
        box_overlap_cost = self.box_overlap_strength * box_overlap_b1wh.mean()


        if self.multi_objective_optimization:

            loss_geco = geco_fgfraction_max.loss + geco_fgfraction_min.loss + \
                        geco_nobj_max.loss + geco_nobj_min.loss + geco_annealing.loss + \
                        geco_kl_box.loss + geco_kl_bg.loss + geco_kl_fg.loss

            # Reconstruction within the acceptable parameter range
            task_rec = mse_av + mask_overlap_cost + box_overlap_cost - iou_bk.sum() / batch_size  + \
                       lambda_fgfraction * fgfraction_coupling + \
                       lambda_nobj * nobj_coupling + all_logit_in_range + \
                       geco_kl_fg.hyperparam * zinstance_kl_av + \
                       geco_kl_bg.hyperparam * zbg_kl_av + \
                       geco_kl_box.hyperparam * zwhere_kl_av

            # task_sparsity = c_attached_bk.mean()
            # loss_tot = torch.stack([task_rec + loss_geco, logit_kl_av, task_sparsity], dim=0)
            loss_tot = torch.stack([task_rec + loss_geco, logit_kl_av], dim=0)

            # Idea. After this is done I can turn on a sparsity term


        else:

            loss_geco = geco_fgfraction_max.loss + geco_fgfraction_min.loss + \
                        geco_nobj_max.loss + geco_nobj_min.loss + geco_annealing.loss

            # Reconstruction within the acceptable parameter range
            task_rec = mse_av + mask_overlap_cost + box_overlap_cost - iou_bk.sum()/batch_size + \
                       lambda_fgfraction * fgfraction_coupling + \
                       all_logit_in_range + lambda_nobj * nobj_coupling

            # these three are tuned based on (rec_fg, rec_bg and iou_av).....
            task_simplicity = logit_kl_av + zinstance_kl_av + zbg_kl_av + zwhere_kl_av

            # then I do a crazy low ramp of this term.
            # N_obj should never be smaller than self.target_nobj_av_per_patch_min
            # Here I couple to logit_bk so that
            task_sparsity = c_attached_bk.mean()

            loss_tot = task_rec + loss_geco + 0.001 * task_simplicity + 0.0 * task_sparsity

        inference = Inference(logit_grid=unet_output.logit.detach(),
                              prob_from_ranking_grid=prob_from_ranking_grid.detach(),
                              background_cwh=out_background_bcwh.detach(),
                              foreground_kcwh=out_img_bkcwh.detach(),
                              mask_overlap_1wh=mask_overlap_b1wh.detach(),
                              mixing_k1wh=mixing_bk1wh.detach(),
                              sample_c_grid_before_nms=c_grid_before_nms.detach(),
                              sample_c_grid_after_nms=c_grid_after_nms.detach(),
                              sample_prob_k=prob_bk.detach(),
                              sample_c_k=c_detached_bk.detach(),
                              sample_bb_k=bounding_box_bk,
                              sample_bb_ideal_k=bb_ideal_bk,
                              feature_map=unet_output.features.detach(),
                              iou_boxes_k=iou_bk.detach(),
                              kl_instance_k=zinstance_kl_bk.detach(),
                              kl_where_k=zwhere_kl_bk.detach())

        similarity_l, similarity_w = self.grid_dpp.similiraty_kernel.get_l_w()

        if backbone_no_grad and self.training:
            unet_output.zwhere.retain_grad()
            unet_output.logit.retain_grad()
            unet_output.zbg.retain_grad()
            zinstance_mu_and_std.retain_grad()
            bottleneck = (unet_output.zwhere, unet_output.logit, unet_output.zbg, zinstance_mu_and_std)
        else:
            bottleneck = None

        metric = MetricMiniBatch(loss=loss_tot,
                                 bottleneck=bottleneck,
                                 # monitoring
                                 mse_av=mse_av.detach().item(),
                                 mse_fg_av=mse_fg_av.detach().item(),
                                 mse_bg_av=mse_bg_av.detach().item(),
                                 fgfraction_smooth_av=fgfraction_av_smooth.detach().item(),
                                 fgfraction_hard_av=fgfraction_av_hard.detach().item(),
                                 nobj_smooth_av=nav_selected_smooth.detach().item(),
                                 nobj_hard_av=nav_selected_hard.detach().item(),
                                 prob_grid_av=unet_prob_b1wh.sum(dim=(-1,-2,-3)).mean().detach().item(),
                                 # terms in the loss function
                                 cost_mse=mse_av.detach().item(),
                                 cost_mask_overlap_av=mask_overlap_cost.detach().item(),
                                 cost_box_overlap_av=box_overlap_cost.detach().item(),
                                 cost_fgfraction=fgfraction_coupling.detach().item(),
                                 kl_zinstance=zinstance_kl_av.detach().item(),
                                 kl_zbg=zbg_kl_av.detach().item(),
                                 kl_zwhere=zwhere_kl_av.detach().item(),
                                 kl_logit=logit_kl_av.detach().item(),
                                 # debug
                                 logit_min=unet_output.logit.min().detach().item(),
                                 logit_mean=unet_output.logit.mean().detach().item(),
                                 logit_max=unet_output.logit.max().detach().item(),
                                 similarity_l=similarity_l.detach().item(),
                                 similarity_w=similarity_w.detach().item(),
                                 iou_boxes=iou_av.detach().item(),
                                 # TODO: change name to geco_kl
                                 lambda_annealing=geco_annealing.hyperparam.detach().item(),
                                 lambda_fgfraction_max=geco_fgfraction_max.hyperparam.detach().item(),
                                 lambda_fgfraction_min=geco_fgfraction_min.hyperparam.detach().item(),
                                 lambda_fgfraction=lambda_fgfraction.detach().item(),
                                 lambda_nobj_max=geco_nobj_max.hyperparam.detach().item(),
                                 lambda_nobj_min=geco_nobj_min.hyperparam.detach().item(),
                                 lambda_nobj=lambda_nobj.detach().item(),
                                 lambda_kl_fg=geco_kl_fg.hyperparam.detach().item(),
                                 lambda_kl_bg=geco_kl_bg.hyperparam.detach().item(),
                                 lambda_kl_box=geco_kl_box.hyperparam.detach().item(),
                                 entropy_ber=entropy_ber.detach().item(),
                                 reinforce_ber=reinforce_ber.detach().item(),
                                 # TODO: remove the running averages
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


#######        # Note that this dynamical parameter can be BOTH POSITIVE AND NEGATIVE
#######        fgfrac_hard_av = (mixing_fg_b1wh > 0.5).float().mean()
#######        fgfraction_in_range = (fgfrac_hard_av > self.target_fgfraction_min) * (fgfrac_hard_av < self.target_fgfraction_max)
#######        geco_fgfraction_min: GECO = self.geco_fgfraction_min.forward(constraint=1.0-2.0*(fgfrac_hard_av > self.target_fgfraction_min))
#######        geco_fgfraction_max: GECO = self.geco_fgfraction_max.forward(constraint=1.0-2.0*(fgfrac_hard_av < self.target_fgfraction_max))
#######        geco_fgfrac_hyperparam = geco_fgfraction_max.hyperparam - geco_fgfraction_min.hyperparam
#######
#######        # REINFORCE
#######        # Reinforce starts at ONE and becomes ZERO at the end of the annealing procedure
#######        # Reducing geco_reinforce.hyperparam leads to fewer objects
#######        nobj_grid_av = c_grid_after_nms.sum(dim=(-1,-2,-3)).float().mean()
#######        nobj_grid_in_range = (nobj_grid_av > target_nobj_min) * (nobj_grid_av < target_nobj_max)
#######        nobj_grid_not_too_small = (nobj_grid_av > target_nobj_min)
#######        nobj_grid_too_small = (nobj_grid_av < target_nobj_min)
#######
#######        nobj_av = (prob_bk > 0.5).sum(dim=-1).float().mean()
#######        nobj_too_large = (nobj_av > target_nobj_max)
#######        nobj_too_small = (nobj_av < target_nobj_min)
#######
#######        decrease_reinforce = (nobj_grid_not_too_small * fgfraction_in_range * mse_in_range) or nobj_too_large
#######        increase_reinforce = nobj_grid_too_small or nobj_too_small
#######        geco_reinforce: GECO = self.geco_reinforce.forward(constraint=1.0*increase_reinforce - 1.0*decrease_reinforce)
#######
#######        # ANNEALING
#######        # Annealing starts at ONE and becomes ZERO at the end of the annealing procedure
#######        # Reducing geco_annealing_factor.hyperparam force the model to learn to localize
#######        # Note that annealing can never be increased. It decrease monotonically
#######        decrease_annealing = mse_in_range * nobj_grid_in_range * fgfraction_in_range
#######        geco_annealing: GECO = self.geco_annealing_factor.forward(constraint=-1.0*decrease_annealing)
#######
#######        # MSE
#######        decrease_lambda_mse = (mse_in_range * nobj_grid_in_range * fgfraction_in_range) or \
#######                              mse_too_small or \
#######                              (nobj_too_large * (geco_reinforce.hyperparam == 0))
#######        increase_lambda_mse = mse_too_large
#######        geco_mse: GECO = self.geco_mse_max.forward(constraint=1.0*increase_lambda_mse-1.0*decrease_lambda_mse)
#######