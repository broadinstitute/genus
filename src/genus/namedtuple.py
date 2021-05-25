import torch
import numpy
from typing import NamedTuple, Optional
import skimage.color
import matplotlib.pyplot as plt

#  ----------------------------------------------------------------  #
#  ------- Stuff Related to PreProcessing -------------------------  #
#  ----------------------------------------------------------------  #


class ImageBbox(NamedTuple):
    """ Follows Scikit Image convention. Pixels belonging to the bounding box are in the half-open interval:
        [min_row;max_row) and [min_col;max_col). """
    min_row: int
    min_col: int
    max_row: int
    max_col: int


class PreProcess(NamedTuple):
    img: torch.Tensor
    roi_mask: torch.Tensor
    bbox_original: ImageBbox
    bbox_crop: ImageBbox

#  --------------------------------------------------------------------------------------  #
#  ------- Stuff Related to PostProcessing (i.e. Graph Clustering Based on Modularity) --  #
#  --------------------------------------------------------------------------------------  #


class Suggestion(NamedTuple):
    best_resolution: float
    best_index: int
    sweep_resolution: torch.Tensor
    sweep_mi: torch.Tensor
    sweep_iou: torch.Tensor
    sweep_delta_n: torch.Tensor
    sweep_seg_mask: torch.Tensor
    sweep_n_cells: torch.Tensor
    sweep_sizes: list
        
    def show_index(self, index: int, figsize: tuple = (12, 12), fontsize: int = 20):
        figure, ax = plt.subplots(figsize=figsize)
        ax.imshow(skimage.color.label2rgb(label=self.sweep_seg_mask[index].cpu().numpy(), bg_label=0))
        ax.set_title('resolution = {0:.3f}, \
                      iou = {1:.3f}, \
                      delta_n = {2:3d}, \
                      n_cells = {3:3d}'.format(self.sweep_resolution[index],
                                               self.sweep_iou[index],
                                               self.sweep_delta_n[index],
                                               self.sweep_n_cells[index]),
                     fontsize=fontsize)
        
    def show_best(self, figsize: tuple = (12, 12), fontsize: int = 20):
        return self.show_index(self.best_index, figsize, fontsize)
        
    def show_graph(self, figsize: tuple = (12, 12), fontsize: int = 20):
        figure, ax = plt.subplots(figsize=figsize)
        ax.set_title('Resolution sweep', fontsize=fontsize)
        ax.set_xlabel("resolution", fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        
        color = 'tab:red'
        _ = ax.plot(self.sweep_resolution, self.sweep_n_cells, '.--', label="n_cell", color=color)
        ax.set_ylabel('n_cell', color=color, fontsize=fontsize)
        ax.tick_params(axis='y', labelcolor=color, labelsize=fontsize)
        ax.legend(loc='upper left', fontsize=fontsize)
        ax.grid()

        ax_2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:green'
        _ = ax_2.plot(self.sweep_resolution, self.sweep_iou, '-', label="iou", color=color)
        ax_2.set_ylabel('Intersection Over Union', color=color, fontsize=fontsize)
        ax_2.tick_params(axis='y', labelcolor=color, labelsize=fontsize)
        ax_2.legend(loc='upper right', fontsize=fontsize)


class ConcordanceIntMask(NamedTuple):
    intersection_mask: torch.tensor
    joint_distribution: torch.tensor
    mutual_information: float
    delta_n: int
    iou: float
    n_reversible_instances: int


class Partition(NamedTuple):
    sizes: torch.tensor  # both for bg and fg. It is simply obtained by numpy.bincount(membership)
    membership: torch.tensor  # bg=0, fg=1,2,3,.....

    def compactify(self):
        """ if there are gaps in the SIZES, then shift both membership and sizes accordingly"""
        if (self.sizes[1:] > 0).all():
            return self
        else:
            my_filter = self.sizes > 0
            my_filter[0] = True
            count = torch.cumsum(my_filter, dim=-1)
            old_2_new = ((count - count[0]) * my_filter).to(self.membership.dtype)
            return Partition(sizes=self.sizes[my_filter], membership=old_2_new[self.membership])

    def filter_by_vertex(self, keep_vertex: torch.tensor):
        assert self.membership.shape == keep_vertex.shape
        assert keep_vertex.dtype == torch.bool
            
        if keep_vertex.all():
            return self
        else:
            new_membership = self.membership * keep_vertex  # put all bad vertices in the background cluster
            return Partition(sizes=torch.bincount(new_membership),
                             membership=new_membership).compactify()

    def filter_by_size(self,
                       min_size: Optional[int] = None,
                       max_size: Optional[int] = None):
        """ If a cluster is too small or too large, its label is set to zero (i.e. background value).
            The other labels are adjusted so that there are no gaps in the labels number.
            Min_size and Max_size are integers specifying the number of pixels.
        """
        if (min_size is None) and (max_size is None):
            return self
        elif (min_size is not None) and (max_size is not None):
            assert max_size > min_size > 0, "Condition max_size > min_size > 0 failed."
            my_filter = (self.sizes > min_size) * (self.sizes < max_size)
        elif min_size is not None:
            assert min_size > 0, "Condition min_size > 0 failed."
            my_filter = (self.sizes > min_size)
        elif max_size is not None:
            assert max_size > 0, "Condition max_size > 0 failed."
            my_filter = (self.sizes < max_size)
        else:
            raise Exception("you should never be here!!")

        my_filter[0] = True  # always keep the bg
        if my_filter.all():
            return self
        else:
            return Partition(sizes=self.sizes * my_filter,
                             membership=self.membership).compactify()

#  ----------------------------------------------------------------  #
#  ------- Stuff Related to Processing (i.e. CompositionalVAE) ----  #
#  ----------------------------------------------------------------  #

class GECO(NamedTuple):
    loss: torch.Tensor
    hyperparam: torch.Tensor


class DIST(NamedTuple):
    """ Container for distribution sample and KL """
    value: torch.Tensor
    kl: torch.Tensor


class BB(NamedTuple):
    bx: torch.Tensor
    by: torch.Tensor
    bw: torch.Tensor
    bh: torch.Tensor


class TT(NamedTuple):
    tx: torch.Tensor
    ty: torch.Tensor
    tw: torch.Tensor
    th: torch.Tensor
    ix: torch.Tensor
    iy: torch.Tensor


class NmsOutput(NamedTuple):
    chosen_mask: torch.Tensor  # mask with exactly k ones
    indices_k: torch.Tensor
    score_k: torch.Tensor # score of the chosen object


class SparseSimilarity(NamedTuple):
    sparse_matrix: torch.sparse.FloatTensor
    index_matrix: Optional[torch.tensor]


class Segmentation(NamedTuple):
    """ Where * is the batch dimension which might be NOT present """
    raw_image: torch.Tensor  # *,ch,w,h
    fg_prob: torch.Tensor  # *,1,w,h
    integer_mask: torch.Tensor  # *,1,w,h
    bounding_boxes: Optional[torch.Tensor] = None  # *,3,w,h
    bounding_boxes_ideal: Optional[torch.Tensor] = None  # *,3,w,h
    similarity: Optional[SparseSimilarity] = None


class UNEToutput(NamedTuple):
    zwhere: torch.Tensor
    zbg: torch.Tensor
    logit: torch.Tensor
    features: torch.Tensor


class Inference(NamedTuple):
    logit_grid: torch.Tensor
    prob_from_ranking_grid: torch.Tensor
    background_cwh: torch.Tensor
    foreground_kcwh: torch.Tensor
    mask_overlap_1wh: torch.Tensor
    mixing_k1wh: torch.Tensor
    # the sample of the 4 latent variables
    sample_c_grid_before_nms: torch.Tensor
    sample_c_grid_after_nms: torch.Tensor
    sample_prob_k: torch.Tensor
    sample_bb_k: BB
    sample_bb_ideal_k: BB
    # Debug
    feature_map: torch.Tensor


class MetricMiniBatch(NamedTuple):
    """
    Container for all metric of interest to monitor the behaviour of the model during training.
    One special element of the metric is the loss which is the only one which will be directly optimized by
    gradient-based methods.

    Note:
        Loss is the only member which is a torch.Tensor because it need to be differentiated.
        All other members are float or numpy.array and are simply quantities which are monitored but
        not differentiated during training.

    Note:
        All entries should be scalars obtained by averaging over minibatch
    """
    loss: torch.Tensor  # this is the only tensor b/c I need to take gradients
    # monitoring
    mse_av: float
    mse_fg_av: float
    mixing_fg_av: float
    fgfraction_smooth_av: float
    fgfraction_hard_av: float
    nobj_hard_av: float
    nobj_soft_av: float
    prob_grid_av: float
    # term in the loss function
    cost_mse: float
    cost_mask_overlap_av: float
    cost_box_overlap_av: float
    cost_fgfraction: float
    # cost_nobj: float
    cost_bb_regression_av: float
    kl_zinstance: float
    kl_zbg: float
    kl_zwhere: float
    kl_logit: float
    # debug
    logit_min: float
    logit_mean: float
    logit_max: float
    similarity_l: float
    similarity_w: float
    lambda_annealing: float
    lambda_fgfraction: float
    lambda_kl_learnz: float
    lambda_kl_learnc: float
    lambda_nobj_max: float
    lambda_nobj_min: float
    entropy_ber: float
    reinforce_ber: float
    # conting accuracy
    count_prediction: numpy.ndarray
    wrong_examples: numpy.ndarray
    accuracy: float

    def pretty_print(self, epoch: int = 0) -> str:
        s = "[epoch {0:4d}] loss={1:.3f}, mse={2:.3f}, mask_overlap={3:.3f}, \
             bb_regression={4:.3f}, fgfraction_hard_av={5:.3f}, n_cell_av={6:.3f}".format(epoch,
                                                                                          self.loss,
                                                                                          self.mse_av,
                                                                                          self.cost_mask_overlap_av,
                                                                                          self.cost_bb_regression_av,
                                                                                          self.fgfraction_hard_av,
                                                                                          self.nobj_hard_av)
        return s


class Output(NamedTuple):
    """
    Container with the output of the :method:`process_batch_imgs`.

    Args:
        metrics: element of :class:`MetricMiniBatch` with all quantities worth measuring during training of the model
        inference: element of :class:`Inference` with many latent variable (not necessarely scalar quantities)
        imgs: reconstructed images. If the reconstruction was not computed this is set to None
        bb_imgs: input images with overimposed the bounding box (both the inferreed ones and the optimal one)
    """
    metrics: MetricMiniBatch
    inference: Inference
    imgs: Optional[torch.Tensor]
    bb_imgs: Optional[torch.Tensor]