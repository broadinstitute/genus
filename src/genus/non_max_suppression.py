import torch
from .util import are_broadcastable
from torch.distributions.utils import broadcast_all
from .namedtuple import BB, NmsOutput


class NonMaxSuppression(object):
    """ Use Intersection_over_Minimum criteria to filter out overlapping proposals. """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _perform_nms_selection(mask_overlap_nn: torch.Tensor,
                               score_n: torch.Tensor,
                               possible_n: torch.Tensor,
                               k_objects_max: int) -> torch.Tensor:
        """
        This algorithm does greedy NMS in parallel if possible.

        Args:
        mask_overlap_nn: Binarized overlap matrix with 1 if IoMIN > threshold and 0 otherwise.
        score_n: Score of the proposal. Higher score porposal have precedence
        possible_n: Vector with 1 if the proposal can be chosen and 0 otherwise
        k_objects_max: the maximum number of proposals to select.
            The algorithm terminates either when :attr:`k_objects_max` have been selected or there are no more suitable
            proposal (becuase they have all been suppressed by higher scoring ones).

        Returns:
             The tensor :attr:`selected_n` has the same shape as :attr:`score_n`. The entries are 1 if that proposal
             has been selected and 0 otherwise.

        Note:
            The :attr:`mask_overlap_nn`, :attr:`score_n`, :attr:`possible_n` can have arbitrary leading dimension.
            The NMS algorithm will be performed independently for all leading dimensions.

        """
        # reshape
        score_1n = score_n.unsqueeze(-2)
        possible_1n = possible_n.unsqueeze(-2)
        idx_n1 = torch.arange(start=0, end=score_n.shape[-1], step=1, device=score_n.device).view(-1, 1).long()
        selected_n1 = torch.zeros_like(score_n).unsqueeze(dim=-1)

        # Loop
        counter = 0
        while (counter <= k_objects_max) and (possible_1n.sum() > 0):
            score_mask_nn = mask_overlap_nn * (score_1n * possible_1n)
            index_n1 = torch.max(score_mask_nn, keepdim=True, dim=-1)[1]
            selected_n1 += possible_1n.transpose(dim0=-1, dim1=-2) * (idx_n1 == index_n1)
            blocks_1n = torch.sum(mask_overlap_nn * selected_n1, keepdim=True, dim=-2)
            possible_1n *= (blocks_1n == 0)
            counter += 1

        return selected_n1.squeeze(-1)

    @staticmethod
    def _unroll_and_compare(x: torch.Tensor, label: str) -> torch.Tensor:
        """ Given a vector of size: (*, n) creates an output of size (*, n, n)
            obtained by comparing all vector entries with all other vector entries
            The comparison is either: MIN,MAX """
        if label == "MAX":
            y = torch.max(x.unsqueeze(-1), x.unsqueeze(-2))
        elif label == "MIN":
            y = torch.min(x.unsqueeze(-1), x.unsqueeze(-2))
        else:
            raise Exception("label is unknown. It is ", label)
        return y

    @staticmethod
    def _compute_box_IoMIN(bx: torch.Tensor, by: torch.Tensor, bw: torch.Tensor, bh: torch.Tensor) -> torch.Tensor:
        """
        Given an input of size (*, n) computes an output of size (*, n, n) with the Intersection Over Min Area among all
        pairs of boxes
        """
        # compute x1,x3,y1,y3
        x1 = bx - 0.5 * bw
        x3 = bx + 0.5 * bw
        y1 = by - 0.5 * bh
        y3 = by + 0.5 * bh
        area = bw * bh

        min_area_nn = NonMaxSuppression._unroll_and_compare(area, "MIN")
        xi1_nn = NonMaxSuppression._unroll_and_compare(x1, "MAX")
        yi1_nn = NonMaxSuppression._unroll_and_compare(y1, "MAX")
        xi3_nn = NonMaxSuppression._unroll_and_compare(x3, "MIN")
        yi3_nn = NonMaxSuppression._unroll_and_compare(y3, "MIN")

        intersection_area_nn = torch.clamp(xi3_nn - xi1_nn, min=0) * torch.clamp(yi3_nn - yi1_nn, min=0)
        return intersection_area_nn / min_area_nn

    @staticmethod
    @torch.no_grad()
    def compute_mask_and_index(score: torch.Tensor,
                               bounding_box: BB,
                               iom_threshold: float,
                               k_objects_max: int,
                               topk_only: bool) -> NmsOutput:
        """
        Filter the proposals according to their score and their Intersection over Minimum.

        Args:
            score: score used to sort the proposals
            bounding_box: bounding boxes for the proposals
            iom_threshold: threshold of Intersection over Minimum. If IoM is larger than this value the boxes
                will be suppressed during NMS. It is imporatant only if :attr:`topk_only` is False.
            k_objects_max: maximum number of proposal to consider.
            topk_only: if True, this function performs a top-K filter and returns the indices of the k-highest
                scoring proposals regardless of their IoU. If False, the function perform NMS and returns the
                 indices of the k-highest scoring weakly-overlapping proposals.

        Returns:
            The container of type :class:`NmsOutput` with the selected proposal.

        Note:
            The :attr:`score_n`, :attr:`bounding_box_n` can have arbitrary leading dimension.
            The NMS algorithm will be performed independently for all leading dimensions.
        """
        assert are_broadcastable(score, bounding_box.bx)
        score_n, bx_n, by_n, bw_n, bh_n = broadcast_all(score,
                                                        bounding_box.bx, bounding_box.by,
                                                        bounding_box.bw, bounding_box.bh)

        if topk_only:
            # If nms_mask = 1 then this is equivalent to do topk only
            chosen_nms_mask_n = torch.ones_like(score_n)
        else:
            # this is O(N^2) algorithm (all boxes compared to all other boxes) but it is very simple
            overlap_measure_nn = NonMaxSuppression._compute_box_IoMIN(bx=bx_n, by=by_n, bw=bw_n, bh=bh_n)
            # Next greedy NMS
            binarized_overlap_nn = (overlap_measure_nn > iom_threshold).float()
            chosen_nms_mask_n = NonMaxSuppression._perform_nms_selection(mask_overlap_nn=binarized_overlap_nn,
                                                                         score_n=score_n,
                                                                         possible_n=torch.ones_like(score_n).bool(),
                                                                         k_objects_max=k_objects_max)

        # select the indices of the top boxes according to the masked_score.
        # Note that masked_score are zero for the boxes which underwent NMS
        assert chosen_nms_mask_n.shape == score_n.shape
        masked_score_n = chosen_nms_mask_n * score_n
        k = min(k_objects_max, score_n.shape[-1])
        indices_k = torch.topk(masked_score_n, k=k, dim=-1, largest=True, sorted=True)[1]

        k_mask_n = torch.zeros_like(masked_score_n).scatter(dim=-1,
                                                            index=indices_k,
                                                            src=torch.ones_like(masked_score_n))
        return NmsOutput(k_mask_n=k_mask_n.bool(), indices_k=indices_k)
