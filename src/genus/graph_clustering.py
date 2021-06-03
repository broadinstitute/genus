import neptune
import torch
import numpy
import skimage.segmentation
import skimage.color
import leidenalg as la
import igraph as ig
from typing import Optional, List, Iterable
from matplotlib import pyplot as plt
from .namedtuple import Segmentation, Partition, SparseSimilarity, Suggestion, ConcordanceIntMask
# from .util_logging import log_img_and_chart
from .util import concordance_integer_masks, remove_label_gaps
import scipy
from scipy.sparse import csr_matrix, coo_matrix
from typing import Union

EPS = 1E-3


def scipy_sparse_complementary(x: csr_matrix) ->  csr_matrix:
    """
    Take a sparse csr_matrix and replace the non-zero element with 1-x.
    The zero element are unchanged. This transformation preserves the number of non-zero elements.
    """
    return csr_matrix((1.0+EPS-x.data, x.indices, x.indptr), shape=x.shape, dtype=x.dtype)


def torch_sparse_complementary(x: torch.sparse.FloatTensor) ->  torch.sparse.FloatTensor:
    """
    Take a sparse torch tensor and replace the non-zero element with 1-x.
    The zero element are unchanged. This transformation preserves the number of non-zero elements.
    """
    if not x.is_coalesced():
        x = x.coalesce()
    return torch.sparse_coo_tensor(indices=x.indices(), values=1.0-x.values(), size=x.size()).coalesce()


def torch_2_coo(x: torch.sparse.FloatTensor, complementary: bool = False) -> scipy.sparse.coo_matrix:
    """
    Convert a torch sparse tensor into a scipy.sparse.coo_matrix
    If complementary = True then the non-zero element are changed x -> 1-x
    """
    assert x.ndim == 2, "Expected a input tensor of dim=2, instead I received {0} dimensions".format(x.ndim)

    if not x.is_coalesced():
        x = x.coalesce()

    data = 1.0 + EPS - x.values().numpy() if complementary else x.values().numpy()
    indices = x.indices().numpy()
    shape = tuple(x.size())
    return coo_matrix((data, (indices[0], indices[1])), shape=shape)

def torch_2_csr(x: torch.sparse_coo_tensor, complementary: bool = False) -> scipy.sparse.csr_matrix:
    """
    Convert a torch sparse tensor into a scipy.sparse.csr_matrix.
    If complementary = True then the non-zero element are changed x -> 1-x
    """
    return torch_2_coo(x, complementary).tocsr()


def coo_2_torch(x: scipy.sparse.coo_matrix, complementary: bool=False) -> torch.sparse.FloatTensor:
    """
    Convert a coo matrix into a torch sparse tensor.
    If complementary = True then the non-zero element are changed x -> 1-x
    """
    shape = tuple(x.shape)
    row = torch.from_numpy(x.row).long()
    col = torch.from_numpy(x.col).long()
    values = 1.0 + EPS - x.data if complementary else x.data
    return torch.sparse_coo_tensor(indices=torch.stack((row,col), dim=0), values=values, size=shape).coalesce()


def csr_2_torch(x: scipy.sparse.csr_matrix, complementary: bool=False) -> torch.sparse.FloatTensor:
    """
    Convert a csr matrix into a torch sparse tensor.
    If complementary = True then the non-zero element are changed x -> 1-x
    """
    coo_matrix = x.tocoo()
    return coo_2_torch(coo_matrix, complementary)


def remove_label_gaps(labels: torch.Tensor):
    """ Remove gaps in the labels """
    sizes = torch.bincount(labels)
    exist = (sizes > 0).to(labels.dtype)
    old2new = torch.cumsum(exist, dim=-1) - exist[0]
    return old2new[labels.long()]


def filter_labels_by_size(labels: torch.Tensor,
                          min_cluster_size: Optional[int] = None,
                          max_cluster_size: Optional[int] = None):
    """ Filter the labels by size """
    if min_cluster_size is None and max_cluster_size is None:
        return labels
    elif (max_cluster_size is not None) and (min_cluster_size is not None):
        sizes = torch.bincount(labels)
        filter_keep = (sizes >= min_cluster_size) * (sizes <= max_cluster_size)
    elif max_cluster_size is not None:
        sizes = torch.bincount(labels)
        filter_keep = (sizes <= max_cluster_size)
    elif min_cluster_size is not None:
        sizes = torch.bincount(labels)
        filter_keep = (sizes >= min_cluster_size)

    old2new = torch.arange(sizes.shape[0]) + 1
    assert old2new.shape == sizes.shape
    old2new[~filter_keep] = 0  # garbage label is zero all the others are > 0.
    new_labels = old2new[labels.long()]  # set some lables to zero
    return remove_label_gaps(new_labels)  # remove the gaps in the counting of labels










# I HAVE LEARNED:
# 1. If I use a lot of negihbours then all methods are roughly equivalent b/c graph becomes ALL-TO-ALL
# 2. Radius=10 means each pixel has 121 neighbours
# 3. CPM does not suffer from the resolution limit which means that it tends to shave off small part from a cell.
# 4. For now I prefer to use a graph with normalized edges,
# modularity and single gigantic cluster (i.e. each_cc_component=False)

class GraphSegmentation(object):
    """
    Produce a consensus segmentation mask by finding communities on a graph.
    Each node is a foreground pixel and each edge is the probability that two pixels belong to the same object.
    """

    def __init__(self, segmentation: Segmentation) -> None:
        """
        Args:
            segmentation: 'class':Segmentation obtained from running the method :meth:`segment_with_tiling`
        """

        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.raw_image = segmentation.raw_image[0].to(self.device)
        self.example_integer_mask = segmentation.integer_mask[0, 0].to(self.device)  # set batch=0, ch=0
        self.fg_prob = segmentation.fg_prob.to(self.device)
        self.sparse_similarity_matrix = segmentation.similarity.sparse_matrix.to(self.device)
        self.similarity_index_matrix = segmentation.similarity.index_matrix.to(self.device)

        # it should be able to handle both DenseSimilarity and SparseSimilarity
        b, c, ni, nj = segmentation.integer_mask.shape
        assert b == c == 1
        assert (1, 1, ni, nj) == segmentation.fg_prob.shape

        self.index_matrix = None
        self.n_fg_pixel = None
        self.i_coordinate_fg_pixel = None
        self.j_coordinate_fg_pixel = None
        self.fingerprint = (None, None)  # min_fg_prob, min_edge_weight
        self._maximum_spanning_tree = None

    @torch.no_grad()
    def partition(self, edge_threshold: float,
                  min_cluster_size: int,
                  min_fg_prob: float=0.1,
                  min_edge_weight: float=0.01):

        maximum_spanning_tree = self._compute_maximum_spanning_tree(min_fg_prob, min_edge_weight)
        filtered_tree = (maximum_spanning_tree > edge_threshold)

        if isinstance(filtered_tree, csr_matrix):
            n_components, labels = scipy.sparse.csgraph.connected_components(filtered_tree,
                                                                             directed=True,
                                                                             connection='weak',
                                                                             return_labels=True)
            tmp_labels = torch.from_numpy(labels).long().to(self.device) # from zero onward

        else:
            raise NotImplementedError

        labels = filter_labels_by_size(labels=tmp_labels, min_cluster_size=min_cluster_size)
        return labels

    @torch.no_grad()
    def partition_2_integer_mask(self, partition: torch.Tensor):
        integer_mask = torch.zeros_like(self.index_matrix, dtype=partition.dtype, device=partition.device)
        integer_mask[self.i_coordinate_fg_pixel, self.j_coordinate_fg_pixel] = partition
        return integer_mask

    @torch.no_grad()
    def _compute_maximum_spanning_tree(self, min_fg_prob: float, min_edge_weight: float):
        current_fingerprint = (min_fg_prob, min_edge_weight)
        if (current_fingerprint != self.fingerprint) or (self._maximum_spanning_tree is None):
            graph = self._similarity_2_graph(min_fg_prob=min_fg_prob,
                                             min_edge_weight=min_edge_weight)
            self._maximum_spanning_tree = self._graph_2_maximum_spanning_tree(graph)
            self.fingerprint = current_fingerprint
        return self._maximum_spanning_tree


    @torch.no_grad()
    def _similarity_2_graph(self,
                            min_fg_prob: float,
                            min_edge_weight: float) -> torch.sparse.FloatTensor:
        """
        Internal functions which creates a graph from the pixel similarity.

        Args:
            min_fg_prob: float, threshold for the fg_probability
            min_edge_weight: float, threshold for the edge weight

        Note:
            Nodes (i.e. pixels) with probability less than :attr:`min_fg_prob` are ignored.
            Edges with weight less than :attr:`min_edge_weight` are ignored.
            Therefore :attr:`min_fg_prob` and :attr:`min_edge_weight` can be used to control the size of the
            graph by pruning unimportant nodes and edges
        """

        assert self.sparse_similarity_matrix._nnz() > 0, "WARNING: Graph is empty. Nothing to do"

        # Map the location with small fg_prob to index = -1
        vertex_mask = self.fg_prob[0, 0] > min_fg_prob
        n_max = torch.max(self.similarity_index_matrix).item()
        transform_index = -1 * torch.ones(n_max + 1, dtype=torch.long, device=self.sparse_similarity_matrix.device)
        transform_index[self.similarity_index_matrix[vertex_mask]] = self.similarity_index_matrix[vertex_mask]

        # Do the remapping (old to medium)
        v_tmp = self.sparse_similarity_matrix._values()
        ij_tmp = transform_index[self.sparse_similarity_matrix._indices()]

        # Do the filtering
        my_filter = (v_tmp > min_edge_weight) * (ij_tmp[0, :] >= 0) * (ij_tmp[1, :] >= 0)
        v = v_tmp[my_filter]
        ij = ij_tmp[:, my_filter]

        # Shift the labels so that there are no gaps (medium to new)
        ij_present = (torch.bincount(ij.view(-1)) > 0)
        self.n_fg_pixel = ij_present.sum().item()
        medium_2_new = (torch.cumsum(ij_present, dim=-1) * ij_present) - 1
        ij_new = medium_2_new[ij]

        # Make a transformation of the index_matrix (old to new)
        transform_index.fill_(-1)
        transform_index[self.sparse_similarity_matrix._indices()[:, my_filter]] = ij_new
        self.index_matrix = transform_index[self.similarity_index_matrix]
        ni, nj = self.index_matrix.shape[-2:]

        i_matrix, j_matrix = torch.meshgrid([torch.arange(ni, dtype=torch.long, device=self.index_matrix.device),
                                             torch.arange(nj, dtype=torch.long, device=self.index_matrix.device)])
        self.i_coordinate_fg_pixel = i_matrix[self.index_matrix >= 0]
        self.j_coordinate_fg_pixel = j_matrix[self.index_matrix >= 0]
        print("created graph")

        return torch.sparse_coo_tensor(indices=ij_new,
                                       values=v,
                                       size=[self.n_fg_pixel, self.n_fg_pixel],
                                       dtype=torch.float,
                                       requires_grad=False,
                                       device=self.device)

    @torch.no_grad()
    def _graph_2_maximum_spanning_tree(self, graph: torch.sparse.FloatTensor) -> csr_matrix:

###        if torch.cuda.is_available():
###            import cudf
###            import cugraph.components.connectivity as cu_connectivity
###            import cugraph.structure.graph as cu_graph
###            import cugraph.tree.minimum_spanning_tree as cu_tree
###
###            df = cudf.DataFrame()
###            df['row'] = graph.indices()[0]
###            df['col'] = graph.indices()[1]
###            df['edge_weights'] = graph.values()
###            G = cu_graph.Graph()
###            G.from_cudf_edgelist(df, source='row', destination='col', edge_attr='edge_weights', renumber=False)
###            maximum_spanning_tree = cu_tree.maximum_spanning_tree(G=G)
###
###
###        else:
        # scipy does not have the maximum spanning tree therefore I do:
        # a. computer the complementary (i.e. 1-x) of the input
        # b. compute maximum_spanning_tree
        # c. compute the complementary (i.e. 1-x) of the tree
        csr = torch_2_csr(graph, complementary=True)
        minimum_spanning_tree = scipy.sparse.csgraph.minimum_spanning_tree(csr)
        maximum_spanning_tree = scipy_sparse_complementary(minimum_spanning_tree)
        return maximum_spanning_tree




###############################################
######
######
######
######    def suggest_resolution_parameter(self,
######                                     window: Optional[tuple] = None,
######                                     min_size: Optional[float] = 20,
######                                     max_size: Optional[float] = None,
######                                     cpm_or_modularity: str = "cpm",
######                                     each_cc_separately: bool = False,
######                                     sweep_range: Optional[Iterable] = None) -> Suggestion:
######        """ This function select the resolution parameter which gives the hightest
######            Intersection Over Union with the target partition.
######            By default the target partition is self.partition_sample_segmask.
######
######            To speed up the calculation the optimal resolution parameter is computed based
######            on a windows of the original image. If window is None the entire image is used. This might be very slow.
######
######            Only CPM is scale invariant.
######            If using modularity the same resolution paerameter will give different results depending on the size of the analyzed window.
######
######            The suggested resolution parameter is NOT necessarily optimal.
######            Try smaller values to undersegment and larger value to oversegment.
######
######            window = (min_row, min_col, max_row, max_col)
######        """
######        # filter by window
######        if window is None:
######            window = [0, 0, self.raw_image.shape[-2], self.raw_image.shape[-1]]
######        else:
######            window = (max(0, window[0]),
######                      max(0, window[1]),
######                      min(self.raw_image.shape[-2], window[2]),
######                      min(self.raw_image.shape[-1], window[3]))
######
######        other_integer_mask = remove_label_gaps(self.example_integer_mask[window[0]:window[2], window[1]:window[3]]).long()
######
######        resolutions = torch.arange(0.5, 10, 0.5) if sweep_range is None else sweep_range
######        iou = torch.zeros(len(resolutions), dtype=torch.float)
######        mi = torch.zeros_like(iou)
######        n_reversible_instances = torch.zeros_like(iou)
######        total_intersection = torch.zeros_like(iou)
######        integer_mask = torch.zeros((resolutions.shape[0], window[2]-window[0],
######                                    window[3]-window[1]), dtype=torch.int)
######        delta_n_cells = torch.zeros(resolutions.shape[0], dtype=torch.int)
######        n_cells = torch.zeros_like(delta_n_cells)
######        sizes_list = list()
######
######        for n, res in enumerate(resolutions):
######            if (n % 10 == 0) or (n == len(resolutions)-1):
######                print("resolution sweep, {0:3d} out of {1:3d}".format(n, resolutions.shape[0]-1))
######
######            p_tmp = self.find_partition_leiden(resolution=res,
######                                               window=window,
######                                               min_size=min_size,
######                                               max_size=max_size,
######                                               cpm_or_modularity=cpm_or_modularity,
######                                               each_cc_separately=each_cc_separately)
######            int_mask = self.partition_2_integer_mask(p_tmp)[window[0]:window[2], window[1]:window[3]]
######            sizes_list.append(p_tmp.sizes.cpu())
######
######            n_cells[n] = len(p_tmp.sizes)-1
######            integer_mask[n] = int_mask.cpu()
######            c_tmp: ConcordanceIntMask = concordance_integer_masks(integer_mask[n].to(other_integer_mask.device,
######                                                                                     other_integer_mask.dtype), other_integer_mask)
######            delta_n_cells[n] = c_tmp.delta_n
######            iou[n] = c_tmp.iou
######            mi[n] = c_tmp.mutual_information
######            n_reversible_instances[n] = c_tmp.n_reversible_instances
######            total_intersection[n] = c_tmp.intersection_mask.sum().float()
######
######        i_max = torch.argmax(iou).item()
######        try:
######            best_resolution = resolutions[i_max].item()
######        except:
######            best_resolution = resolutions[i_max]
######
######        return Suggestion(best_resolution=best_resolution,
######                          best_index=i_max,
######                          sweep_resolution=resolutions,
######                          sweep_mi=mi,
######                          sweep_iou=iou,
######                          sweep_delta_n=delta_n_cells,
######                          sweep_seg_mask=integer_mask,
######                          sweep_sizes=sizes_list,
######                          sweep_n_cells=n_cells)
######
######    # TODO use built-in leiden algorithm instead of leidenalg?
######    def find_partition_leiden(self,
######                              resolution: float,
######                              window: Optional[tuple] = None,
######                              min_size: Optional[float] = 20,
######                              max_size: Optional[float] = None,
######                              cpm_or_modularity: str = "cpm",
######                              each_cc_separately: bool = False,
######                              n_iterations: int = 2,
######                              initial_membership: Optional[numpy.ndarray] = None) -> Partition:
######        """ Find a partition of the graph by greedy maximization of CPM or Modularity metric.
######            The graph can have both normalized and un-normalized weight.
######            The strong recommendation is to use CPM with normalized edge weight.
######
######            The metric can be both cpm or modularity
######            The results are all similar (provided the resolution parameter is tuned correctly).
######
######            If you want to use the suggest_resolution_parameter function with full automatic you should use either:
######            1. CPM with normalized edge weight
######            2. MODULARITY with UN Normalized edge_weight
######
######            You can also pass a sweep_range.
######
######            The resolution parameter can be increased (to obtain smaller communities) or
######            decreased (to obtain larger communities).
######
######            To speed up the calculation the graph partitioning can be done separately for each connected components.
######            This is absolutely ok for CPM metric while a bit questionable for Modularity metric.
######            It is not likely to make much difference either way.
######
######            window has the same convention as scikit image, i.e. window = (min_row, min_col, max_row, max_col)
######        """
######
######        if cpm_or_modularity == "cpm":
######            partition_type = la.CPMVertexPartition
######
######            # Rescale the resolution by some (robust) properties of the full graph
######            # so that the right resolution parameter is about 1
######            n = self.graph["total_nodes"]
######            overall_graph_density = self.graph["total_edge_weight"] * 2.0 / (n * (n - 1))
######            resolution = overall_graph_density * resolution
######
######        elif cpm_or_modularity == "modularity":
######            partition_type = la.RBConfigurationVertexPartition
######        else:
######            raise Exception("Warning!! Argument not recognized. \
######                                       CPM_or_modularity can only be 'CPM' or 'modularity'")
######
######        # Subset graph by connected components and windows if necessary
######        max_label = 0
######        membership = torch.zeros(self.n_fg_pixel, dtype=torch.long, device=self.device)
######        partition_for_subgraphs = self.get_cc_partition() if each_cc_separately else None
######
######        for n, g in enumerate(self.subgraphs_by_partition_and_window(window=window,
######                                                                     partition=partition_for_subgraphs)):
######
######            # With this rescaling the value of the resolution parameter optimized
######            # for a small window can be used to segment a large window
######            if cpm_or_modularity == "modularity":
######                tmp = numpy.sum(g.es["weight"]) / g["total_edge_weight"]
######                resolution = resolution * tmp
######
######            if g.vcount() > 0:
######                # Only if the graph has node I tried to find the partition
######
######                print("find partition internal")
######                p = la.find_partition(graph=g,
######                                      partition_type=partition_type,
######                                      initial_membership=initial_membership,
######                                      weights=g.es['weight'],
######                                      n_iterations=n_iterations,
######                                      resolution_parameter=resolution)
######
######                labels = torch.tensor(p.membership, device=self.device, dtype=torch.long) + 1
######                shifted_labels = labels + max_label
######                max_label += torch.max(labels)
######                membership[g.vs['label']] = shifted_labels
######
######        # TODO: filter_by_size is slow
######        return Partition(sizes=torch.bincount(membership),
######                         membership=membership).filter_by_size(min_size=min_size, max_size=max_size)
######
######    def plot_partition(self, partition: torch.Tensor,
######                       figsize: Optional[tuple] = (12, 12),
######                       window: Optional[tuple] = None,
######                       experiment: Optional[neptune.experiments.Experiment] = None,
######                       neptune_name: Optional[str] = None,
######                       **kargs) -> torch.tensor:
######        """
######            If partition is None it prints the connected components
######            window has the same convention as scikit image, i.e. window = (min_row, min_col, max_row, max_col)
######            kargs can include:
######            density=True, bins=50, range=(10,100), ...
######        """
######
######        if window is None:
######            w = [0, 0, self.raw_image.shape[-2], self.raw_image.shape[-1]]
######            sizes_fg = partition.sizes[1:]  # no background
######        else:
######            sizes = torch.bincount(self.is_vertex_in_window(window=window) * partition.membership)
######            sizes_fg = sizes[1:]  # no background
######            sizes_fg = sizes_fg[sizes_fg > 0]  # since I am filtering the vertex some sizes might become zero
######            w = window
######
######        integer_mask = self.partition_2_integer_mask(partition)[w[0]:w[2], w[1]:w[3]].cpu().long().numpy()  # shape: w, h
######        image = self.raw_image[:, w[0]:w[2], w[1]:w[3]].permute(1, 2, 0).cpu().float().numpy()  # shape: w, h, ch
######        if len(image.shape) == 3 and (image.shape[-1] != 3):
######            image = image[..., 0]
######
######        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=figsize)
######        axes[0, 0].imshow(skimage.color.label2rgb(label=integer_mask,
######                                                  bg_label=0))
######        axes[0, 1].imshow(skimage.color.label2rgb(label=integer_mask,
######                                                  image=image,
######                                                  alpha=0.25,
######                                                  bg_label=0))
######        axes[1, 0].imshow(image)
######        axes[1, 1].hist(sizes_fg.cpu(), **kargs)
######
######        title_partition = 'Partition, #cells -> '+str(sizes_fg.shape[0])
######        axes[0, 0].set_title(title_partition)
######        axes[0, 1].set_title(title_partition)
######        axes[1, 0].set_title("raw image")
######        axes[1, 1].set_title("size distribution")
######
######        fig.tight_layout()
######        #if neptune_name is not None:
######        #    log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
######        plt.close(fig)
######        return fig
######
######
######