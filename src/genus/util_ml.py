import torch
import numpy
import torch.nn.functional as F
from torch.distributions.utils import broadcast_all
from typing import Union, Optional, Tuple
from collections import OrderedDict, deque
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from .util import invert_convert_to_box_list, convert_to_box_list, are_broadcastable
from .util_vis import plot_img_and_seg
from .namedtuple import DIST


class MetricsAccumulator(object):
    """
    Accumulate a tuple or dictionary into an OrderDictionary. The intended uses is to accumulate the metrics
    from the mini-batches into a dictionary so that at the end of each epoch we can report the averaged values.

    Returns:
        An ordered dictionary with the
    """

    def __init__(self):
        """
        Initialize an empty accumulator object.
        """
        super().__init__()
        self._counter = 0
        self._dict_accumulate = OrderedDict()

    def _accumulate_key_value(self, _key, _value, counter_increment):

        if isinstance(_value, torch.Tensor):
            x = _value.detach().item()
        elif isinstance(_value, float):
            x = _value
        elif isinstance(_value, numpy.ndarray):
            x = _value
        else:
            raise Exception("value of unrecognized type", _key, type(_value))

        try:
            self._dict_accumulate[_key] = x * counter_increment + self._dict_accumulate.get(_key, 0)
        except ValueError:
            # often the case if accumulating two numpy array of different sizes
            pass

    def accumulate(self, source: Union[tuple, dict], counter_increment: int = 1):
        """
        Accumulate the source into the an internal Ordered_Dictionary and increase the counter
        by :attr:`counter_increment`.
        """
        self._counter += counter_increment

        if isinstance(source, tuple):
            for key in source._fields:
                value = getattr(source, key)
                self._accumulate_key_value(key, value, counter_increment)
        elif isinstance(source, dict):
            for key, value in source.items():
                self._accumulate_key_value(key, value, counter_increment)
        else:
            raise Exception("source type is not recongnized", type(source))

    def get_average(self) -> OrderedDict:
        """
        Returns:
             An OrderDictionary with the averaged quantities by dividing the accumulate quantities
             by the current value of counter.
        """
        tmp = self._dict_accumulate.copy()
        for k, v in self._dict_accumulate.items():
            tmp[k] = v / self._counter
        return tmp

    def set_value(self, key, value):
        """ Set a key value pair in the internal OrderDictionary to do the accumulation """
        self._dict_accumulate[key] = value


class SimilarityKernel(torch.nn.Module):
    """
    Square gaussian kernel with learnable parameters (weight and length_scale).
    """

    def __init__(self,
                 length_scale: float,
                 weight: float,
                 length_scale_min_max: Optional[Tuple[float, float]] = None,
                 weight_min_max: Optional[Tuple[float, float]] = None,
                 pbc: bool = False,
                 eps: float = 1E-4):
        """
        Args:
            length_scale: Initial value for learnable parameter specifying the length scale of the kernel.
            weight: Initial value for learnable parameter specifying the weight of the kernel.
            length_scale_min_max: Tuple with the minimum and maximum allowed value for the :attr:`length_scale`.
                During training the value of length_scale is clamped back inside the allowed range.
            weight_min_max: Tuple with the minimum and maximum allowed value for the :attr:`weight`.
                During training the value of weight is clamped back inside the allowed range.
            pbc: If True periodic boundary condition (pbc) are used (i.e. the grid points are placed on a torus),
                if False open boundary condition are used. It is safer to set :attr:`pbc` to False b/c the
                matrix might become ill-conditioned otherwise.
            eps: Small regularization factor which is added to the diagonal to avoid ill-conditioning. It should be set
                to 1E-4.

        Note:
            Large values of :attr:`length_scale` means that far-away points have strong similarity.

        Note:
            When used in combination with :class:`FiniteDPP` the value of :attr:`length_scale` and :attr:`weight`
            determine the typical number of points in a random DPP sample. In particular :attr:`weight`
            can be seen as a chemical potential. The higher this value the more points are generated.
        """

        super().__init__()
        # Checks
        assert length_scale > 0, "Error: length_scale MUST BE > 0"
        assert weight > 0, "Error: weight MUST BE > 0"

        if length_scale_min_max is None:
            length_scale_min_max = [min(2.0, length_scale), max(50.0, length_scale)]
        else:
            assert (length_scale_min_max[0] > 0) and (length_scale_min_max[0] < length_scale) and \
                   (length_scale_min_max[-1] >
                    length_scale), "length_scale and/or lenght_scale_min_max have invalid values"

        if weight_min_max is None:
            weight_min_max = [min(2.0, weight), max(50.0, weight)]
        else:
            assert (weight_min_max[0] > 0) and (weight_min_max[0] < weight) and (weight_min_max[-1] > weight), \
                "weigth and/or weigth_min_max have invalid values"

        # Set the members variables
        self.eps = eps
        self.pbc = pbc
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.length_scale_min = length_scale_min_max[0]
        self.length_scale_max = length_scale_min_max[-1]
        self.weight_min = weight_min_max[0]
        self.weight_max = weight_min_max[-1]
        self.length_scale_value = torch.nn.Parameter(data=torch.tensor(length_scale, device=self.device,
                                                                       dtype=torch.float), requires_grad=True)
        self.weight_value = torch.nn.Parameter(data=torch.tensor(weight, device=self.device, dtype=torch.float),
                                               requires_grad=True)

    def _compute_d2_diag(self, n_width: int, n_height: int):
        with torch.no_grad():
            ix_array = torch.arange(start=0, end=n_width, dtype=torch.int, device=self.device)
            iy_array = torch.arange(start=0, end=n_height, dtype=torch.int, device=self.device)
            ix_grid, iy_grid = torch.meshgrid([ix_array, iy_array])
            map_points = torch.stack((ix_grid, iy_grid), dim=-1)  # n_width, n_height, 2
            locations = map_points.flatten(start_dim=0, end_dim=-2)  # (n_width*n_height, 2)
            d = (locations.unsqueeze(-2) - locations.unsqueeze(-3)).abs()  # (n_width*n_height, n_width*n_height, 2)
            if self.pbc:
                d_pbc = d.clone()
                d_pbc[..., 0] = -d[..., 0] + n_width
                d_pbc[..., 1] = -d[..., 1] + n_height
                d2 = torch.min(d, d_pbc).pow(2).sum(dim=-1).float()
            else:
                d2 = d.pow(2).sum(dim=-1).float()

            values = self.eps * torch.ones(d2.shape[-2], dtype=torch.float, device=self.device)
            diag = torch.diag_embed(values, offset=0, dim1=-2, dim2=-1)
            return d2, diag

    def get_l_w(self):
        """ Returns the current value of length scale and weight of the similarity kernel """
        return self.length_scale_value, self.weight_value

    def _clamp_and_get_l_w(self):
        """ Clamps the parameters before they are used to compute the similarity matrix. """
        self.length_scale_value.data.clamp_(min=self.length_scale_min, max=self.length_scale_max)
        self.weight_value.data.clamp_(min=self.weight_min, max=self.weight_max)
        return self.length_scale_value, self.weight_value

    def forward(self, n_width: int, n_height: int) -> torch.Tensor:
        """
        Generates a similarity matrix according to the learnable parameters of the square exponential kernel.

        Args:
            n_width: integer size of the grid width
            n_height: integer size of the grid height

        Returns:
            A similarity matrix of shape :math:`(N, N)` where
            :math:`N =` :attr:`n_width` :math:`\\times` :attr:`n_height`.
        """
        l, w = self._clamp_and_get_l_w()
        d2, diag = self._compute_d2_diag(n_width=n_width, n_height=n_height)
        return diag + w * torch.exp(-0.5*d2/(l*l))


class FiniteDPP(Distribution):
    """
    Determinental Point Process (DPP) defined on a finite set of discrete points.
    You can draw random samples and compute the log_prob of a given point configuration.

    Note:
        See https://dppy.readthedocs.io/en/latest/ for general information about FiniteDPP

    Note:
        It relies on svd decomposition which can become unstable on GPU or CPU, see
        https://github.com/pytorch/pytorch/issues/28293
    """
    arg_constraints = {'K': constraints.positive_definite,
                       'L': constraints.positive_definite,
                       'eigen_L': constraints.positive_definite}
    support = constraints.boolean
    has_rsample = False

    def __init__(self, K=None, L=None, eigen_L=None, validate_args=None):
        """
        A Finite DPP is FULLY specified by the correlation matrix (K), the likelihood matrix (L)
        and the L_eigenvalues. All three are necessary to perform different operations.
        For example:
        (1) the computation of the log_probability of a configuration requires the L matrix and L_eigenvalues.
        (2) Drawing a new random sample requires the K matrix

        The MINIMAL specification of the DPP is by either the K or L matrix. The remaining information can be obtained
        via SVD decomposition. If all three elements are specified, no svd decomposition will be performed.

        Args:
            K: correlation matrix of shape :math:`(*, n, n)` is positive-semidefinite, symmetric with
                eigenvalues in :math:`[0,1]`. It might be difficult to ensure that the eigenvalues are in :math:`[0,1]`,
                therefore it is recommended to define the DPP via the likelihood matrix :attr:`L`.
            L: likelihood matrix of shape :math:`(*, n, n)` is positive-semidefinite, symmetric with
                eigenvalues in :math:`>= 0`. Any expoentially decaying similarity kernel will give rise
                to a valid likelihood matrix.
            eigen_L: the eigenvalues of the L matrix

        Note:
            This class is often used in combination with :class:`SimilarityKernel`.

        Examples:
            >>> kernel = SimilarityKernel(length_scale=10.0, weight=1.0)
            >>> likelihood_matrix = kernel.forward(n_width=20, n_height=20)
            >>> DPP = FiniteDPP(L=likelihood_matrix)
        """

        if (K is None) and (L is None):
            raise Exception("One between K and L need to be defined")
        elif L is None:
            self._L = None
            self._K = 0.5 * (K + K.transpose(-1, -2))
            batch_shape, event_shape = self._K.shape[:-2], self._K.shape[-1:]
        elif K is None:
            self._K = None
            self._L = 0.5 * (L + L.transpose(-1, -2))
            batch_shape, event_shape = self._L.shape[:-2], self._L.shape[-1:]
        else:
            # neither is none
            assert L.shape == K.shape
            self._L = L
            self._K = K
            batch_shape, event_shape = self._L.shape[:-2], self._L.shape[-1:]

        self._eigen_l = eigen_L
        super(FiniteDPP, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def K(self):
        if self._K is None:
            # print("svd from L")
            try:
                u, s_l, v = torch.svd(self._L)
            except:
                # torch.svd may have convergence issues for GPU and CPU.
                u, s_l, v = torch.svd(self._L + 1e-3 * self._L.mean() * torch.ones_like(self._L))
            s_k = s_l / (1.0 + s_l)
            self._K = torch.matmul(u * s_k.unsqueeze(-2), v.transpose(-1, -2))
            self._eigen_l = s_l
        return self._K

    @property
    def L(self):
        if self._L is None:
            # print("svd from K")
            try:
                u, s_k, v = torch.svd(self._K)
            except:
                # torch.svd may have convergence issues for GPU and CPU.
                u, s_k, v = torch.svd(self._K + 1e-3 * self._K.mean() * torch.ones_like(self._K))
            self._eigen_l = s_k / (1.0 - s_k)
            self._L = torch.matmul(u * self._eigen_l.unsqueeze(-2), v.transpose(-1, -2))
        return self._L

    @property
    def eigen_l(self):
        if self._eigen_l is None:
            # print("eigen from L")
            try:
                u, s_l, v = torch.svd(self.L)
            except:
                # torch.svd may have convergence issues for GPU and CPU.
                u, s_l, v = torch.svd(self.L + 1e-3 * self.L.mean() * torch.ones_like(self.L))
            s_k = s_l / (1.0 + s_l)
            self._K = torch.matmul(u * s_k.unsqueeze(-2), v.transpose(-1, -2))
            self._eigen_l = s_l
        return self._eigen_l

    @property
    def n_mean(self):
        p = self.eigen_l / (1 + self.eigen_l)
        return p.sum()

    @property
    def n_variance(self):
        p = self.eigen_l / (1 + self.eigen_l)
        return torch.sum(p*(1-p))

    @property
    def n_stddev(self):
        return self.n_variance.sqrt()

    def expand(self, batch_shape, _instance=None):
        """
        :meta private:
        """
        new = self._get_checked_instance(FiniteDPP, _instance)
        batch_shape = torch.Size(batch_shape)
        kernel_shape = batch_shape + self.event_shape + self.event_shape
        value_shape = batch_shape + self.event_shape
        new._eigen_l = None if self._eigen_l is None else self._eigen_l.expand(value_shape)
        new._L = None if self._L is None else self._L.expand(kernel_shape)
        new._K = None if self._K is None else self._K.expand(kernel_shape)
        super(FiniteDPP, new).__init__(batch_shape,
                                       self.event_shape,
                                       validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        """
        Draw random samples from the DPP.

        Args:
            sample_shape: specify how many independent samples to draw.

        Returns:
            The binary configaration of shape = sample_shape + batch_shape + event_shape where
            the batch_shape and event_shape are determined by the shape of the correlation matrix (K) or
            likelihood matrix (L) used to initialize the instance and :attr:`sample_shape` can be passed explicitely.

        Note:
            The samples are NOT differentiable.

        Examples:
             >>> KERNEL = SimilarityKernel(length_scale=2.3, weight=4.5)
             >>> likelihood_matrix = KERNEL.forward(n_height=10, n_width=10)
             >>> DPP = FiniteDPP(L=likelihood_matrix)
             >>> value = DPP.sample(sample_shape=torch.Size([2]))
             >>> print(value.shape)
        """
        shape_value = self._extended_shape(sample_shape)  # shape = sample_shape + batch_shape + event_shape
        shape_kernel = shape_value + self._event_shape  # shape = sample_shape + batch_shape + event_shape + event_shape

        with torch.no_grad():
            k_matrix = self.K.expand(shape_kernel).clone()
            value = torch.zeros(shape_value, dtype=torch.bool, device=k_matrix.device)
            rand = torch.rand(shape_value, dtype=k_matrix.dtype, device=k_matrix.device)

            for j in range(rand.shape[-1]):
                c = rand[..., j] < k_matrix[..., j, j]
                value[..., j] = c
                k_matrix[..., j, j] -= torch.tensor(~c, dtype=k_matrix.dtype, device=k_matrix.device)
                k_matrix[..., j + 1:, j] /= k_matrix[..., j, j].unsqueeze(-1)
                k_matrix[..., j + 1:, j + 1:] -= \
                    k_matrix[..., j + 1:, j].unsqueeze(-1) * k_matrix[..., j, j + 1:].unsqueeze(-2)

            return value

    def log_prob(self, value):
        """
        Computes the log_probability of the configuration given the current value of the correlation matrix (K) or
        likelihood matrix (L) used to initialize the instance. It requires the L matrix and L_eigenvalues

        Args:
            value: The binarized configuration of shape :math:`(*, \\text{event_shape})`

        Returns:
            The lop_probability of the configuration of shape :math:`(*)`

        Note:
            The log_probability is computed as :math:`\\text{log_prob} = \\text{logdet}(L_s) - \\text{logdet}(L+I)`
            where :math:`L_s` is the likelihood matrix subset to the row and columns corresponding to :attr:`value` = 1.

            The first term can be computed by embedding :math:`L_s` into a sufficiently large identity matrix (so that
            the tensor is not ragged) and using torch.logdet.
            The second term can be computed from the eigenvalues of L in the following way:

            .. math::
                \\text{logdet}(L+I) = \\text{log prod eigen}(L+I) =
                \\text{sum log eigen}(L+1) = \\text{sum log}\\left( \\text{eigen}(L) + 1\\right)

        Examples:
            >>> KERNEL = SimilarityKernel(length_scale=2.3, weight=4.5)
            >>> likelihood_matrix = KERNEL.forward(n_width=20, n_height=20)
            >>> DPP = FiniteDPP(L=likelihood_matrix)
            >>> value = DPP.sample(sample_shape=torch.Size([2]))  # -> torch.Size([2, 400])
            >>> logp = DPP.log_prob(value=value)
            >>> print(logp.shape)  # -> torch.Size([2])
        """
        assert are_broadcastable(value, self.L[..., 0])
        assert self.L.device == value.device
        assert value.dtype == torch.bool

        if self._validate_args:
            self._validate_sample(value)

        # Reshapes
        value = value.unsqueeze(0)  # trick to make it work even if not-batched
        independet_dims = list(value.shape[:-1])
        value = value.flatten(start_dim=0, end_dim=-2)  # *, event_shape
        L = self.L.expand(independet_dims + [-1, -1]).flatten(start_dim=0, end_dim=-3)  # *, event_shape, event_shape

        # I need to compute the logdet of square matrix of different shapes
        # Select sub-row and columns and embed everything inside a larger identity matrix since
        #     | A  B  0 |
        # det | C  D  0 | = determinant of the sub_matrix
        #     | 0  0  1 |
        n_c = torch.sum(value, dim=-1)
        n_max = n_c.max().item()
        matrix = torch.eye(n_max, dtype=L.dtype, device=L.device).expand(L.shape[-3], n_max, n_max).clone()

        # Option without for loops by fancy slicing
        mask_s = (torch.arange(1, n_max + 1, dtype=n_c.dtype, device=n_c.device).view(1, -1) <= n_c.view(-1, 1))
        mask_ss = torch.logical_and(mask_s.unsqueeze(-1), mask_s.unsqueeze(-2))
        mask_ll = torch.logical_and(value.unsqueeze(-1), value.unsqueeze(-2))
        matrix[mask_ss] = L[mask_ll]

        # Compute the log_determinant of the full and submatrices
        logdet_Ls = torch.logdet(matrix).view(independet_dims)
        logdet_L_plus_I = (self.eigen_l + 1).log().sum(dim=-1)  # sum over the event_shape
        return (logdet_Ls - logdet_L_plus_I).squeeze(0)  # trick to make it work even if not-batched


class Grid_DPP(torch.nn.Module):
    """ Wrapper around :class:`SimilariyKenrnel` and :class:`FiniteDPP` for easy work with 2D grids of points """

    def __init__(self,
                 length_scale: float,
                 weight: float,
                 length_scale_min_max: Optional[Tuple[float, float]] = None,
                 weight_min_max: Optional[Tuple[float, float]] = None,
                 pbc: bool = False,
                 eps: float = 1E-4,
                 learnable_params: bool = True):
        """ See :class:`SimilarityKernel` for an explanation of the arguments. If :attr:`learnable_params` = False
            the parameters :attr:`lenght_scale` and :attr:`weight` are fixed.
        """

        super().__init__()
        self.similiraty_kernel = SimilarityKernel(length_scale=length_scale,
                                                  weight=weight,
                                                  length_scale_min_max=length_scale_min_max,
                                                  weight_min_max=weight_min_max,
                                                  pbc=pbc,
                                                  eps=eps)
        self.finite_dpp:  Optional[FiniteDPP] = None
        self.fingerprint = (None, None, None, None)
        self.learnable_params = learnable_params
        if self.learnable_params:
            raise NotImplementedError("At the moment, \
            the lenght_scale and weight of the DPP prior need to be fixed by the users")

    @property
    def n_mean(self):
        if self.finite_dpp is None:
            raise Exception("You need to draw a random sample first to fix the size of the grid")
        return self.finite_dpp.n_mean

    @property
    def n_variance(self):
        if self.finite_dpp is None:
            raise Exception("You need to draw a random sample first to fix the size of the grid")
        return self.finite_dpp.n_variance

    @property
    def n_stddev(self):
        if self.finite_dpp is None:
            raise Exception("You need to draw a random sample first to fix the size of the grid")
        return self.finite_dpp.n_stddev

    def sample(self, size: torch.Size):
        """
        Draw a random sample of size torch.Size according to the current values of length_scale, weight.
        Note that size must be at least 2D.
        The samples are not differentiable.
        """
        assert len(size) >= 2
        current_figerprint = (self.similiraty_kernel.length_scale_value.data.item(),
                              self.similiraty_kernel.weight_value.data.item(),
                              size[-2], size[-1])

        with torch.no_grad():
            if current_figerprint != self.fingerprint:
                similarity = self.similiraty_kernel.forward(n_width=size[-2], n_height=size[-1])
                self.finite_dpp = FiniteDPP(L=similarity)
                self.fingerprint = current_figerprint

            c_all = self.finite_dpp.sample(sample_shape=size[:-2]).view(size)
            return c_all

    def log_prob(self, value: torch.Tensor):
        """ Compute the log_prob of a configuration. Note that value need to be at least 2D.
            The log_prob is differentiable w.r.t. the parameters of the similarity kernel but not :attr:`value`.
        """
        assert len(value.shape) >= 2
        current_figerprint = (self.similiraty_kernel.length_scale_value.data.item(),
                              self.similiraty_kernel.weight_value.data.item(),
                              value.shape[-2], value.shape[-1])

        if current_figerprint != self.fingerprint:
            if self.learnable_params:
                similarity = self.similiraty_kernel.forward(n_width=value.shape[-2], n_height=value.shape[-1])
                self.finite_dpp = FiniteDPP(L=similarity)
                self.fingerprint = current_figerprint
            else:
                with torch.no_grad():
                    similarity = self.similiraty_kernel.forward(n_width=value.shape[-2], n_height=value.shape[-1])
                    self.finite_dpp = FiniteDPP(L=similarity)
                    self.fingerprint = current_figerprint

        # Identical can immediately draw a sample
        logp = self.finite_dpp.log_prob(value=value.flatten(start_dim=-2))
        return logp


class ConditionalRandomCrop(object):
    """
    Crop a torch Tensor at random locations to obtain output of given size. The random crop is accepted only if it
    contains the Region Of Interest (ROI).
    """

    def __init__(self, desired_w: int, desired_h: int, min_roi_fraction: float = 0.0, n_crops_per_image: int = 1):
        """
        Args:
            desired_w: integer specifying the desired width of the crop
            desired_h: integer specifying the desired height of the crop
            min_roi_fraction: minimum threshold of the Region of Interest (ROI).
                Random crops with less that this amount of ROI will be disregarded.
            n_crops_per_image: number of random crops to generate for each image
        """
        super().__init__()
        self.desired_w = desired_w
        self.desired_h = desired_h
        self.min_roi_fraction = min_roi_fraction
        self.desired_area = desired_w * desired_h
        self.n_crops_per_image = n_crops_per_image

    @staticmethod
    def _get_smallest_corner_for_crop(w_raw: int, h_raw: int, w_desired: int, h_desired: int):
        """ Generate the random location of the lower-left corner of the crop """
        assert w_desired <= w_raw
        assert h_desired <= h_raw

        if w_raw == w_desired and h_raw == h_desired:
            return 0, 0
        else:
            i = torch.randint(low=0, high=w_raw - w_desired + 1, size=[1]).item()
            j = torch.randint(low=0, high=h_raw - h_desired + 1, size=[1]).item()
            return i, j

    def get_index(self,
                  img: torch.Tensor,
                  roi_mask: Optional[torch.Tensor] = None,
                  cum_sum_roi_mask: Optional[torch.Tensor] = None,
                  n_crops_per_image: Optional[int] = None):
        """
        :meta private:
        Create a list of n_crops_per_image tuples indicating the location of the lower-left corner of the random crops.

        Args:
            img: tensor of shape :math:`(*,C,W,H)`
            roi_mask: boolean tensor of shape :math:`(*,1,W,H)` indicating the region of interest (ROI)
            cum_sum_roi_mask: tensor of shape :math:`(*,1,W,H)` with the double integral (both along row and column)
                of the ROI
            n_crops_per_image: how many random crops to generate for each image in the input batch

        Returns:
            A list of n_crops_per_image tuples indicating the location of the lower-left corner of the random crops
        """
        n_crops_per_image = self.n_crops_per_image if n_crops_per_image is None else n_crops_per_image

        if roi_mask is not None and cum_sum_roi_mask is not None:
            raise Exception("Only one between roi_mask and cum_sum_roi_mask can be specified")

        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if cum_sum_roi_mask is not None and len(cum_sum_roi_mask.shape) == 3:
            cum_sum_roi_mask = cum_sum_roi_mask.unsqueeze(0)
        if roi_mask is not None and len(roi_mask.shape) == 3:
            roi_mask = roi_mask.unsqueeze(0)

        assert len(img.shape) == 4
        assert (roi_mask is None or len(roi_mask.shape) == 4)
        assert (cum_sum_roi_mask is None or len(cum_sum_roi_mask.shape) == 4)

        with torch.no_grad():

            bij_list = deque()
            for b in range(img.shape[0]):
                for n in range(n_crops_per_image):
                    fraction = 0
                    while fraction < self.min_roi_fraction:
                        i, j = self._get_smallest_corner_for_crop(w_raw=img[b].shape[-2],
                                                                  h_raw=img[b].shape[-1],
                                                                  w_desired=self.desired_w,
                                                                  h_desired=self.desired_h)

                        if cum_sum_roi_mask is not None:
                            term1 = cum_sum_roi_mask[b, 0, i + self.desired_w - 1, j + self.desired_h - 1].item()
                            term2 = 0 if i < 1 else cum_sum_roi_mask[b, 0, i - 1, j + self.desired_h - 1].item()
                            term3 = 0 if j < 1 else cum_sum_roi_mask[b, 0, i + self.desired_w - 1, j - 1].item()
                            term4 = 0 if (i < 1 or j < 1) else cum_sum_roi_mask[b, 0, i - 1, j - 1].item()
                            fraction = float(term1 - term2 - term3 + term4) / self.desired_area
                        elif roi_mask is not None:
                            fraction = roi_mask[b, 0, i:i+self.desired_w,
                                       j:j+self.desired_h].sum().float()/self.desired_area
                        else:
                            fraction = 1.0

                    bij_list.append([b, i, j])
        return bij_list

    def collate_crops_from_list(self, img: torch.Tensor, bij_list: list):
        """
        Makes a batch of cropped images.

        Args:
            img: tensor of shape :math:`(B,C,W,H)` with the images to crop
            bij_list: list of tuple of type :math:`[b,i,j]` indicating the image index and the location of the
                lower-left corner of the random crop

        Returns:
            A batch with the cropped images of shape :math:`(N,C,\\text{desired}_w,\\text{desired}_h)`
            where N is the length of the list :attr:`bij_list`.

        :meta private:
        """
        return torch.stack([img[b, :, i:i+self.desired_w, j:j+self.desired_h] for b, i, j in bij_list], dim=-4)

    def crop(self,
             img: torch.Tensor,
             roi_mask: Optional[torch.Tensor] = None,
             cum_sum_roi_mask: Optional[torch.Tensor] = None,
             n_crops_per_image: Optional[int] = None):
        """
        Crop a image or batch of images to generate :attr:`n_crops_per_imag` random crops for each image.

        Args:
            img: image or batch of images of shape :math:`(*,c,W,H)`
            roi_mask: boolean tensor of shape :math:`(*,1,W,H)` indicating the region of interest (ROI)
            cum_sum_roi_mask: tensor of shape :math:`(*,1,W,H)` with the double integral (both along row and column)
                of the ROI
            n_crops_per_image: how many random crops to generate for each image in the batch
                (this is a data_augmentation multiplicative factor)

        Note:
            If neither :attr:`roi_mask` nor :attr:`cum_sum_roi_mask` are specified the entire image is considered ROI.

        Note:
            It can be used once, at the beginning, to create a test-dataset from a large image or on the fly to create a
            continously changing training-dataset from a large image.
            Used often in combination with :class:`SpecialDataSet`

        Raises:
            Exception: If both :attr:`roi_mask` and :attr:`cum_sum_roi_mask` are specified

        Examples:
            >>> # Create the random crops once and add them in the test_dataloader
            >>> crop_size, test_dataset_size, minibatch_size = 80, 128, 64
            >>> conditional_crop_test = ConditionalRandomCrop(desired_w=crop_size,
            >>>                                               desired_h=crop_size,
            >>>                                               min_roi_fraction=0.1,
            >>>                                               n_crops_per_image=test_dataset_size)
            >>> test_data = conditional_crop_test.crop(img=one_large_image, roi_mask=one_large_image_roi_mask)
            >>> test_loader = SpecialDataSet(img=test_data, batch_size=minibatch_size)
            >>> # Create random crops on the fly (i.e. data-augmentation) to create a constantly changing train_dataset
            >>> conditional_crop_train = ConditionalRandomCrop(desired_w=crop_size,
            >>>                                                desired_h=crop_size,
            >>>                                                min_roi_fraction=0.9,
            >>>                                                n_crops_per_image=minibatch_size)
            >>> train_loader = SpecialDataSet(img=one_large_image,
            >>>                               roi_mask=one_large_image_roi_mask)
            >>>                               data_augmentation=conditional_crop_train,
            >>>                               batch_size=minibatch_size)
        """

        n_crops_per_image = self.n_crops_per_image if n_crops_per_image is None else n_crops_per_image
        bij_list = self.get_index(img, roi_mask, cum_sum_roi_mask, n_crops_per_image)
        return self.collate_crops_from_list(img, list(bij_list))


class SpecialDataSet(object):
    """
    Dataset and dataloader combined into single class with extra features.

    Note:
        This class is useful for small datasets which fit into CPU or GPU memory.

    Note:
        It can be used in combination with :class:`ConditionalRandomCrop` to create a
        dataset out of a single large image.
    """

    def __init__(self,
                 x: torch.Tensor,
                 x_roi: Optional[torch.Tensor] = None,
                 y: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None,
                 data_augmentation: Optional[ConditionalRandomCrop] = None,
                 store_in_cuda: bool = False,
                 batch_size: int = 4,
                 drop_last: bool = False,
                 shuffle: bool = True):
        """
        Args:
            x: the underlying image or batch of images which composed the dataset of shape :math:`(B,C,W,H)`
            x_roi: boolean tensor indicating the Region of Interest (ROI) of shape :math:`(B,1,W,H)`
            y: integer tensor with the pixel-level labels of shape :math:`(B,1,W,H)`
            labels: integer tensor with the image-level labels of shape :math:`(B)`
            data_augmentation: if specified both x and y are processed by this function on the fly every
                time the :meth:`__getitem__` is called.
            store_in_cuda: if true the dataset is stored into GPU memory, if false the dataset is stored in CPU memory.
                Either way the dataset is stored in memory for faster processing.
                This means that this class is useful for small dataset.
            batch_size: default value of the size of the minibatch which will be loaded every
                time :meth:`__iter__` is called
            drop_last: if true the last minibatch in each epoch will be dropped if the dataset size is not
                an exact multiple of the minibatch size.
            shuffle: it true the order of the element is randomized. If false the element are loaded in exactly the same
                order at each epoch.

        Note:
            The underlying data is all stored in memory for faster processing
            (either CPU or GPU depending on :attr:`store_in_cuda`). This means that this class is usefull only for
            relatively small datasets.

        Note:
            If :attr:`data_augmentation` is equal to :class:`ConditionalRandomCrop` the dataloader returns different
            random crops at every call thus implementing data augmentation.

        Examples:
            >>> # Create the random crops once and add them in the test_dataloader
            >>> crop_size, test_dataset_size, minibatch_size = 80, 128, 64
            >>> conditional_crop_test = ConditionalRandomCrop(desired_w=crop_size,
            >>>                                               desired_h=crop_size,
            >>>                                               min_roi_fraction=0.1,
            >>>                                               n_crops_per_image=test_dataset_size)
            >>> test_data = conditional_crop_test.crop(img=one_large_image, roi_mask=one_large_image_roi_mask)
            >>> # save the small test_dataset on GPU memory
            >>> test_loader = SpecialDataSet(img=test_data,
            >>>                              store_in_cuda=True,
            >>>                              shuffle=False,
            >>>                              drop_last=False)
            >>> # Create random crops on the fly (i.e. data-augmentation) to create a constantly changing train_dataset
            >>> conditional_crop_train = ConditionalRandomCrop(desired_w=crop_size,
            >>>                                                desired_h=crop_size,
            >>>                                                min_roi_fraction=0.9,
            >>>                                                n_crops_per_image=minibatch_size)
            >>> # save the large image on CPU memory and every time draw random crops from it.
            >>> train_loader = SpecialDataSet(img=one_large_image,
            >>>                               roi_mask=one_large_image_roi_mask,
            >>>                               store_in_cuda=False,
            >>>                               shuffle=True,
            >>>                               data_augmentation=conditional_crop_train,
            >>>                               batch_size=minibatch_size)
        """
        assert len(x.shape) == 4
        assert (x_roi is None or len(x_roi.shape) == 4)
        assert (labels is None or labels.shape[0] == x.shape[0])

        storing_device = torch.device('cuda') if store_in_cuda else torch.device('cpu')

        self.drop_last = drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Expand the dataset so that I can do one crop per image
        if data_augmentation is None:
            new_batch_size = x.shape[0]
            self.data_augmentaion = None
        else:
            new_batch_size = x.shape[0] * data_augmentation.n_crops_per_image
            self.data_augmentaion = data_augmentation

        # Note that data is first moved to the storing device and the expanded. This makes sure that memory footprint
        # remains low
        self.x = x.to(storing_device).detach().expand(new_batch_size, -1, -1, -1)

        if labels is None:
            self.labels = -1*torch.ones(self.x.shape[0], device=storing_device).detach()
        else:
            self.labels = labels.to(storing_device).detach()
        self.labels = self.labels.expand(new_batch_size)

        if x_roi is None:
            self.x_roi = None
            self.x_roi_cumulative = None
        else:
            self.x_roi = x_roi.to(storing_device).detach().expand(new_batch_size, -1, -1, -1)
            self.x_roi_cumulative = x_roi.to(storing_device).detach().cumsum(dim=-1).cumsum(
                dim=-2).expand(new_batch_size, -1, -1, -1)

        if y is None:
            self.y = None
        else:
            self.y = y.to(storing_device).detach().expand(new_batch_size, -1, -1, -1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index: torch.Tensor):
        """
        :meta public:

        Args:
            torch.Tensor of type long specifying the index of the images to load. Each entry should be between
                0 and self.__len__(). Repeated index are allowed.

        Returns:
             A tuple with x, y, labels, index. Where x,y are the images and labels and index are scalars.
             The size of the output is determined by the length of :attr:`index`.
        """
        assert isinstance(index, torch.Tensor)

        if self.data_augmentaion is None:
            x = self.x[index]
            y = -1*torch.ones_like(x) if self.y is None else self.y[index]
            return x, y, self.labels[index], index
        else:
            bij_list = []
            for i in index:
                bij_list += self.data_augmentaion.get_index(img=self.x[i],
                                                            cum_sum_roi_mask=self.x_roi_cumulative[i],
                                                            n_crops_per_image=1)
            x = self.data_augmentaion.collate_crops_from_list(self.x, bij_list)
            y = -1*torch.ones_like(x) if self.y is None \
                else self.data_augmentaion.collate_crops_from_list(self.y, bij_list)
            return x, y, self.labels[index], index

    def __iter__(self, batch_size=None, drop_last=None, shuffle=None):
        # If not specified use defaults
        batch_size = self.batch_size if batch_size is None else batch_size
        drop_last = self.drop_last if drop_last is None else drop_last
        shuffle = self.shuffle if shuffle is None else shuffle

        # Actual generation of iterator
        n_max = max(1, self.__len__() - (self.__len__() % batch_size) if drop_last else self.__len__())
        index = torch.randperm(self.__len__()).long() if shuffle else torch.arange(self.__len__()).long()
        for pos in range(0, n_max, batch_size):
            yield self.__getitem__(index[pos:pos + batch_size])

    def load(self, batch_size: Optional[int] = None, index: Optional[torch.Tensor] = None):
        """
        Load a batch of images

        Args:
            batch_size: number of images to randomly load from the dataset
            index: torch.Tensor of type long specifying the index of the images to load. Each entry should be between
                0 and self.__len__(). Repeated index are allowed. The batch_size will be equal to the length of
                :attr:`index`.

        Returns:
            A batch of images
        """
        if (batch_size is None and index is None) or (batch_size is not None and index is not None):
            raise Exception("Only one between batch_size and index must be specified")
        random_index = torch.randperm(self.__len__(), dtype=torch.long, device=self.x.device)[:batch_size]
        index = random_index if index is None else index
        return self.__getitem__(index)

    def check_batch(self, batch_size: int = 8):
        """
        Print some information about the dataset and load a random batch of images

        Args:
            batch_size: the size of the batxh of images to load
        """
        print("Dataset lenght:", self.__len__())
        print("img.shape", self.x.shape)
        print("img.dtype", self.x.dtype)
        print("img.device", self.x.device)
        random_index = torch.randperm(self.__len__(), dtype=torch.long, device=self.x.device)[:batch_size]
        img, seg, labels, index = self.__getitem__(random_index)
        print("MINIBATCH: img.shapes seg.shape labels.shape, index.shape ->", img.shape, seg.shape,
              labels.shape, index.shape)
        print("MINIBATCH: min and max of minibatch", torch.min(img), torch.max(img))
        return plot_img_and_seg(img=img, seg=seg, figsize=(6, 12))


def sample_and_kl_diagonal_normal(posterior_mu: torch.Tensor,
                                  posterior_std: torch.Tensor,
                                  prior_mu: torch.Tensor,
                                  prior_std: torch.Tensor,
                                  noisy_sampling: bool,
                                  sample_from_prior: bool,
                                  squeeze_mc: bool,
                                  mc_samples: int = 1) -> DIST:
    """
    Analytically computes KL divergence between two gaussian distributions
    and draw a sample from either the prior or posterior depending of the values of :attr:`sample_from_prior`.

    Args:
        posterior_mu: torch.Tensor with the posterior mean
        posterior_std: torch.Tensor with the posterior standard deviation
        prior_mu: torch.Tensor with the prior mean
        prior_std: torch.Tensor with the prior standard deviation
        noisy_sampling: if True a random sample is generated,
            if False the mode of the distribution (i.e. the mean) is returned.
        sample_from_prior: if True the sample is drawn from the prior distribution, if False the posterior distribution
            is used.
        squeeze_mc: whether or not to squeeze the leading dimension corresponding to different mc_samples.
            This has effect only if :attr:`mc_samples` == 1.
        mc_samples: number of monte_carlo samples

    Returns:
        :class:`DIST` with the KL divergence and the sample from either the prior or posterior
        depending of the values of :attr:`sample_from_prior`. The shape of the sample and KL divergence is equal to
        the common broadcast shape of the :attr:`posterior_mu`, :attr:`posterior_std`, :attr:`prior_mu`
        and :attr:`prior_std`. If :attr:`mc_samples` > 1 or :attr:`squeeze_mc` is False a leading dimension is
        added to the left to represent the different monte carlo samples

    Note:
        :attr:`posterior_mu`, :attr:`posterior_std`, :attr:`prior_mu` and :attr:`prior_std`
        must the broadcastable to a common shape.

    Note:
        The KL divergence is :math:`KL = \\int dz q(z) \\log \\left(p(z)/q(z)\\right)` where
        :math:`q(z)` is the posterior and :math:`p(z)` is the prior.
    """
    post_mu, post_std, pr_mu, pr_std = broadcast_all(posterior_mu, posterior_std, prior_mu, prior_std)
    random = torch.randn([mc_samples] + list(post_mu.shape), device=post_mu.device, dtype=post_mu.dtype)

    # Compute the KL divergence
    tmp = (post_std + pr_std) * (post_std - pr_std) + (post_mu - pr_mu).pow(2)
    kl = (tmp / (2 * pr_std * pr_std) - post_std.log() + pr_std.log()).expand_as(random)

    # Sample
    if sample_from_prior:
        # working with the prior
        sample = pr_mu + pr_std * random if noisy_sampling else pr_mu.expand_as(random)
    else:
        # working with the posterior
        sample = post_mu + post_std * random if noisy_sampling else post_mu.expand_as(random)

    if squeeze_mc:
        # Note that squeeze has effect only if the squeezed dimension has shape 1
        sample = sample.squeeze(dim=0)
        kl = kl.squeeze(dim=0)

    return DIST(sample=sample, kl=kl)


def compute_entropy_bernoulli(logit: torch.Tensor):
    """ Compute the entropy of the bernoulli distribution, i.e. H= - [p * log(p) + (1-p) * log(1-p)] """
    p = torch.sigmoid(logit)
    one_m_p = torch.sigmoid(-logit)
    log_p = F.logsigmoid(logit)
    log_one_m_p = F.logsigmoid(-logit)
    entropy = - (p * log_p + one_m_p * log_one_m_p)
    return entropy


def compute_logp_bernoulli(c: torch.Tensor, logit: torch.Tensor):
    """
     Compute the log_probability of the :attr:`c` configuration under the Bernoulli distribution specified by
     :attr:`logit`.

     Args:
         c: Boolean tensor with the configuration
         logit: Logit of the Bernoulli distributions

     Returns:
         :math:`log_prob(c_grid | BERNOULLI(logit_grid))` of the same shape as :attr:`c`.
         This value is differentiable w.r.t. the :attr:`logit` but not differentiable w.r.t. :attr:`c`.
     """

    log_p = F.logsigmoid(logit)
    log_one_m_p = F.logsigmoid(-logit)
    log_prob_bernoulli = (c.detach() * log_p + ~c.detach() * log_one_m_p)
    return log_prob_bernoulli
