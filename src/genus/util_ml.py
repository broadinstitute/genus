import torch
import numpy
import torch.nn.functional as F
from torch.distributions.utils import broadcast_all
from typing import Union, Optional, Tuple
from collections import OrderedDict
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from collections import deque
from .util import invert_convert_to_box_list, convert_to_box_list
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

    def _accumulate_key_value(self, key, value, counter_increment):
        if isinstance(value, torch.Tensor):
            x = value.detach().item() * counter_increment
        elif isinstance(value, float):
            x = value * counter_increment
        elif isinstance(value, numpy.ndarray):
            x = value * counter_increment
        else:
            print(type(value))
            raise Exception

        try:
            self._dict_accumulate[key] = x + self._dict_accumulate.get(key, 0)
        except ValueError:
            # oftent the case if accumulating two numpy array of different sizes
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
            tmp[k] = v/self._counter
        return tmp

    def set_value(self, key, value):
        """ Set a key value pair in the internal OrderDictionary to do the accumulation """
        self._dict_accumulate[key] = value


def are_broadcastable(a: torch.Tensor, b: torch.Tensor) -> bool:
    """ Returns True if the two tensor are broadcastable to each other, False otherwise. """
    return all((m == n) or (m == 1) or (n == 1) for m, n in zip(a.shape[::-1], b.shape[::-1]))


def sample_and_kl_diagonal_normal(posterior_mu: torch.Tensor,
                                  posterior_std: torch.Tensor,
                                  prior_mu: torch.Tensor,
                                  prior_std: torch.Tensor,
                                  noisy_sampling: bool,
                                  sample_from_prior: bool) -> DIST:
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

    Returns:
        :class:`DIST` with the KL divergence and the sample from either the prior or posterior
        depending of the values of :attr:`sample_from_prior`. The shape of the sample and KL divergence is equal to
        the common broadcast shape of the :attr:`posterior_mu`, :attr:`posterior_std`, :attr:`prior_mu`
        and :attr:`prior_std`

    Note:
        :attr:`posterior_mu`, :attr:`posterior_std`, :attr:`prior_mu` and :attr:`prior_std`
        must the broadcastable to a common shape.

    Note:
        The KL divergence is :math:`KL = \\int dz q(z) \\log \\left(p(z)/q(z)\\right)` where
        :math:`q(z)` is the posterior and :math:`p(z)` is the prior.
    """
    # Compute the KL divergence
    post_mu, post_std, pr_mu, pr_std = broadcast_all(posterior_mu, posterior_std, prior_mu, prior_std)
    tmp = (post_std + pr_std) * (post_std - pr_std) + (post_mu - pr_mu).pow(2)
    kl = tmp / (2 * pr_std * pr_std) - post_std.log() + pr_std.log()

    if sample_from_prior:
        # working with the prior
        sample = pr_mu + pr_std * torch.randn_like(pr_mu) if noisy_sampling else pr_mu
    else:
        # working with the posterior
        sample = post_mu + post_std * torch.randn_like(post_mu) if noisy_sampling else post_mu

    return DIST(sample=sample, kl=kl)


def sample_c_grid(logit_grid: torch.Tensor,
                  similarity_matrix: torch.Tensor,
                  noisy_sampling: bool,
                  sample_from_prior: bool) -> torch.Tensor:
    """
    Sample c_grid either from a Determinental Point Process (DPP) prior specified by a similarity matrix or
    from a posterior which is a collection of independent Bernoulli specified by the logit_grid.

    Args:
        logit_grid: torch.Tensor of size :math:`(B,1,W,H)` with the logit specifying the Bernoulli probabilities
        similarity_matrix: torch.Tensor of size :math:`(W x H, W x H)` with the similarity between all grid points
        noisy_sampling: if True draw a random sample, if False the sample is the mode of the distribution
        sample_from_prior: If True draw from the prior (DPP), if False draw from the posterior (independent Bernoulli)

    Returns:
        Binarized torch.Tensor with c_grid which has the same size as :attr:`logit_grid`.

    Note:
        If :attr:`sample_from_prior` is True the returned value (c_grid) is NOT differentiable w.r.t.
        the :attr:`similarity_matrix`. If :attr:`sample_from_prior` is False the returned value (c_grid)
        IS differentiable w.r.t. :attr:`logit_grid`. We use a pass-thought approximation,
        i.e. :math:`\delta c/\delta logit = \delta p/\delta logit * \delta c/\delta p = \delta p/\delta logit * 1'
    """
    assert len(logit_grid.shape) == 4
    assert logit_grid.shape[-3] == 1
    assert len(similarity_matrix.shape) == 2
    assert similarity_matrix.shape[-2] == similarity_matrix.shape[-1] == logit_grid.shape[-1]*logit_grid.shape[-2]

    if sample_from_prior:
        # sample from DPP specified by the similarity kernel
        with torch.no_grad():
            batch_size = torch.Size([logit_grid.shape[0]])
            s = similarity_matrix.requires_grad_(False)
            c_all = FiniteDPP(L=s).sample(sample_shape=batch_size)  # shape: batch_size, n_points
            c_reshaped = c_all.transpose(-1, -2).float().unsqueeze(-1)  # shape: n_points, batch_size, 1
            c_grid = invert_convert_to_box_list(c_reshaped,
                                                original_width=logit_grid.shape[-2],
                                                original_height=logit_grid.shape[-1])
            return c_grid  # shape: batch_size, 1, p_map.shape[-2], p_map.shape[-1]
    else:
        # sample from posterior which is a collection of independent Bernoulli variables
        p_grid = torch.sigmoid(logit_grid)
        with torch.no_grad():
            c_grid = (torch.rand_like(p_grid) < p_grid) if noisy_sampling else (0.5 < p_grid)

        # trick so that value is given by c_grid but derivatives from p_grid
        return c_grid.float() + p_grid - p_grid.detach()


def compute_logp_dpp(c_grid: torch.Tensor,
                     similarity_matrix: torch.Tensor):
    """
    Compute the log_probability of the :attr:`c_grid` configuration under the DPP distribution specified by the
    :attr:`similarity_matrix`.

    Args:
        c_grid: Binarized configuration of shape :math:`(B,1,W,H)`
        similarity_matrix: Matrix with the similarity matrix between grid points of shape :math:`(W x H, W x H)`

    Returns:
        :math:`log_prob(c_grid | DPP(similarity_matrix))` of shape :math:`(B)`. This value is differentiable w.r.t.
        the :attr:`similarity_matrix` but not differentiable w.r.t. :attr:`c_grid`.
    """
    c_no_grad = convert_to_box_list(c_grid).squeeze(-1).bool().detach()  # shape n_points, batch_size
    log_prob_prior = FiniteDPP(L=similarity_matrix).log_prob(c_no_grad.transpose(-1, -2))  # shape: batch_shape
    return log_prob_prior


def compute_logp_bernoulli(c_grid: torch.Tensor,
                           logit_grid: torch.Tensor):
    """
    Compute the log_probability of the :attr:`c_grid` configuration under the collection
    of independent Bernoulli distributions specified by the :attr:`logit_grid`.

    Args:
        c_grid: Binarized configuration of shape :math:`(B,1,W,H)`
        logit_grid: Logit of the Bernoulli distributions of shape :math:`(B,1,W,H)`

    Returns:
        :math:`log_prob(c_grid | BERNOULLI(logit_grid))` of shape :math:`(B)`. This value is differentiable w.r.t.
        the :attr:`logit_grid` but not differentiable w.r.t. :attr:`c_grid`.
    """
    log_p_grid = F.logsigmoid(logit_grid)
    log_1_m_p_grid = F.logsigmoid(-logit_grid)
    log_prob_bernoulli = (c_grid.detach() * log_p_grid +
                          (c_grid - 1).detach() * log_1_m_p_grid).sum(dim=(-1, -2, -3))  # sum over ch=1, w, h
    return log_prob_bernoulli


class SimilarityKernel(torch.nn.Module):
    """
    Square gaussian kernel with learnable parameters (weight and lenght_scale).
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
            length_scale: Initial value for learnable parameter specifying the lenght scale of the kernel.
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

        # Initialization
        self.n_width = -1
        self.n_height = -1
        self.d2 = None
        self.diag = None

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

    def sample_2_mask(self, sample):
        """
        Conversion routine

        Args:
            sample: A tensor fo size :math:`(*, W \\times H)`

        Returns:
            A grid of size :math:`(*,W,H)`
        """
        independent_dims = list(sample.shape[:-1])
        mask = sample.view(independent_dims + [self.n_width, self.n_height])
        return mask

    def get_l_w(self):
        """ Returns the current value of lenght scale and weight of the similarity kernel """
        return self.length_scale_value.data, self.weight_value.data

    def _clamp_and_get_l_w(self):
        """ Clamps the parameters before they are used to compute the similarity matrix. """
        l_clamped = self.length_scale_value.data.clamp_(min=self.length_scale_min, max=self.length_scale_max)
        w_clamped = self.weight_value.data.clamp_(min=self.weight_min, max=self.weight_max)
        return l_clamped, w_clamped

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

        if (n_width != self.n_width) or (n_height != self.n_height):
            self.n_width = n_width
            self.n_height = n_height
            self.d2, self.diag = self._compute_d2_diag(n_width=n_width, n_height=n_height)

        return self.diag + w * torch.exp(-0.5*self.d2/(l*l))


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
                       'L': constraints.positive_definite}
    support = constraints.boolean
    has_rsample = False

    def __init__(self, K=None, L=None, validate_args=None):
        """
        A Finite DPP distribution is defined either via the likelihood matrix (L) or the correlation matrix (K).

        Args:
            K: correlation matrix of shape :math:`(*, n, n)` is positive-semidefinite, symmetric with
                eigenvalues in :math:`[0,1]`. It might be difficult to ensure that the eigenvalues are in :math:`[0,1]`,
                therefore it is recommended to define the DPP via the likelihood matrix :attr:`L`.
            L: likelihood matrix of shape :math:`(*, n, n)` is positive-semidefinite, symmetric with
                eigenvalues in :math:`>= 0`. Any expoentially decaying similarity kernel will give rise
                to a valid likelihood matrix.

        Note:
            This class is often used in combination with :class:`SimilarityKernel`.

        Examples:
            >>> kernel = SimilarityKernel(length_scale=10.0, weight=1.0)
            >>> likelihood_matrix = kernel.forward(n_width=20, n_height=20)
            >>> DPP = FiniteDPP(L=likelihood_matrix)
        """
        if (K is None and L is None) or (K is not None and L is not None):
            raise Exception("only one among K and L need to be defined")

        elif K is not None:
            self.K = 0.5 * (K + K.transpose(-1, -2))  # make sure it is symmetrized
            try:
                u, s_k, v = torch.svd(self.K)
            except:
                # torch.svd may have convergence issues for GPU and CPU.
                u, s_k, v = torch.svd(self.K + 1e-3 * self.K.mean() * torch.ones_like(self.K))
            s_l = s_k / (1.0 - s_k)
            self.L = torch.matmul(u * s_l.unsqueeze(-2), v.transpose(-1, -2))
        elif L is not None:
            self.L = 0.5 * (L + L.transpose(-1, -2))  # make sure it is symmetrized
            try:
                u, s_l, v = torch.svd(self.L)
            except:
                # torch.svd may have convergence issues for GPU and CPU.
                u, s_l, v = torch.svd(self.L + 1e-3 * self.L.mean() * torch.ones_like(self.L))
            s_k = s_l / (1.0 + s_l)
            self.K = torch.matmul(u * s_k.unsqueeze(-2), v.transpose(-1, -2))
        else:
            raise Exception

        self.s_l = s_l
        batch_shape, event_shape = self.K.shape[:-2], self.K.shape[-1:]
        super(FiniteDPP, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        """
        :meta private:
        """
        new = self._get_checked_instance(FiniteDPP, _instance)
        batch_shape = torch.Size(batch_shape)
        kernel_shape = batch_shape + self.event_shape + self.event_shape
        value_shape = batch_shape + self.event_shape
        new.s_l = self.s_l.expand(value_shape)
        new.L = self.L.expand(kernel_shape)
        new.K = self.K.expand(kernel_shape)
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
        likelihood matrix (L) used to initialize the instance.

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

        logdet_L_plus_I = (self.s_l + 1).log().sum(dim=-1)  # sum over the event_shape

        # Reshapes
        value = value.unsqueeze(0)  # trick to make it work even if not-batched
        independet_dims = list(value.shape[:-1])
        value = value.flatten(start_dim=0, end_dim=-2)  # *, event_shape
        # print("internal. value.shape", value.shape)
        L = self.L.expand(independet_dims + [-1, -1]).flatten(start_dim=0, end_dim=-3)  # *, event_shape, event_shape
        # print("internal. L.shape", L.shape)

        # Here I am computing the logdet of square matrix of different shapes
        # I use the trick to embed everything inside a larger identity matrix since
        #     | A  B  0 |
        # det | C  D  0 | = determinant of the sub_matrix
        #     | 0  0  1 |
        n_max = torch.sum(value, dim=-1).max().item()
        matrix = torch.eye(n_max, dtype=L.dtype, device=L.device).expand(L.shape[-3], n_max, n_max).clone()
        # print("internal. matrix.shape", matrix.shape)
        for i in range(value.shape[0]):
            n = torch.sum(value[i]).item()
            matrix[i, :n, :n] = L[i, value[i], :][:, value[i]]
        logdet_Ls = torch.logdet(matrix).view(independet_dims)  # sample_shape, batch_shape
        return (logdet_Ls - logdet_L_plus_I).squeeze(0)  # trick to make it work even if not-batched


class ConditionalRandomCrop(object):
    """
    Crop a torch Tensor at random locations to obtain output of given size.
    The random crop is accepted only if it contains the Region Of Interest (ROI).

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
        :meta private:
        Makes a batch of cropped images.

        Args:
            img: tensor of shape :math:`(B,C,W,H)` with the images to crop
            bij_list: list of tuple of type [b,i,j] indicating the image index and the location of the lower-left corner
                of the random crop

        Returns:
            A batch with the cropped images of shape :math:`(N,C,\\text{desired}_w,\\text{desired}_h)`
            where N is the length of the list :attr:`bij_list`.
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

