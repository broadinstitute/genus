import yaml
import torch
import numpy
import dill
import skimage.measure
from typing import Union, Optional, List
from .namedtuple import BB, ConcordanceIntMask
import torch.nn.functional as F


""" This modules has many low-level utilities such as saving and loading files, flatten dictionary to list, 
    reshape of tensor and so on. """

def are_broadcastable(a: torch.Tensor, b: torch.Tensor) -> bool:
    """ Returns True if the two tensor are broadcastable to each other, False otherwise. """
    return all((m == n) or (m == 1) or (n == 1) for m, n in zip(a.shape[::-1], b.shape[::-1]))


def roller_2d(a: torch.tensor, b: Optional[torch.tensor] = None, radius: int = 2, shape: str = "circle"):
    """
    Performs rolling of the last two dimensions of the tensor.
    For each point consider half of the connection (the other half will be recorded by the other member of the pair).
    For example for a square shape of radius = 2 the full neighbourhood is 5x5 and for each pixel
    I only need to record 12 neighbouring pixels

    Args:
        a: First tensor to roll
        b: Second tensor to roll
        radius: size of the
        shape: Either "circle" or "square".

    Returns:
        An iterator container with the all metric of interest.

    Examples:
        >>> for mixing_shifted, index_shifted in roller_2d(a=mixing, b=pad_index, radius=radius_nn):
        >>>     v = (mixing * mixing_shifted).sum(dim=-5)
        >>>     row = index_shifted
        >>>     col = index
        >>>     index_tensor = torch.stack((row[mask], col[mask]), dim=0)
        >>>     similarity = torch.sparse.FloatTensor(index_tensor, v[mask])
    """

    assert len(a.shape) > 2
    assert len(b.shape) > 2 or b is None

    dxdy_list = []
    for dx in range(0, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy <= 0:
                continue
            if shape == "square":
                dxdy_list.append((dx, dy))
            elif shape == "circle":
                if dx*dx + dy*dy <= radius*radius:
                    dxdy_list.append((dx, dy))
            else:
                raise Exception("Invalid shape argument. It can only be 'square' or 'circle'")

    for dxdy in dxdy_list:
        a_tmp = torch.roll(torch.roll(a, dxdy[0], dims=-2), dxdy[1], dims=-1)
        b_tmp = None if b is None else torch.roll(torch.roll(b, dxdy[0], dims=-2), dxdy[1], dims=-1)
        yield a_tmp, b_tmp


def convert_to_box_list(x: torch.Tensor) -> torch.Tensor:
    """ takes input of shape: (*, ch, width, height)
        and returns output of shape: (*, n_list, ch)
        where n_list = width x height
    """
    return x.flatten(start_dim=-2).transpose(dim0=-1, dim1=-2)


def invert_convert_to_box_list(x: torch.Tensor, original_width: int, original_height: int) -> torch.Tensor:
    """ takes input of shape: (*, width x height, ch)
        and return shape: (*, ch, width, height)
    """
    assert x.shape[-2] == original_width * original_height
    return x.transpose(dim0=-1, dim1=-2).view(list(x.shape[:-2]) + [x.shape[-1], original_width, original_height])


def linear_interpolation(t: Union[numpy.array, float], values: tuple, times: tuple) -> Union[numpy.array, float]:
    """ Makes an interpolation between (t_in,v_in) and (t_fin,v_fin)
        For time t>t_fin and t<t_in the value of v is clamped to either v_in or v_fin
        Usage:
        epoch = numpy.arange(0,100,1)
        v = linear_interpolation(epoch, values=[0.0,0.5], times=[20,40])
        plt.plot(epoch,v)
    """
    v_in, v_fin = values  # initial and final values
    t_in, t_fin = times   # initial and final times

    if t_fin >= t_in:
        den = max(t_fin-t_in, 1E-8)
        m = (v_fin-v_in)/den
        v = v_in + m*(t-t_in)
    else:
        raise Exception("t_fin should be greater than t_in")

    v_min = min(v_in, v_fin)
    v_max = max(v_in, v_fin)
    return numpy.clip(v, v_min, v_max)


def QC_on_integer_mask(integer_mask: Union[torch.Tensor, numpy.ndarray], min_area):
    """ This function filter the labels by some criteria (for example by min size).
        Add more QC in the future"""
    if isinstance(integer_mask, torch.Tensor):
        labels = skimage.measure.label(integer_mask.cpu(), background=0, return_num=False, connectivity=2)
    else:
        labels = skimage.measure.label(integer_mask, background=0, return_num=False, connectivity=2)

    mydict = skimage.measure.regionprops_table(labels, properties=['label', 'area'])
    my_filter = mydict["area"] > min_area

    bad_labels = mydict["label"][~my_filter]
    old2new = numpy.arange(mydict["label"][-1] + 1)
    old2new[bad_labels] = 0
    new_integer_mask = old2new[labels]

    if isinstance(integer_mask, torch.Tensor):
        return torch.from_numpy(new_integer_mask).to(dtype=integer_mask.dtype, device=integer_mask.device)
    else:
        return new_integer_mask.astype(integer_mask.dtype)


def remove_label_gaps(label: torch.Tensor):
    assert torch.is_tensor(label)
    assert len(label.shape) == 2
    count = torch.bincount(label.flatten())
    exist = (count > 0).to(label.dtype)
    old2new = torch.cumsum(exist, dim=-1) - exist[0]
    print(old2new)
    return old2new[label.long()]
    

def concordance_integer_masks(mask1: torch.Tensor, mask2: torch.Tensor) -> ConcordanceIntMask:
    """ Compute measure of concordance between two partitions:
        joint_distribution
        mutual_information
        delta_n
        iou

        We use the peaks of the join distribution to extract the mapping between membership labels.
    """
    assert torch.is_tensor(mask1)
    assert torch.is_tensor(mask2)
    assert mask1.shape == mask2.shape
    assert mask1.device == mask2.device

    nx = torch.max(mask1).item()  # 0,1,..,nx
    ny = torch.max(mask2).item()  # 0,1,..,ny
    z = mask2 + mask1 * (ny + 1)  # 0,1,...., nx*ny
    count_z = torch.bincount(z.flatten(),
                             minlength=(nx+1)*(ny+1)).float().view(nx+1, ny+1).to(dtype=torch.float,
                                                                                  device=mask1.device)
    # Now remove the background
    pxy = count_z.clone()
    pxy[0, :] = 0
    pxy[:, 0] = 0

    # computing the mutual information
    pxy /= torch.sum(pxy)
    px = torch.sum(pxy, dim=-1)
    py = torch.sum(pxy, dim=-2)
    term_xy = pxy * torch.log(pxy)
    term_x = px * torch.log(px)
    term_y = py * torch.log(py)
    mutual_information = term_xy[pxy > 0].sum() - term_x[px > 0].sum() - term_y[py > 0].sum()

    # Extract the most likely mappings
    mask1_to_mask2 = torch.max(pxy, dim=-1)[1]
    mask2_to_mask1 = torch.max(pxy, dim=-2)[1]
    assert mask1_to_mask2.shape[0] == nx + 1
    assert mask2_to_mask1.shape[0] == ny + 1

    # Find one-to-one correspondence among IDs
    id_mask1 = torch.arange(nx + 1, device=mask1.device, dtype=torch.long)  # [0,1,.., nx]
    is_id_reversible = (mask2_to_mask1[mask1_to_mask2[id_mask1]] == id_mask1)
    n_reversible_instances = is_id_reversible[1:].sum().item()  # exclude id=0 bc background

    # Find one-to-one correspondence among pixels
    intersection_mask = (mask1_to_mask2[mask1] == mask2) * (mask2_to_mask1[mask2] == mask1) * (mask1 > 0) * (mask2 > 0)
    intersection = intersection_mask.int().sum()
    union = (mask1 > 0).int().sum() + (mask2 > 0).int().sum() - intersection  # exclude background
    iou = intersection.float() / union

    return ConcordanceIntMask(intersection_mask=intersection_mask,
                              joint_distribution=pxy,
                              mutual_information=mutual_information.item(),
                              delta_n=ny - nx,
                              iou=iou.item(),
                              n_reversible_instances=n_reversible_instances)


def flatten_list(ll):
    if not ll:  # equivalent to if ll == []
        return ll
    elif isinstance(ll[0], list):
        return flatten_list(ll[0]) + flatten_list(ll[1:])
    else:
        return ll[:1] + flatten_list(ll[1:])


def flatten_dict(dd, separator='_', prefix=''):
    return {prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
            } if isinstance(dd, dict) else {prefix: dd}


def save_obj(obj, path):
    with open(path, 'wb') as f:
        torch.save(obj, f,
                   pickle_module=dill,
                   pickle_protocol=2,
                   _use_new_zipfile_serialization=True)


def load_obj(path):
    with open(path, 'rb') as f:
        return torch.load(f, pickle_module=dill)


def load_yaml_as_dict(path) -> dict:
    """
    Read the content of a YAML file into a dictionary.

    Args:
        path: Path to a YAML file

    Returns:
        dictionary with the content of the YAML file
    """
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_dict_as_yaml(input_dict, path):
    """
    Save a dictionary to a YAML file.

    Args:
        input_dict: Dictionary which will be saved to file
        path: Path to the YAML file which will be (over-) written.
    """
    with open(path, 'w') as f:
        return yaml.dump(input_dict, f)


def ckpt2file(ckpt: dict, path: str):
    """
    Saves the ckpt dictionary to a file. It is a thin wrapper around torch.save.

    Args:
        ckpt: Dictionary containing the information create by invoking :class:`create_ckpt` or
            :class:`file2ckpt` (required).
        path: The path to the file to be written (required).

    Note:
        Used in combination with :class:`create_ckpt` to save the state of the current simulation.

    Examples:
        >>> ckpt = vae.create_ckpt()
        >>> ckpt2file(ckpt=ckpt, path="./latest_ckpt.pt")
    """
    torch.save(ckpt, path)
    return


def file2ckpt(path: str, device: Optional[str] = None) -> dict:
    """
    Load the ckpt file into a dictionary to restart a past simulation. It is a thin wrapper around torch.load.

    Args:
        path: A string specifying the location of the ckpt file (required)
        device: A string either "cuda" or "cpu" specifying the device of the intended (optional).
            This is usefull when a model trained on a GPU machine need to be loaded into a CPU machine or viceversa.

    Examples:
        >>> # to load a ckpt of a model which was trained on GPU into a CPU machine.
        >>> ckpt = file2ckpt(path="pretrained_GPU_model.pt", device="cpu")
        >>> vae = CompositionalVae(params=ckpt.get("params"))
    """
    if device is None:
        ckpt = torch.load(path)
    elif device == 'cuda':
        ckpt = torch.load(path, map_location="cuda:0")
    elif device == 'cpu':
        ckpt = torch.load(path, map_location=torch.device('cpu'))
    else:
        raise Exception("device is not recognized")
    return ckpt


def append_to_dict(source: Union[tuple, dict],
                   destination: dict,
                   prefix_include: str = None,
                   prefix_exclude: str = None,
                   prefix_to_add: str = None):
    """
    Append the content of a tuple or dictionary to another dictionary.

    Args:
        source: tuple or dictionary whose value will be appended
        destination: dictionary where the values will be appended
        prefix_include: If specified only the keys in :attr:`source` which start with this prefix will be appended
        prefix_exclude: If specified the keys in :attr:`source` which start with this prefix will NOT be appended
        prefix_to_add: If specified this prefix will be added to the key before copying them in the destination
            dictionary.

    Note:
        If both :attr:`prefix_include` and :attr:`prefix_exclude` are None then the default behavior is to
        append to dictionary.
    """

    def _get_x_y(_key, _value):
        if (prefix_include is None or _key.startswith(prefix_include)) and (prefix_exclude is None or
                                                                            not _key.startswith(prefix_exclude)):
            x_tmp = _key if prefix_to_add is None else prefix_to_add + _key
            try:
                y_tmp = _value.item()
            except (AttributeError, ValueError):
                y_tmp = _value
            return x_tmp, y_tmp
        else:
            return None, None

    if isinstance(source, dict):
        for key, value in source.items():
            x, y = _get_x_y(key, value)
            if x is not None:
                destination[x] = destination.get(x, []) + [y]
    elif isinstance(source, tuple):
        for key in source._fields:
            value = getattr(source, key)
            x, y = _get_x_y(key, value)
            if x is not None:
                destination[x] = destination.get(x, []) + [y]
    else:
        print(type(source))
        raise Exception

    return destination


@torch.no_grad()
def compute_ranking(x: torch.Tensor) -> torch.Tensor:
    """ Given a vector of shape: (*, n) compute the ranking along the last dimension.

        Note:
            It works for arbitrary leading dimension. Each leading dimension will be treated independently.
    """
    indices = torch.sort(x, dim=-1, descending=False)[1]

    # this is the fast way which uses indexing on the left
    src = torch.arange(start=0, end=x.shape[-1], step=1, dtype=indices.dtype, device=indices.device).expand_as(indices)
    rank = torch.zeros_like(indices)
    rank.scatter_(dim=-1, index=indices, src=src)
    return rank


@torch.no_grad()
def select_topK_2D(values_b1wh: torch.Tensor, k: int) -> torch.Tensor:
    """
    Given a vector of shape (B,1,W,H) it returns a boolean mask with the location of K
    maxima across the last two spatial dimensions """

    # Now select the top k maxima across the last two spatial dimensions
    assert values_b1wh.shape[-3] == 1
    assert len(values_b1wh.shape) == 4

    values_nb = convert_to_box_list(values_b1wh).squeeze(-1)
    index_kb = torch.topk(values_nb, k=k, dim=-2, largest=True, sorted=True)[1]
    k_local_maxima_mask_nb = torch.zeros_like(values_nb).scatter(dim=-2,
                                                                 index=index_kb,
                                                                 src=torch.ones_like(values_nb))
    k_local_maxima_mask_b1wh = invert_convert_to_box_list(k_local_maxima_mask_nb.unsqueeze(-1),
                                                          original_width=values_b1wh.shape[-2],
                                                          original_height=values_b1wh.shape[-1])
    return k_local_maxima_mask_b1wh


@torch.no_grad()
def get_local_maxima_mask(values_b1wh: torch.Tensor) -> torch.Tensor:
    """ Given a vector of shape (B,1,W,H) it returns a boolean mask with the location of the local-maxima """
    assert values_b1wh.shape[-3] == 1
    assert len(values_b1wh.shape) == 4
    values_b1wh_pooled = F.max_pool2d(values_b1wh, kernel_size=3, stride=1,
                                      padding=1, return_indices=False)
    mask_b1wh = (values_b1wh_pooled == values_b1wh)
    return mask_b1wh


@torch.no_grad()
def compute_tensor_ranking(x_nb: torch.Tensor) -> torch.Tensor:
    """ Given a vector of shape: n, batch_size
        For each batch dimension it ranks the n elements. """
    assert len(x_nb.shape) == 2
    n, batch_size = x_nb.shape
    _, order = torch.sort(x_nb, dim=-2, descending=False)

    # this is the fast way which uses indexing on the left
    rank_nb = torch.zeros_like(order)
    batch_index = torch.arange(batch_size, dtype=order.dtype, device=order.device).view(1, -1).expand(n, batch_size)
    rank_nb[order, batch_index] = torch.arange(n,
                                               dtype=order.dtype,
                                               device=order.device).view(-1, 1).expand(n, batch_size)
    return rank_nb


def compute_average_in_box(delta_imgs: torch.Tensor, bounding_box: BB) -> torch.Tensor:
    """
    Compute the average of the input tensor in each bounding box.

    Args:
        delta_imgs: The quantity to average of shape :math:`(*, C, W, H)`
        bounding_box: The bounding box of :class:`BB` of shape :math:`(*, K)`

    Returns:
        A tensor of shape :math:`(*, K)` with the average value of :attr:`delta_imgs` in each bounding box.

    Note:
        This works for any number of leading dimensions. Each leading dimension will be analyzed independently.
    """
    assert delta_imgs.shape[:-3] == bounding_box.bx.shape[:-1]

    # cumulative sum in width and height, standard sum in channels of shape (-1, w, h)
    cum_sum_wh = torch.cumsum(torch.cumsum(delta_imgs.sum(dim=-3), dim=-1), dim=-2).flatten(end_dim=-3)

    # compute the x1,y1,x3,y3 of shape (-1, K)
    x1 = (bounding_box.bx - 0.5 * bounding_box.bw).long().clamp(min=0, max=delta_imgs.shape[-2]).flatten(end_dim=-2)
    x3 = (bounding_box.bx + 0.5 * bounding_box.bw).long().clamp(min=0, max=delta_imgs.shape[-2]).flatten(end_dim=-2)
    y1 = (bounding_box.by - 0.5 * bounding_box.bh).long().clamp(min=0, max=delta_imgs.shape[-1]).flatten(end_dim=-2)
    y3 = (bounding_box.by + 0.5 * bounding_box.bh).long().clamp(min=0, max=delta_imgs.shape[-1]).flatten(end_dim=-2)

    # compute the area
    # There are 2 ways of computing the area:
    # AREA_1 = (x3-x1)*(y3-y1)
    # AREA_2 = bounding_box_nb.bw * bounding_box_nb.bh
    # The difference is that for out-of-bound boxes AREA_1 < AREA_2
    area = ((x3 - x1) * (y3 - y1)).float()

    # compute the average intensity
    index_independent_dim = torch.arange(start=0, end=x1.shape[0], step=1,
                                         device=x1.device, dtype=x1.dtype).view(-1, 1).expand_as(x1)

    x1_ge_1 = (x1 >= 1).float()
    x3_ge_1 = (x3 >= 1).float()
    y1_ge_1 = (y1 >= 1).float()
    y3_ge_1 = (y3 >= 1).float()
    tot_intensity = cum_sum_wh[index_independent_dim, x3-1, y3-1] * x3_ge_1 * y3_ge_1 + \
                    cum_sum_wh[index_independent_dim, x1-1, y1-1] * x1_ge_1 * y1_ge_1 - \
                    cum_sum_wh[index_independent_dim, x1-1, y3-1] * x1_ge_1 * y3_ge_1 - \
                    cum_sum_wh[index_independent_dim, x3-1, y1-1] * x3_ge_1 * y1_ge_1

    av_intensity = (tot_intensity.float() / area).view_as(bounding_box.bx)
    return av_intensity
