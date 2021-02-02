import json
import torch
import numpy
import dill
import skimage.measure
from typing import Union, Optional
from .namedtuple import BB, ConcordanceIntMask

""" This modules has many low-level utilities such as saving and loading files, flatten dictionary to list, 
    reshape of tensor and so on. """


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
    """ takes input of shape: (batch, ch, width, height)
        and returns output of shape: (n_list, batch, ch)
        where n_list = width x height
    """
    assert len(x.shape) == 4
    batch_size, ch, width, height = x.shape
    return x.permute(2, 3, 0, 1).view(width*height, batch_size, ch)


def invert_convert_to_box_list(x: torch.Tensor, original_width: int, original_height: int) -> torch.Tensor:
    """ takes input of shape: (width x height, batch, ch)
        and return shape: (batch, ch, width, height)
    """
    assert len(x.shape) == 3
    n_list, batch_size, ch = x.shape
    assert n_list == original_width * original_height
    return x.permute(1, 2, 0).view(batch_size, ch, original_width, original_height)


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


def load_json_as_dict(path) -> dict:
    """
    Read the content of a json file into a dictionary.

    Args:
        path: Path to a json file

    Returns:
        dictionary with the content of the json file
    """
    with open(path, 'rb') as f:
        return json.load(f)


def save_dict_as_json(input_dict, path):
    """
    Save a dictionary to a json file.

    Args:
        input_dict: Dictionary which will be saved to file
        path: Path to the json file which will be (over-) written.
    """
    with open(path, 'w') as f:
        return json.dump(input_dict, f)


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


def compute_ranking(x_nb: torch.Tensor) -> torch.Tensor:
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


def compute_average_in_box(delta_imgs_bcwh: torch.Tensor, bounding_box_nb: BB) -> torch.Tensor:
    """
    Compute the average of the input tensor in each bounding box.

    Args:
        delta_imgs_bcwh: The quantity to average of shape :math:`(B, C, W, H)`
        bounding_box_nb: The bounding box of :class:`BB` of shape :math:`(N, B)`

    Returns:
        A tensor of shape :math:`(N, B)` with the average value of :attr:`delta_imgs_bcwh` in each bounding box.
    """
    # cumulative sum in width and height, standard sum in channels
    cum_sum_bwh = torch.cumsum(torch.cumsum(delta_imgs_bcwh.sum(dim=-3), dim=-1), dim=-2)

    # compute the x1,y1,x3,y3
    x1_nb = (bounding_box_nb.bx - 0.5 * bounding_box_nb.bw).long().clamp(min=0, max=delta_imgs_bcwh.shape[-2])
    x3_nb = (bounding_box_nb.bx + 0.5 * bounding_box_nb.bw).long().clamp(min=0, max=delta_imgs_bcwh.shape[-2])
    y1_nb = (bounding_box_nb.by - 0.5 * bounding_box_nb.bh).long().clamp(min=0, max=delta_imgs_bcwh.shape[-1])
    y3_nb = (bounding_box_nb.by + 0.5 * bounding_box_nb.bh).long().clamp(min=0, max=delta_imgs_bcwh.shape[-1])

    # compute the area
    # Note that this way penalizes boxes that go out-of-bound
    # This is in contrast to area = (x3-x1)*(y3-y1) which does NOT penalize boxes out of bound
    area_nb = bounding_box_nb.bw * bounding_box_nb.bh

    # compute the total intensity in each box
    index_nb = torch.arange(start=0, end=x1_nb.shape[-1], step=1, device=x1_nb.device,
                            dtype=x1_nb.dtype).view(1, -1).expand(x1_nb.shape[-2], -1)

    x1_ge_1 = (x1_nb >= 1).float()
    x3_ge_1 = (x3_nb >= 1).float()
    y1_ge_1 = (y1_nb >= 1).float()
    y3_ge_1 = (y3_nb >= 1).float()
    tot_intensity_nb = cum_sum_bwh[index_nb, x3_nb-1, y3_nb-1] * x3_ge_1 * y3_ge_1 + \
                       cum_sum_bwh[index_nb, x1_nb-1, y1_nb-1] * x1_ge_1 * y1_ge_1 - \
                       cum_sum_bwh[index_nb, x1_nb-1, y3_nb-1] * x1_ge_1 * y3_ge_1 - \
                       cum_sum_bwh[index_nb, x3_nb-1, y1_nb-1] * x3_ge_1 * y1_ge_1
    return tot_intensity_nb / area_nb