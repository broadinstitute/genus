import torch
from typing import Optional, Union, Tuple
from collections import deque
from .util_vis import plot_img_and_seg
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset


class ImageFolderWithIndex(ImageFolder):
    """ Wrapper around :class:`ImageFolder` which returns the index of the input.
        This is helpful to keep track of the examples that you get wrong
    """
    def __getitem__(self, index: int):
        x, y = super().__getitem__(index)
        return x, y, index


class DataloaderWithLoad(DataLoader):
    """ Wrapper around :class:`Dataloader` with an extra load function which loads specific examples.
        This is helpful to check the examples you got wrong.
    """

    def load(self, n_example: Optional[int] = None, index: Optional[torch.Tensor] = None):
        """
        Load random or specific examples from the dataloader

        Args:
            n_example: number of inputs to randomly load
            index: torch.Tensor of type long specifying the index of the images to load. Each entry should be between
                0 and self.__len__(). Repeated index are allowed.

        Returns:
            A batch of inputs of size :attr:`batch_size` or len(:attr:`index`)
        """
        if (n_example is None and index is None) or (n_example is not None and index is not None):
            raise Exception("Only one between batch_size and index must be specified")
        random_index = torch.randperm(self.dataset.__len__(), dtype=torch.long)[:n_example]
        index = random_index if index is None else index
        return self.collate_fn([self.dataset[i] for i in index])


class ConditionalRandomCrop(torch.nn.Module):
    """
    Crop a single torch Tensor at random locations to obtain output of given size.
    The random crop is accepted only if it contains a certain fraction of the Region Of Interest (ROI).
    """

    def __init__(self, size: Union[int, Tuple[int, int]],
                 min_roi_fraction: float = 0.1,
                 n_crops_per_image: int = 1):
        """
        Args:
            size (sequence or int): Desired output size of the crop. If size is an int instead of sequence like (h, w),
                a square crop (size, size) is made. If provided a tuple or list of length 1, it will be interpreted as
                (size[0], size[0]).
            min_roi_fraction: minimum threshold of the Region of Interest (ROI).
                Random crops with less that this amount of ROI will be disregarded.
            n_crops_per_image: number of random crops to generate for each image
        """
        super().__init__()
        if isinstance(size, int):
            self.desired_w = self.desired_h = size
        elif len(size) == 1:
            self.desired_w = self.desired_h = size[0]
        elif len(size) == 2:
            self.desired_w = size[0]
            self.desired_h = size[1]
        else:
            raise Exception("size is invalid type or shape", type(size))

        self.min_roi_fraction = min_roi_fraction
        self.desired_area = self.desired_w * self.desired_h
        self.n_crops_per_image = n_crops_per_image

    @staticmethod
    @torch.no_grad()
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

    @torch.no_grad()
    def _compute_roi_fraction(self,
                              roi_mask: Optional[torch.Tensor],
                              roi_mask_cumulative: Optional[torch.Tensor],
                              b: int, i: int, j: int):
        if roi_mask is None and roi_mask_cumulative is None:
            fraction = 1.0
        elif roi_mask is not None:
            fraction = roi_mask[b, 0, i:i + self.desired_w,
                       j:j + self.desired_h].sum().float() / self.desired_area
        elif roi_mask_cumulative is not None:
            term1 = roi_mask_cumulative[b, 0, i + self.desired_w - 1, j + self.desired_h - 1].item()
            term2 = 0 if i < 1 else roi_mask_cumulative[b, 0, i - 1, j + self.desired_h - 1].item()
            term3 = 0 if j < 1 else roi_mask_cumulative[b, 0, i + self.desired_w - 1, j - 1].item()
            term4 = 0 if (i < 1 or j < 1) else roi_mask_cumulative[b, 0, i - 1, j - 1].item()
            fraction = float(term1 - term2 - term3 + term4) / self.desired_area
        return fraction

    @torch.no_grad()
    def get_index(self,
                  roi_mask: Optional[torch.Tensor] = None,
                  roi_mask_cumulative: Optional[torch.Tensor] = None,
                  n_crops_per_image: Optional[int] = None):
        """
        :meta private:
        Create a list of n_crops_per_image tuples indicating the location of the lower-left corner of the random crops.

        Args:
            roi_mask: boolean tensor of shape :math:`(*,1,W,H)` indicating the region of interest (ROI)
            roi_mask_cumulative: tensor of shape :math:`(*,1,W,H)` with the double integral (both along row and column)
                of the ROI
            n_crops_per_image: how many random crops to generate for each image in the input batch

        Returns:
            A list of n_crops_per_image tuples indicating the location of the lower-left corner of the random crops

        Raises:
            Exception if both :attr:`roi_mask` and :attr:`roi_mask_cumulative` are None
        """
        n_crops_per_image = self.n_crops_per_image if n_crops_per_image is None else n_crops_per_image

        if (roi_mask is None and roi_mask_cumulative is None) or \
                (roi_mask is not None and roi_mask_cumulative is not None):
            raise Exception("Only one between roi_mask and cum_sum_roi_mask can be specified")

        if roi_mask_cumulative is not None:
            if len(roi_mask_cumulative.shape) == 3:
                roi_mask_cumulative = roi_mask_cumulative.unsqueeze(0)
            batch_shape, ch, w_raw, h_raw = roi_mask_cumulative.shape
        elif roi_mask is not None:
            if len(roi_mask.shape) == 3:
                roi_mask = roi_mask.unsqueeze(0)
            batch_shape, ch, w_raw, h_raw = roi_mask.shape
        else:
            raise Exception()

        bij_list = deque()
        for b in range(batch_shape):
            for n in range(n_crops_per_image):
                fraction = 0
                first_iter = True
                while (fraction < self.min_roi_fraction) or first_iter:
                    first_iter = False  # trick to guarantee at least one iteration
                    i, j = self._get_smallest_corner_for_crop(w_raw=w_raw,
                                                              h_raw=h_raw,
                                                              w_desired=self.desired_w,
                                                              h_desired=self.desired_h)
                    fraction = self._compute_roi_fraction(roi_mask, roi_mask_cumulative, b, i, j)
                bij_list.append([b, i, j])
        return bij_list

    @torch.no_grad()
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

    @torch.no_grad()
    def forward(self,
                img: torch.Tensor,
                roi_mask: Optional[torch.Tensor] = None,
                roi_mask_cumulative: Optional[torch.Tensor] = None,
                n_crops_per_image: Optional[int] = None):
        """
        Crop a image or batch of images to generate :attr:`n_crops_per_imag` random crops for each image.

        Args:
            img: image or batch of images of shape :math:`(*,c,W,H)`
            roi_mask: boolean tensor of shape :math:`(*,1,W,H)` indicating the region of interest (ROI)
            roi_mask_cumulative: tensor of shape :math:`(*,1,W,H)` with the double integral (both along row and column)
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
            Exception: If both :attr:`roi_mask` and :attr:`roi_mask_cumulative` are specified
        """
        if roi_mask is not None and roi_mask_cumulative is not None:
            raise Exception("Only one between roi_mask and roi_mask_cumulative can be specified")
        n_crops_per_image = self.n_crops_per_image if n_crops_per_image is None else n_crops_per_image
        bij_list = self.get_index(roi_mask, roi_mask_cumulative, n_crops_per_image)
        return self.collate_crops_from_list(img, list(bij_list))


class DatasetInMemory(Dataset):
    """
    Wrapper around :class:`Dataset` which plays well with :class:`ConditionalRandomCrop`.

    Note:
        This class is specially useful for small datasets which fit into CPU or GPU memory.

    Note:
        It can be used in combination with :class:`ConditionalRandomCrop` to create a
        dataset out of a single large image.
    """

    def __init__(self,
                 x: torch.Tensor,
                 x_roi: Optional[torch.Tensor] = None,
                 y: Optional[torch.Tensor] = None,
                 x_transform: Optional[ConditionalRandomCrop] = None,
                 store_in_cuda: bool = False):
        """
        Args:
            x: the underlying image or batch of images which composed the dataset of shape :math:`(B,C,W,H)`
            x_roi: boolean tensor indicating the Region of Interest (ROI) of shape :math:`(B,1,W,H)`
            y: integer tensor with the image-level labels of shape :math:`(B)`
            x_transform: Either None or a ConditionalRandomCrop which modifies the image on the fly
            store_in_cuda: if true the dataset is stored into GPU memory, if false the dataset is stored in CPU memory.

        Note:
            The underlying data is all stored in memory for faster processing
            (either CPU or GPU depending on :attr:`store_in_cuda`). This means that this class is useful only for
            relatively small datasets.

        Note:
            If :attr:`data_augmentation` is equal to :class:`ConditionalRandomCrop` the dataloader returns different
            random crops at every call thus implementing data augmentation.

        """
        assert len(x.shape) == 4
        assert (x_roi is None or len(x_roi.shape) == 4)
        assert (y is None or y.shape[0] == x.shape[0])
        x_roi_cumulative = None if x_roi is None else x_roi.cumsum(dim=-1).cumsum(dim=-2)
        y = -1*torch.ones(x.shape[0]) if y is None else y

        storing_device = torch.device('cuda') if store_in_cuda else torch.device('cpu')

        self.x_transform = x_transform

        # Compute the new image size
        new_batch_size = x.shape[0] if x_transform is None else x.shape[0] * self.x_transform.n_crops_per_image

        # Note that data is first moved to the storing device and the expanded.
        # This makes sure that memory footprint remains low
        self.x = x.to(storing_device).detach().expand(new_batch_size, -1, -1, -1)
        self.x_roi_cumulative = None if x_roi_cumulative is None else x_roi_cumulative.to(storing_device).detach().expand(new_batch_size, -1, -1, -1)
        self.y = y.to(storing_device).detach().expand(new_batch_size)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index: int):
        if self.x_transform is None:
            return self.x[index], self.y[index], index
        else:
            i, j = self.x_transform.get_index(roi_mask_cumulative=self.x_roi_cumulative[index],
                                              n_crops_per_image=1)[0][-2:]
            return self.x_transform.collate_crops_from_list(self.x, [[index, i, j]]).squeeze(0), self.y[index], index