import torch
import torch.nn.functional as F
import numpy
import pathlib
from typing import Tuple, Optional, Callable
from .namedtuple import Inference, MetricMiniBatch, Segmentation, SparseSimilarity, Output
from .util_vis import draw_bounding_boxes, draw_img
from .model_parts import InferenceAndGeneration
from .util_ml import MetricsAccumulator, SpecialDataSet
from .util import load_json_as_dict, save_dict_as_json, roller_2d, flatten_dict


class CompositionalVae(torch.nn.Module):
    """
    Fundamental class which implements the model described in the
    `CellSegmenter paper <https://arxiv.org/abs/2011.12482/>`_.
    """

    PATH_DEFAULT_CONFIG = (pathlib.Path(__file__).parent.absolute()).joinpath('_default_config_CompositionalVae.json')
    DEFAULT_PARAMS = load_json_as_dict(path=PATH_DEFAULT_CONFIG)

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: dictionary with all simulation hyper-parameters. This dictionary will be saved as a member variable
                and copied to a ckpt when calling :meth:`create_ckpt` so that the entire simulation can be restarted.
                See :meth:`default_config` for an example of a valid dictionary.
        """
        super().__init__()

        # Save the configuration dictionary as-is. This is usefull to save/restart the simulation
        self._config = config

        # Save the configuration variable as member variables. This is useful for accessing the variable later
        for key, value in flatten_dict(config, separator='_', prefix='config').items():
            setattr(self, key, value)

        # Instantiate all the modules
        self.inference_and_generator = InferenceAndGeneration(config=self._config)

        if torch.cuda.is_available():
            self.cuda()

    @classmethod
    def default_config(cls) -> dict:
        """
        Show a dictionary with the default configurations. The intended use is to see the default parameters,
        modify them as necessary before instantiating CompositionalVae.

        Note:
            Can be used in combination with :func:`load_json_as_dict` and :func:`save_dict_as_json` to write/read
            the default and/or modified parameter to json file

        Examples:
            >>> config = CompositionalVae.default_config()
            >>> print(config)
            >>> config["input_image"]["n_objects_max"] = 25
            >>> save_dict_as_json(input_dict=config, path="./new_config.json")
            >>> vae = CompositionalVae(config=load_json_as_dict("./new_config.json"))
        """
        return cls.DEFAULT_PARAMS

    def create_ckpt(self,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    history_dict: Optional[dict] = None,
                    epoch: Optional[int] = None) -> dict:
        """
        Creates a dictionary containing checkpinting information to save and resume the current simulation.
        The state of the model is always saved. In addition the state of the optimizer, the training history and
        the current epoch can also be saved.

        Args:
            optimizer: The optimizer whose state will be saved.
            history_dict: A dictionary with the history of the training up to this point.
            epoch: The current epoch.

        Returns:
            A dictionary containing the current state of the simulation.

        Note:
            If all the arguments are specified then the ckpt is complete and the training can be resumed from the ckpt.
            It this case the state of model at the end of training will be almost identical to the one obtained
            if the training was conducted into a single chunck.
            If no arguments are specified, only the model is saved and the ckpt can be used to initialize a model
            and perform transfer learning.

        Note:
            This function is often used in combination with :class:`ckpt2file` to write the checkpoint to disk.

        Examples:
            >>> history_dict = {}
            >>> for epochs in range(100):
            >>>    train_metrics = process_one_epoch(model=vae,
            >>>                                      dataloader=train_loader,
            >>>                                      optimizer=optimizer)
            >>>    history_dict = append_to_dict(source=train_metrics,
            >>>                                  destination=history_dict,
            >>>                                  prefix_to_add="train_")
            >>>    if (epochs % 10) == 0:
            >>>        ckpt = vae.create_ckpt(optimizer=optimizer,
            >>>                               epoch=epoch,
            >>>                               history_dict=history_dict)
            >>>        ckpt2file(ckpt=ckpt, path='./latest_ckpt.pt')
        """
        all_member_var = self.__dict__
        member_var_to_save = {}
        for k, v in all_member_var.items():
            if not k.startswith("_") and k != 'training' and not k.startswith("config"):
                member_var_to_save[k] = v
                print("saving member variable ->", k)

        ckpt = {'config': self._config,
                'model_member_var': member_var_to_save,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': None if (optimizer is None) else optimizer.state_dict(),
                'history_dict': history_dict,
                'epoch': epoch}

        return ckpt


    @staticmethod
    def _compute_sparse_similarity_matrix(mixing_k: torch.tensor,
                                          batch_of_index: torch.tensor,
                                          max_index: int,
                                          radius_nn: int,
                                          min_threshold: float = 0.01) -> torch.sparse.FloatTensor:
        """
        Compute the similarity between two pixels by computing the dot product of the mixing probabilities.
        To save memory, if the similarity is less than min_threshold the value is not recorded (i.e. effectively zero).
        The user should not use this method directly.
        This method is called internally by the method :meth:`process_batch_imgs` which is exposed to the user.

        Args:
            mixing_k: Tensor of shape :math:`(N_{max_instances}, B, 1, W, H)` containing the mixing probabilities.
            batch_of_index: Tensor of shape :math:`(B, 1, W, H)` with a unique integer identifying each pixel IDs.
                Valid pixel IDs are >= 0. Pixel with ID value less than 0 will be ignored.
            max_index: The maximum pixel ID. This is necessary b/c a sparse matrix can only be constructed if the
                largest possible row and col ID is known.
            radius_nn: size of the neighborhood used to compute the connectivity of each pixel.
            min_threshold: minimum value of the connectivity which will be recorded.

        Returns:
            A sparse matrix in the COO(rdinate) format of size (max_index, max_index). The number of non-zero entries
            is O(max_index) not O(max_index^2) as for dense matrices.

        Note:
            This function is a thin wrapper around a dot product. For each displacement
            (let's say +1 in the x direction) all pixel pairs are computed simultaneously.
            Unfortunately there is no way to compute all displacement without using a for-loop.

        Todo:
            * investigate other measure of similarity such as sqrt(dot_product) or dot_product^2.
        """
        with torch.no_grad():
            # start_time = time.time()
            n_boxes, batch_shape, ch_in, w, h = mixing_k.shape
            assert ch_in == 1
            assert (batch_shape, 1, w, h) == batch_of_index.shape

            # Pad width and height with zero before rolling to avoid spurious connections due to PBC
            pad = radius_nn + 1
            pad_mixing_k = F.pad(mixing_k, pad=[pad, pad, pad, pad], mode="constant", value=0.0)
            pad_index = F.pad(batch_of_index, pad=[pad, pad, pad, pad], mode="constant", value=-1)
            row = batch_of_index[:, 0]  # shape: batch, w, h
            row_ge_0 = (row >= 0)

            sparse_similarity = torch.sparse.FloatTensor(max_index, max_index).to(mixing_k.device)
            # sequential loop over the allowed displacements
            for pad_mixing_k_shifted, pad_index_shifted in roller_2d(a=pad_mixing_k,
                                                                     b=pad_index,
                                                                     radius=radius_nn):
                v = (pad_mixing_k *
                     pad_mixing_k_shifted).sum(dim=-5)[:, 0, pad:(pad + w), pad:(pad + h)]  # shape: batch, w, h
                col = pad_index_shifted[:, 0, pad:(pad + w), pad:(pad + h)]  # shape: batch, w, h

                # check that pixel IDs are valid and connectivity larger than threshold
                mask = (v > min_threshold) * (col >= 0) * row_ge_0

                index_tensor = torch.stack((row[mask], col[mask]), dim=0)
                tmp_similarity = torch.sparse.FloatTensor(index_tensor, v[mask],
                                                          torch.Size([max_index, max_index]))

                # Accumulate the results obtained for different displacements and coalesce the matrix
                # so that if pair of pixel appear more than one the number of non-zero entries is reduced and
                # size of matrix is kept small
                sparse_similarity.add_(tmp_similarity)
                sparse_similarity = sparse_similarity.coalesce()

            # print("similarity time", time.time()-start_time)
            return sparse_similarity

    def segment(self, imgs_in: torch.Tensor,
                k_objects_max: Optional[int] = None,
                prob_corr_factor: Optional[float] = None,
                topk_only: bool = False,
                iom_threshold: float = 0.3,
                noisy_sampling: bool = False,
                draw_boxes: bool = False,
                draw_boxes_ideal: bool = False) -> Segmentation:
        """
        Segment a batch of input images to produce an integer segmentation mask.
        This is intended to be used at test time on images of small size only.
        To segment large images, use :meth:`segment_with_tiling` instead.

        Args:
            imgs_in: Tensor of shape :math:`(B, C, W, H)` with the images to be segmented.
            k_objects_max: Maximum allowed number of foreground object.
                If this number is too low some foreground objects will be missed. If it is too high, computational
                resources will be wasted. It might be beneficial to use a larger value than the one used during
                training. If it is not specified, it is set to the value used during training.
            prob_corr_factor: Number in (0,1) used to encourage the model to focus its attentions on regions which are
                poorly reconstructed by the background component. See :meth:`process_batch_imgs` for details. This value
                should be zero except during the early phase of training.
            topk_only: This value should be set to False. See :meth:`process_batch_imgs` for details.
            iom_threshold: This value has effect only if :attr:`topk_only` is False. Threshold value of the
                IntersectionOverMinimum between two bounding boxes before the non-max-suppressison kicks-in.
                Typical values are :math:`0.3 - 0.5`. It might be beneficial to use a larger value than the one used
                during training.
            noisy_sampling: If true a random sample from the posterior is used. If false the mode of the posterior is
                used. It might beneficial to segment the same images multiple times with :attr:`noisy_sampling` = True.
                A single segmentation with :attr:`noisy_sampling` = False usually gives good results.
            draw_boxes: If true the object bounding boxes are added to the integer segmentation mask.
            draw_boxes_ideal: If true the ideal object bounding boxes are added to the integer segmentation mask.

        Returns:
            A container of type :class:`Segmentation` with both the input images and the integer segmentation mask.
            See :class:`Segmentation` for details.

        Note:
            To segment large images, use :meth:`segment_with_tiling` instead.

        Examples:
            >>> ckpt = file2ckpt(path='./last_ckpt', device='cuda')
            >>> vae = CompositionalVae(config=ckpt.get('config'))
            >>> b, ch, w, h = 10, 1, 80, 80
            >>> img_to_segment = test_loader.load(10)
            >>> segmentation = vae.segment(imgs_in=img_to_segment)
        """

        prob_corr_factor = getattr(self, "prob_corr_factor", 0.0) if prob_corr_factor is None else prob_corr_factor
        k_objects_max = getattr(self, "config_input_image_max_objects_per_patch") if k_objects_max is None else k_objects_max
        iom_threshold = getattr(self, "config_architecture_iom_threshold_test") if iom_threshold is None else iom_threshold

        return self._segment_internal(batch_imgs=imgs_in,
                                      k_objects_max=k_objects_max,
                                      prob_corr_factor=prob_corr_factor,
                                      iom_threshold=iom_threshold,
                                      noisy_sampling=noisy_sampling,
                                      topk_only=topk_only,
                                      draw_boxes=draw_boxes,
                                      draw_boxes_ideal=draw_boxes_ideal,
                                      batch_of_index=None,
                                      max_index=None,
                                      radius_nn=10)

    def _segment_internal(self,
                          batch_imgs: torch.tensor,
                          k_objects_max: int,
                          prob_corr_factor: float,
                          iom_threshold: float,
                          noisy_sampling: bool,
                          topk_only: bool,
                          draw_boxes: bool,
                          draw_boxes_ideal: bool,
                          batch_of_index: Optional[torch.tensor],
                          max_index: Optional[int],
                          radius_nn: int) -> Segmentation:
        """
        This is an private method called by :meth:`segment' and :meth:`segment_with_tiling`.
        The user should not use this method directly.
        All attributes with the exception of :att:`batch_of_index` are explained
        in :meth:`segment' and :meth:`segment_with_tiling`.

        :attr:`batch_of_index` is a integer tensor of shape :math:`(B, 1, W, H) containing the ID for each pixel.
        This allows to compute the pixel similarity in a consistent way even when using a sliding-window approach.
        """

        # start_time = time.time()

        with torch.no_grad():

            inference: Inference
            metrics: MetricMiniBatch
            inference, metrics = self.inference_and_generator(imgs_bcwh=batch_imgs,
                                                              generate_synthetic_data=False,
                                                              prob_corr_factor=prob_corr_factor,
                                                              iom_threshold=iom_threshold,
                                                              k_objects_max=k_objects_max,
                                                              topk_only=topk_only,
                                                              noisy_sampling=noisy_sampling)

            # Now compute fg_prob, integer_segmentation_mask, similarity
            most_likely_mixing, index = torch.max(inference.mixing_kb1wh, dim=-5, keepdim=True)  # 1, batch_size, 1, w, h
            integer_mask = ((most_likely_mixing > 0.5) * (index + 1)).squeeze(-5).to(dtype=torch.int32)  # bg=0 fg=1,2,.
            fg_prob = torch.sum(inference.mixing_kb1wh, dim=-5)  # sum over instances

            bounding_boxes = draw_bounding_boxes(c_kb=inference.sample_c_kb,
                                                 bounding_box_kb=inference.sample_bb_kb,
                                                 width=integer_mask.shape[-2],
                                                 height=integer_mask.shape[-1],
                                                 color='red') if draw_boxes else None

            bounding_boxes_ideal = draw_bounding_boxes(c_kb=inference.sample_c_kb,
                                                       bounding_box_kb=inference.sample_bb_ideal_kb,
                                                       width=integer_mask.shape[-2],
                                                       height=integer_mask.shape[-1],
                                                       color='blue') if draw_boxes_ideal else None

            # print("inference time", time.time()-start_time)

            if batch_of_index is None:
                return Segmentation(raw_image=batch_imgs,
                                    fg_prob=fg_prob,
                                    integer_mask=integer_mask,
                                    bounding_boxes=bounding_boxes,
                                    bounding_boxes_ideal=bounding_boxes_ideal,
                                    similarity=None)

            else:
                max_index = torch.max(batch_of_index) if max_index is None else max_index
                similarity_matrix = CompositionalVae._compute_sparse_similarity_matrix(mixing_k=inference.mixing_kb1wh,
                                                                                       batch_of_index=batch_of_index,
                                                                                       max_index=max_index,
                                                                                       radius_nn=radius_nn,
                                                                                       min_threshold=0.1)
                return Segmentation(raw_image=batch_imgs,
                                    fg_prob=fg_prob,
                                    integer_mask=integer_mask,
                                    bounding_boxes=bounding_boxes,
                                    bounding_boxes_ideal=bounding_boxes_ideal,
                                    similarity=SparseSimilarity(sparse_matrix=similarity_matrix,
                                                                index_matrix=None))

    def segment_with_tiling(self,
                            single_img: torch.Tensor,
                            roi_mask: Optional[torch.Tensor],
                            patch_size: Optional[Tuple[int, int]] = None,
                            stride: Optional[Tuple[int, int]] = None,
                            k_objects_max_per_patch: Optional[int] = None,
                            prob_corr_factor: float = 0.0,
                            topk_only: bool = False,
                            iom_threshold: float = 0.3,
                            radius_nn: int = 5,
                            batch_size: int = 32) -> Segmentation:
        """
        Segment a single, possibly very large, image using an overlapping sliding-window approach.
        It returns an integer segmentation mask obtained by stitching together non-overlapping windows and a
        sparse matrix with the pixel similarities which can be ingested by :class:`GraphSegmentation`
        to perform a graph-based segmentation. See `our paper <https://arxiv.org/abs/2011.12482/>`_ for details.

        Note:
            This method is computationally expensive. It is expected to be used only at test time at the
            end of training. To monitor the quality of the segmentation during training the method :meth:`segment`
            should be applied on a few small sample regions.

        Note:
            This method can segment images which are too large to fit into GPU memory since only few sliding windows
            at the time are copied on GPU for processing. The attribute :attr:`batch_size` controls the number of
            sliding windows which are processed simultaneously and can be adjusted to ensure that the computation fits
            into GPU memory.

        Note:
            To segment many small images (instead of a single large one) it is better to use :meth:`segment`

        Args:
            single_img: Tensor of shape :math:`(C, W, H)` with a (possibly very large)
                image to be segmented.
            roi_mask: Boolean tensor of shape :math:`(C, W, H)` indicating the region of interest in the images (ROI).
                To save computational resources only sliding-windows with at least 1% of ROI will be processed. This is
                specially helpful for large biological images in which the sample of interest occupies a small,
                irregularly shaped region of the image. If it is not specified the entire image is processed.
            patch_size: The size of the sliding-window. If it is not specified, the size of the sliding window is set to
                the size used during training.
            stride: The stride used to displace the sliding windows. The :attr:`crop_size` should be a multiple of
                the stride. If it is not specified is set to :math:`1/4` of the :attr:`crop_size`.
            k_objects_max_per_patch: Maximum allowed number of foreground object in each sliding-window.
                If this number is too low some foreground objects will be missed. If it is too high, computational
                resources will be wasted. It might be beneficial to use a larger value than the one used during
                training. If it is not specified, it is set to the value used during training.
            prob_corr_factor: Number in (0,1) used to encourage the model to focus its attentions on regions which are
                poorly reconstructed by the background component. See :meth:`process_batch_imgs` for details. This value
                should be zero except during the early phase of training.
            topk_only: This value should be set to False. See :meth:`process_batch_imgs` for details.
            iom_threshold: This value has effect only if :attr:`topk_only` is False. Threshold value of the
                IntersectionOverMinimum between two bounding boxes before the non-max-suppressison kicks-in.
                Typical values are :math:`0.3 - 0.5`. It might be beneficial to use a larger value than the one used
                during training. If it is not specified, it is set to the value used during training.
            radius_nn: Size of neighborhood used to compute the similarity among pixel-pairs.
                This value should a fraction (0.25 - 0.5) of the typical cell size. If this value is big the
                computation of the sparse similarity matrix becomes very expensive. It if it too small the graph-based
                segmentation (see :class:`GraphSegmentation`) might lead to poor results.
            batch_size: Multiple sliding windows will be processed in batches of size = :attr:`batch_size`. To be
                adjusted to guaranteed that the calculation fits into GPU memory.

        Returns:
            :class:`Segmentation` containing an integer segmentation mask obtained by stitching together
            non-overlapping windows and a sparse matrix with the pixel similarities which can be ingested
            by :class:`GraphSegmentation` to perform a graph-based segmentation. See `our paper
            <https://arxiv.org/abs/2011.12482/>`_ for details.

        Examples:
            >>> # get the image to segment
            >>> img_to_segment = train_loader.x[0, :, :300, :300]
            >>> roi_mask_to_segment = train_loader.x_roi[0, :, :300, :300]
            >>> # load the trained model
            >>> ckpt = file2ckpt(path='./last_ckpt', device='cuda')
            >>> vae = CompositionalVae(config=ckpt.get('config'))
            >>> tiling = vae.segment_with_tiling(single_img=img_to_segment,
            >>>                                  roi_mask=roi_mask_to_segment,
            >>>                                  patch_size=(80, 80),
            >>>                                  stride=(40, 40),
            >>>                                  k_objects_max_per_patch=25,
            >>>                                  prob_corr_factor=0,
            >>>                                  iom_threshold=0.4,
            >>>                                  radius_nn=10,
            >>>                                  batch_size=64)
        """
        assert len(single_img.shape) == 3
        assert roi_mask is None or len(roi_mask.shape) == 3

        patch_size = (getattr(self, "config_input_image_size_image_patch"),
                      getattr(self, "config_input_image_size_image_patch")) if patch_size is None else patch_size
        stride = (int(patch_size[0] // 4), int(patch_size[1] // 4)) if stride is None else stride
        k_objects_max_per_patch = getattr(self, "config_input_image_max_objects_per_patch") if \
            k_objects_max_per_patch is None else k_objects_max_per_patch

        assert patch_size[0] % stride[0] == 0, "crop and stride size are NOT compatible"
        assert patch_size[1] % stride[1] == 0, "crop and stride size are NOT compatible"
        assert len(single_img.shape) == 3  # ch, w, h

        with torch.no_grad():

            w_img, h_img = single_img.shape[-2:]
            n_prediction = (patch_size[0] // stride[0]) * (patch_size[1] // stride[1])
            print(f'Each pixel will be segmented {n_prediction} times')

            pad_w = patch_size[0] - stride[0]
            pad_h = patch_size[1] - stride[1]
            pad_list = [pad_w, patch_size[0], pad_h, patch_size[1]]

            # This is duplicating the single_img on the CPU
            # Note: unsqueeze, pad, suqeeze
            try:
                img_padded = F.pad(single_img.cpu().unsqueeze(0),
                                   pad=pad_list, mode='reflect')  # 1, ch_in, w_pad, h_pad
            except RuntimeError:
                img_padded = F.pad(single_img.cpu().unsqueeze(0),
                                   pad=pad_list, mode='constant', value=0)  # 1, ch_in, w_pad, h_pad
            w_paddded, h_padded = img_padded.shape[-2:]

            # This is creating the index matrix on the cpu
            max_index = w_img * h_img
            index_matrix_padded = F.pad(torch.arange(max_index,
                                                     dtype=torch.long,
                                                     device=torch.device('cpu')).view(1, 1, w_img, h_img),
                                        pad=pad_list, mode='constant', value=-1)  # 1, 1, w_pad, h_pad

            assert index_matrix_padded.shape[-2:] == img_padded.shape[-2:]
            assert index_matrix_padded.shape[0] == img_padded.shape[0]
            assert len(index_matrix_padded.shape) == len(img_padded.shape)

            # Build a list with the locations of the corner of the images
            location_of_corner = []
            for i in range(0, w_img + pad_w, stride[0]):
                for j in range(0, h_img + pad_h, stride[1]):
                    location_of_corner.append([i, j])

            ij_tmp = torch.tensor(location_of_corner, device=torch.device('cpu'), dtype=torch.long)  # shape: N, 2
            x1 = ij_tmp[..., 0]
            y1 = ij_tmp[..., 1]
            del ij_tmp

            if roi_mask is not None:
                assert roi_mask.shape[-2:] == single_img.shape[-2:]

                # pad before computing the cumsum
                roi_mask_padded = F.pad(roi_mask, pad=pad_list, mode='constant', value=0)
                cum_roi_mask = torch.cumsum(torch.cumsum(roi_mask_padded, dim=-1), dim=-2)
                assert cum_roi_mask.shape[-2:] == img_padded.shape[-2:]

                # Exclude stuff if outside the roi_mask
                integral = cum_roi_mask[0, x1 + patch_size[0] - 1, y1 + patch_size[1] - 1] - \
                           cum_roi_mask[0, x1 - 1, y1 + patch_size[1] - 1] * (x1 > 0) - \
                           cum_roi_mask[0, x1 + patch_size[0] - 1, y1 - 1] * (y1 > 0) + \
                           cum_roi_mask[0, x1 - 1, y1 - 1] * (x1 > 0) * (y1 > 0)
                fraction = integral.float() / (patch_size[0] * patch_size[1])
                mask = fraction > 0.01  # if there is more than 1% ROI the patch will be processed.
                x1 = x1[mask]
                y1 = y1[mask]
                del cum_roi_mask
                del mask

            print(f'I am going to process {x1.shape[0]} patches')
            if not (x1.shape[0] >= 1):
                raise Exception("No patches will be analyzed. Something went wrong!")

            # split the list in chunks of batch_size
            index = torch.arange(0, x1.shape[0], dtype=torch.long, device=torch.device('cpu'))
            n_list_of_list = [index[n:n + batch_size] for n in range(0, index.shape[0], batch_size)]
            n_instances_tot = 0
            need_initialization = True
            for n_batches, n_list in enumerate(n_list_of_list):

                batch_imgs = torch.cat([img_padded[...,
                                        x1[n]:x1[n] + patch_size[0],
                                        y1[n]:y1[n] + patch_size[1]] for n in n_list], dim=-4)

                batch_index = torch.cat([index_matrix_padded[...,
                                         x1[n]:x1[n] + patch_size[0],
                                         y1[n]:y1[n] + patch_size[1]] for n in n_list], dim=-4)

                # print progress
                if (n_batches % 10 == 0) or (n_batches == len(n_list_of_list) - 1):
                    print(f'{n_batches} out of {len(n_list_of_list) - 1} -> batch_of_imgs.shape = {batch_imgs.shape}')

                segmentation = self._segment_internal(batch_imgs=batch_imgs.to(self.sigma_fg.device),
                                                      k_objects_max=k_objects_max_per_patch,
                                                      prob_corr_factor=prob_corr_factor,
                                                      iom_threshold=iom_threshold,
                                                      noisy_sampling=True,
                                                      topk_only=topk_only,
                                                      draw_boxes=False,
                                                      draw_boxes_ideal=False,
                                                      batch_of_index=batch_index.to(self.sigma_fg.device),
                                                      max_index=max_index,
                                                      radius_nn=radius_nn)
                # print("segmentation time", time.time()-start_time)

                # Initialize only the fist time
                if need_initialization:
                    # Probability and integer mask are dense tensor
                    big_fg_prob = torch.zeros((w_paddded, h_padded),
                                              device=torch.device('cpu'),
                                              dtype=segmentation.fg_prob.dtype)
                    big_integer_mask = torch.zeros((w_paddded, h_padded),
                                                   device=torch.device('cpu'),
                                                   dtype=segmentation.integer_mask.dtype)
                    # Similarity is a sparse tensor
                    sparse_similarity_matrix = torch.sparse.FloatTensor(max_index, max_index).cpu()
                    need_initialization = False

                # Unpack the data from batch
                sparse_similarity_matrix.add_(segmentation.similarity.sparse_matrix.cpu())
                sparse_similarity_matrix = sparse_similarity_matrix.coalesce()
                fg_prob = segmentation.fg_prob.cpu()
                integer_mask = segmentation.integer_mask.cpu()

                for k, n in enumerate(n_list):
                    big_fg_prob[x1[n]:x1[n] + patch_size[0], y1[n]:y1[n] + patch_size[1]] += fg_prob[k, 0]

                    # Find a set of not-overlapping tiles to obtain a sample segmentation (without graph clustering)
                    if ((x1[n] - pad_w) % patch_size[0] == 0) and ((y1[n] - pad_h) % patch_size[1] == 0):
                        n_instances = torch.max(integer_mask[k])
                        shifted_integer_mask = (integer_mask[k] > 0) * \
                                               (integer_mask[k] + n_instances_tot)
                        n_instances_tot += n_instances
                        big_integer_mask[x1[n]:x1[n] + patch_size[0],
                            y1[n]:y1[n] + patch_size[1]] = shifted_integer_mask[0]

            # End of loop over batches
            sparse_similarity_matrix.div_(n_prediction)
            big_fg_prob.div_(n_prediction)

            return Segmentation(raw_image=single_img[None],
                                fg_prob=big_fg_prob[None, None, pad_w:pad_w + w_img, pad_h:pad_h + h_img],
                                integer_mask=big_integer_mask[None, None, pad_w:pad_w + w_img, pad_h:pad_h + h_img],
                                bounding_boxes=None,
                                bounding_boxes_ideal=None,
                                similarity=SparseSimilarity(sparse_matrix=sparse_similarity_matrix,
                                                            index_matrix=index_matrix_padded[0, 0, pad_w:pad_w + w_img,
                                                                                             pad_h:pad_h + h_img]))

    # this is the fully generic function which has all the options unspecified
    def process_batch_imgs(self,
                           imgs_in: torch.Tensor,
                           generate_synthetic_data: bool,
                           topk_only: bool,
                           draw_image: bool,
                           draw_bg: bool,
                           draw_boxes: bool,
                           draw_boxes_ideal: bool,
                           noisy_sampling: bool,
                           prob_corr_factor: float,
                           iom_threshold: float,
                           k_objects_max: int) -> Output:
        """
        General method which process a batch of input images through the Variational Auto Encoder (i.e. applying both
        the encoder and the decoder). The output contains the latent variables, the metrics which are worth
        monitoring during training and, if draw_image=True, the reconstructed images.

        Args:
            imgs_in: Bath of input images of shape :math:`(B, C, W, H)` to be processed.
            generate_synthetic_data: If true the generator draw samples from the prior not the posterior. This means
                that the output images will not be a reconstruction of the input but new images.
            topk_only: If true the top-K bounding boxes by probability are processed. If false, the bounding box undergo
                non-max-suppression before applying the top-K filter. In practice :attr:`topk_only` should always set
                to False. This flag is kept to test the effect of the non-max-suppression operation on the result.
            draw_image: If true the reconstructed images are generated.
            draw_bg: This flag has effect only if :attr:`draw_image` is True. If true the reconstructed background is
                added to the output images. If false only the reconstructed foreground instances will be present in
                the output images.
            draw_boxes: This flag has effect only if :attr:`draw_image` is True. If true the object bounding boxes are
                added to the output images in red.
            draw_boxes_ideal: This flag has effect only if :attr:`draw_image` is True.
                If true the optimal object bounding boxes are added to the output images in blue.
            noisy_sampling: If true a random sample from either the prior or the posterior (depending on the value of
                :attr:`generate_synthetic_data` ) is used. If false the mode of the prior or posterior is used.
            prob_corr_factor: Number in (0,1) used to encourage the model to focus its attentions on
                regions which are poorly reconstructed by the background component. The object recognition
                probabilities are modified as:

                .. math::
                    p = (1- \\text{prob_corr_factor} ) * p_{net} + \\text{prob_corr_factor} * \\delta_p

                where :math:`p_{net}` is the probability generated by the NeuralNet and :math:`\\delta_p` in (0,1) is
                a measure of discordance between the input image the reconstructed background component. The typical
                use is to anneal :attr:`prob_corr_factor` from 0.5 to 0 during the initial phase of training.
            iom_threshold:  This value has effect only if :attr:`topk_only` is False. Threshold value of the
                IntersectionOverMinimum between two bounding boxes before the non-max-suppressison kicks-in.
                Typical values are :math:`0.3 - 0.5`.
            k_objects_max: Maximum number of foreground instances. If this value if too small some foreground objects
                will not be reconstructed or the model will be forced to clamp multiple objects into few bounding boxes.
                If this value is too high, the model will waste computational resources.

        Returns:
            A container of type :class:`output` with the latent variables, the metrics which are worth
            monitoring during training and, if :attr:`draw_image` is True, the reconstructed images.

        Note:
            All the arguments need to be explicitly specified.

        Note:
            Setting both :attr:`draw_boxes` = :attr:`draw_boxes_ideal` = True is a great way to visualize how the model
            is learning. If the actual and ideal bounding boxes are very different consider increasing the
            bounding_box_regression_penalty_strength in the config file.

        Note:
            Other methods such as :meth:`generate` and :meth:`forward` are thin wrappers around this general method.
            These methods have predefined parameters specified for different use cases.

        Note:
            During training, it is a good practice to set :attr:`k_objects_max` to be 50% larger than the typical
            number of instances in the image. To achieve high recall, it might be beneficial to increase
            :attr:`k_objects_max` at test time.

        Note:
            It might be beneficial to use a stringent value of :attr:`iom_threshold` (0.5) during training and
            a increase :attr:`iom_threshold` (0.7) during testing to achieve high recall.

        Examples:
            >>> # initialize a new model with default parameters
            >>> config = CompositionalVae.default_config()
            >>> vae = CompositionalVae(config=config)
            >>> output = vae.process_batch_imgs(imgs_in=train_loader.load(8),
            >>>                                 generate_synthetic_data=False,
            >>>                                 topk_only=False,
            >>>                                 draw_image=False,
            >>>                                 draw_bg=False,
            >>>                                 draw_boxes=False,
            >>>                                 noisy_sampling=True,
            >>>                                 prob_corr_factor=0.0,
            >>>                                 iom_threshold=0.3,
            >>>                                 n_objects_max=-25)
        """

        # Checks
        assert len(imgs_in.shape) == 4
        assert getattr(self, "config_architecture_n_ch_img") == imgs_in.shape[-3]
        # End of Checks #
        inference: Inference
        metrics: MetricMiniBatch
        inference, metrics = self.inference_and_generator(imgs_bcwh=imgs_in,
                                                          generate_synthetic_data=generate_synthetic_data,
                                                          prob_corr_factor=prob_corr_factor,
                                                          iom_threshold=iom_threshold,
                                                          k_objects_max=k_objects_max,
                                                          topk_only=topk_only,
                                                          noisy_sampling=noisy_sampling)

        with torch.no_grad():
            if draw_image:
                imgs_rec = draw_img(inference=inference,
                                    draw_bg=draw_bg,
                                    draw_boxes=draw_boxes,
                                    draw_ideal_boxes=draw_boxes_ideal)
            else:
                imgs_rec = None  # -1*torch.ones_like(imgs_in)

        return Output(metrics=metrics, inference=inference, imgs=imgs_rec)

    def forward(self,
                imgs_in: torch.tensor,
                iom_threshold: float,
                noisy_sampling: bool = True,
                draw_image: bool = False,
                draw_bg: bool = False,
                draw_boxes: bool = False,
                draw_boxes_ideal: bool = False):
        """
        Wrapper around :meth:`process_batch_imgs` with some parameters pre-specified to values that
        make sense for training the model. See :meth:`process_batch_imgs` for details.

        Examples:
            >>> batch, ch, w, h = 4, 1, 80, 80
            >>> config = CompositionalVae.default_config()
            >>> vae = CompositionalVae(config)
            >>> imgs_in = torch.zeros((batch,ch,w,h), device=torch.device('cuda'), dtype=torch.float)
            >>> output = vae.forward(imgs_in)
        """

        return self.process_batch_imgs(imgs_in=imgs_in,
                                       generate_synthetic_data=False,
                                       topk_only=False,
                                       draw_image=draw_image,
                                       draw_bg=draw_bg,
                                       draw_boxes=draw_boxes,
                                       draw_boxes_ideal=draw_boxes_ideal,
                                       noisy_sampling=noisy_sampling,
                                       prob_corr_factor=getattr(self, "prob_corr_factor", 0.0),
                                       iom_threshold=iom_threshold,
                                       k_objects_max=getattr(self, "config_input_image_max_objects_per_patch"))

    def generate(self,
                 imgs_in: torch.Tensor,
                 draw_bg: bool = False,
                 draw_boxes: bool = False,
                 draw_boxes_ideal: bool = False):
        """
        Wrapper around :meth:`process_batch_imgs` with some parameters pre-specified to draw random samples
        from the generator. See :meth:`process_batch_imgs` for details.

        Note:
            The content of :attr:`imgs_in` is ignored but it is used to determine the size of the output images.
            The generated images will have the same shape as :attr:`imgs_in`.

        Example:
            >>> batch, ch, w, h = 4, 1, 80, 80
            >>> config = CompositionalVae.default_config()
            >>> vae = CompositionalVae(config)
            >>> imgs_in = torch.zeros((batch,ch,w,h), device=torch.device('cuda'), dtype=torch.float)
            >>> output = vae.generate(imgs_in)
            >>> imgs_generated = output.imgs
        """

        with torch.no_grad():
            return self.process_batch_imgs(imgs_in=torch.zeros_like(imgs_in),
                                           generate_synthetic_data=True,
                                           topk_only=False,
                                           draw_image=True,
                                           draw_bg=draw_bg,
                                           draw_boxes=draw_boxes,
                                           draw_boxes_ideal=draw_boxes_ideal,
                                           noisy_sampling=True,
                                           prob_corr_factor=0.0,
                                           iom_threshold=-1.0,
                                           k_objects_max=getattr(self, "config_input_image_max_objects_per_patch"))


def load_from_ckpt(ckpt: dict,
                   model: Optional[CompositionalVae] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   overwrite_member_var: bool = False):
    """
    Use a ckpt dictionary to reset the state of the model and optimizer.

    Args:
        ckpt: Dictionary containing the information create by invoking :meth:`create_ckpt`
            or :func:`file2ckpt` (required).
        model: Instance of :class:`CompositionalVae` module whose state will be changed.
        optimizer: Torch optimizer whose state will be changed
        overwrite_member_var: If true the module member variable are overwritten according to ckpt.
            If false only the weight and biases of the net are overwrittedn but not the member variables.

    Note:
        Used in combination with :class:`file2ckpt` and :class:`create_from_ckpt` can be used to resume a
        simulation which was saved to file

    Examples:
        >>> # to resume training everything is reset to the value saved in the ckpt
        >>> ckpt = file2ckpt(path="ckpt.pt", device='cuda')
        >>> config = ckpt.get("config")
        >>> epoch = ckpt.get("epoch")
        >>> history_dict = ckpt.get("history_dict")
        >>> vae = CompositionalVae(config=config)
        >>> optimizer = instantiate_optimizer(model=vae, config_optimizer=config["optimizer"])
        >>> load_from_ckpt(ckpt=ckpt, model=vae, optimizer=optimizer, overwrite_member_var=True)
        >>>
        >>> # alternatively to perform transfer learning only weight and biases are changed
        >>> config = CompositionalVae.default_config()
        >>> epoch = 0
        >>> history_dict = {}
        >>> vae = CompositionalVae(config=config)
        >>> load_from_ckpt(ckpt=ckpt, model=vae)
        >>> optimizer = instantiate_optimizer(model=vae, config_optimizer=config["optimizer"])
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if model is not None:
        # load member variables
        if overwrite_member_var:
            for key, value in ckpt['model_member_var'].items():
                setattr(model, key, value)

        # load the modules
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return


def instantiate_optimizer(model: CompositionalVae, config_optimizer: dict) -> torch.optim.Optimizer:
    """
    Instantiate a optimizer object to optimize the trainable parameters of the model.

    Args:
        model: The torch.module whose parameter will be optimized (required).
        dict_params_optimizer: Dictionary containing the hyperparameters of the optimizers such
            as learning rate, betas, weight_decay etc. (required).

    Returns:
        An optimizer object

    Note:
        The state of the optimizer can be loaded from a ckpt by invoking :class:`load_from_ckpt`.

    Note:
        The instantiated optimizer separates the parameters into two groups. The parameters related to the GECO loss
        function can be optimized with different learning rate than the test.

    Raises:
        Exception: If dict_params_optimizer["type"] is neither "adam" nor "SGD"

    Examples:
        >>> # to instantiate a new model and optimizer
        >>> config = CompositionalVae.default_config()
        >>> vae = CompositionalVae(param=param)
        >>> optimizer = instantiate_optimizer(model=vae, config_optimizer=config["optimizer"])
    """
    # split the parameters between GECO and NOT_GECO
    geco_params, similarity_params, other_params = [], [], []
    for name, param in model.named_parameters():
        if ".geco" in name:
            # print("geco params -->", name)
            geco_params.append(param)
        elif ".similarity" in name:
            # print("similarity params -->", name)
            similarity_params.append(param)
        else:
            other_params.append(param)

    if config_optimizer["type"] == "adam":
        optimizer = torch.optim.Adam([{'params': geco_params, 'lr': config_optimizer["base_lr_geco"],
                                       'betas': config_optimizer["betas_geco"]},
                                      {'params': similarity_params, 'lr': config_optimizer["base_lr_similarity"],
                                       'betas': config_optimizer["betas_similarity"]},
                                      {'params': other_params, 'lr': config_optimizer["base_lr"],
                                       'betas': config_optimizer["betas"]}],
                                     eps=config_optimizer["eps"],
                                     weight_decay=config_optimizer["weight_decay"])

    elif config_optimizer["type"] == "SGD":
        optimizer = torch.optim.SGD([{'params': geco_params, 'lr': config_optimizer["base_lr_geco"]},
                                     {'params': similarity_params, 'lr': config_optimizer["base_lr_similarity"]},
                                     {'params': other_params, 'lr': config_optimizer["base_lr"]}],
                                    weight_decay=config_optimizer["weight_decay"])
    else:
        raise Exception("optimizer type is not recognized")
    return optimizer


def instantiate_scheduler(optimizer: torch.optim.Optimizer, config_scheduler: dict) -> torch.optim.lr_scheduler:
    """
    Instantiate a optimizer scheduler.

    Args:
        optimizer: The optimizer whose state will be controlled by the scheduler.
        config_scheduler: Dictionary containing the hyperparameters of the scheduler.

    Returns:
        A scheduler object.

    Raises:
        Exception: If dict_params_scheduler["type"] is not "step_LR"

    Examples:
        >>> # to instantiate a new model, optimizer and scheduler
        >>> config = CompositionalVae.default_config()
        >>> vae = CompositionalVae(config)
        >>> optimizer = instantiate_optimizer(model=vae, config_optimizer=config["optimizer"])
        >>> scheduler = instantiate_scheduler(optimizer=optimizer, config_scheduler=config["scheduler"])
        >>> for epoch in range(100):
        >>>     train_metrics = process_one_epoch(model=vae,
        >>>                                       dataloader=train_loader,
        >>>                                       optimizer=optimizer,
        >>>                                       scheduler=scheduler)
    """
    if config_scheduler["type"] == "step_LR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config_scheduler["step_size"],
                                                    gamma=config_scheduler["gamma"],
                                                    last_epoch=-1)
    else:
        raise Exception("scheduler type is not recognized")
    return scheduler


def process_one_epoch(model: CompositionalVae,
                      dataloader: SpecialDataSet,
                      optimizer: torch.optim.Optimizer,
                      scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                      weight_clipper: Optional[Callable[[None], None]] = None,
                      verbose: bool = False,
                      noisy_sampling: bool = True,
                      iom_threshold: float = 0.5) -> MetricMiniBatch:
    """
    Takes a model, a dataloader and an optimizer and process one full epoch.

    Args:
        model: An instace of the `class:`CompositionalVae` which process a batch of images and returns a
            metric namedtuple which must include the ".loss" member
        dataloader: An instace of the `class:`SpeacialDataset` which
            returns images, pixel_level_labels, image_level_labels and index of the image
        optimizer: Any torch optimizer
        scheduler: Any torch learning rate scheduler
        weight_clipper: If specified the model's weights are clipped after each optimization step
        verbose: if true information are printed for each mini-batch.
        noisy_sampling: value which is passed to the model.forward method
        iom_threshold: value which is passed to the model.forward method
    """
    metric_accumulator = MetricsAccumulator()  # initialize an empty accumulator
    n_exact_examples = 0
    wrong_examples = []

    # Anomaly detection is slow but help with debugging
    with torch.autograd.set_detect_anomaly(mode=False):

        # Start loop over minibatches
        for i, (imgs, seg_mask, labels, index) in enumerate(dataloader):

            # Put data in GPU if available
            imgs = imgs.cuda() if (torch.cuda.is_available() and (imgs.device == torch.device('cpu'))) else imgs

            metrics = model.forward(imgs_in=imgs,
                                    iom_threshold=iom_threshold,
                                    noisy_sampling=noisy_sampling,
                                    draw_image=False,
                                    draw_bg=False,
                                    draw_boxes=False,
                                    draw_boxes_ideal=False).metrics  # the forward function returns metric and other stuff

            if verbose:
                print("i = %3d train_loss=%.5f" % (i, metrics.loss.item()))

            # Accumulate metrics over an epoch
            with torch.no_grad():
                metric_accumulator.accumulate(source=metrics, counter_increment=len(index))

                # special treatment for count_prediction, wrong_example, accuracy
                metric_accumulator.set_value(key="count_prediction", value=-1 * numpy.ones(1))
                metric_accumulator.set_value(key="wrong_examples", value=-1 * numpy.ones(1))
                metric_accumulator.set_value(key="accuracy", value=-1)

                mask_exact = (metrics.count_prediction == labels.cpu().numpy())
                n_exact_examples += numpy.sum(mask_exact)
                wrong_examples += list(index[~mask_exact].cpu().numpy())

            # Only if training I apply backward
            if model.training:
                optimizer.zero_grad()
                metrics.loss.backward()  # do back_prop and compute all the gradients
                optimizer.step()  # update the parameters

                # apply the weight clipper
                if weight_clipper is not None:
                    model.__self__.apply(weight_clipper)
                    # torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=clipping_value)

        # End of loop over minibatches

        # Only if training apply the scheduler
        if model.training and scheduler is not None:
            scheduler.step()

        # At the end of the loop compute a dictionary with the average of the metrics over one epoch
        with torch.no_grad():
            metric_one_epoch = metric_accumulator.get_average()
            accuracy = float(n_exact_examples) / (n_exact_examples + len(wrong_examples))
            metric_one_epoch["accuracy"] = accuracy
            metric_one_epoch["wrong_examples"] = numpy.array(wrong_examples)
            metric_one_epoch["count_prediction"] = -1 * numpy.ones(1)

            # Make a namedtuple out of the OrderDictionary.
            # Since OrderDictionary preserves the order this preserved order of term.
            return MetricMiniBatch._make(metric_one_epoch.values())  # is this a robust way to convert dict to namedtuple?
