import PIL.Image
import PIL.ImageDraw
import torch
import numpy
import neptune.new as neptune
from neptune.new.types import File
import skimage.color
import skimage.morphology
from typing import Tuple, Optional, Union
from torch.distributions.utils import broadcast_all
from torchvision import utils
from matplotlib import colors as mplclr
from matplotlib import pyplot as plt
from IPython.display import HTML
from matplotlib import animation
from .namedtuple import BB, Output, Segmentation, Suggestion, Inference

""" This module contains specialized plotting functions to visualize the segmentation results 
and the training process. In most cases if :attr:`experiment` (of :class:`neptune.run.Run`) and 
:attr:`neptune_name` of class str are specified then the corresponding image is logged into neptune """


#  Functions whose documentation should NOT be exposed

def _contours_from_labels(labels: numpy.ndarray,
                         contour_thickness: int = 1) -> numpy.ndarray:
    """
    This is an private functions.
    Given an image with integer labels, it returns a boolean mask with the contours.
    The user should not use this function directly. Call draw_contours_from_labels instead.
    """
    assert isinstance(labels, numpy.ndarray)
    assert len(labels.shape) == 2
    assert contour_thickness >= 1
    contours = (skimage.morphology.dilation(labels) != labels)

    for i in range(1, contour_thickness):
        contours = skimage.morphology.binary_dilation(contours)
    return contours


def _draw_contours(image: numpy.ndarray, contours: numpy.ndarray, contours_color: str = 'red') -> numpy.ndarray:
    """
    This is an private functions.
    Given an image and a boolean mask with the location of the contours it generate the image with the
    contours.
    The user should not use this function directly. Call draw_contours_from_labels instead.
    """
    assert isinstance(image, numpy.ndarray)
    assert isinstance(contours, numpy.ndarray)
    assert contours.dtype == bool
    if (len(image.shape) == 3) and (image.shape[-1] == 3):
        image_with_contours = image
    elif len(image.shape) == 2:
        image_with_contours = skimage.color.gray2rgb(image)
    else:
        raise Exception
    if contours_color == 'red':
        ch_index = 0
    elif contours_color == 'green':
        ch_index = 1
    elif contours_color == 'blue':
        ch_index = 2
    else:
        raise Exception("contours_color not recognized. Should be 'red' or 'green' or 'blue'")

    image_with_contours[contours, :] = 0
    image_with_contours[contours, ch_index] = numpy.max(image_with_contours)
    return image_with_contours


# --------------------------------------------------
#  Functions whose documentation should be exposed
# --------------------------------------------------


def draw_contours_from_labels(image: numpy.ndarray,
                              label: numpy.ndarray,
                              contour_thickness: int = 1,
                              contour_color: str = 'red') -> numpy.ndarray:
    """
    Args:
        image: Tensor of dimension 2 (gray scale) or 3 (RGB)
        label: Tensor of dimension 2 with the integer labels
        contour_thickness: Integer, the thickness of the contour
        contour_color: String with the color of the contours. Possible values are: "red". "green" or "blue"

    Returns:
        A numpy array with the same spatial dimension as the input and 3 color channels.
        The array contains the image with the contours overlaid.
    """
    contours = _contours_from_labels(label, contour_thickness)
    image_with_contours = _draw_contours(image, contours, contour_color)
    return image_with_contours


def plot_img_contours_label(image: Union[torch.Tensor, numpy.ndarray],
                            label: Union[torch.Tensor, numpy.ndarray],
                            window: Optional[tuple] = None,
                            contour_thickness: int = 2,
                            contour_color: str = 'red',
                            figsize: tuple = (24, 24),
                            experiment: Optional[neptune.run.Run] = None,
                            neptune_name: Optional[str] = None):
    """ Simple wrapper which plots 3 things:
        a) the raw image
        b) the raw image with overlaid the contours
        c) the raw image with overlaid the labels

        Args:
            image: Tensor of dimension 2 (gray scale) or 3 (RGB)
            label: Tensor of dimension 2 with the integer labels
            window: Tuple with 4 integers: (min_row, min_col, max_row, max_col) to identify the region to visualize
            contour_thickness: Integer, the thickness of the contour
            contour_color: String with the color of the contours. Possible values are: "red". "green" or "blue"
            figsize: Tuple with 2 integers, the size of the figure
            experiment: Neptune experiment to log the resulting image
            neptune_name: Name to log the image in Neptune
    """
    assert len(label.shape) == 2, f"Lable should be a 2D tensor. Got {len(label.shape)} instead."
    assert len(image.shape) == 2 or len(image.shape) == 3, f"Image should be a 2D (grayscale) or 3D (RGB) array." \
                                                           f"Got {len(image.shape)} instead."
    
    assert len(label.shape) == 2
    if torch.is_tensor(label):
        label = label.cpu().numpy()

    if torch.is_tensor(image):
        if len(image.shape) == 3:
            image = image.permute(1, 2, 0).cpu().numpy()
        else:
            image = image.cpu().numpy()
    if len(image.shape) == 3 and (image.shape[-1] != 3):
        image = image[..., 0]

    assert image.shape[:2] == label.shape[:2]

    if window is None:
        window = [0, 0, label.shape[-2], label.shape[-1]]
    else:
        window = (max(0, window[0]),
                  max(0, window[1]),
                  min(label.shape[-2], window[2]),
                  min(label.shape[-1], window[3]))

    img_with_contours = draw_contours_from_labels(image=image[window[0]:window[2], window[1]:window[3]],
                                                  label=label[window[0]:window[2], window[1]:window[3]],
                                                  contour_thickness=contour_thickness,
                                                  contour_color=contour_color)

    fig, ax = plt.subplots(ncols=3, figsize=figsize)
    ax[0].imshow(image[window[0]:window[2], window[1]:window[3]], cmap='gray')
    ax[1].imshow(img_with_contours)
    ax[2].imshow(skimage.color.label2rgb(label=label[window[0]:window[2], window[1]:window[3]],
                                         image=image[window[0]:window[2], window[1]:window[3]],
                                         alpha=0.25, 
                                         bg_label=0))

    fig.tight_layout()
    if (neptune_name is not None) and (experiment is not None):
        experiment[neptune_name].log(neptune.types.File.as_image(fig))

    plt.close(fig)
    return fig


def draw_img(imgs_in: torch.Tensor,
             inference: Inference,
             draw_bg: bool,
             draw_boxes: bool,
             draw_ideal_boxes: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Draw the reconstructed image and input image with the bounding box on top.

    Args:
        imgs_in: The raw image of shape (*, c, W, H)
        inference: :class:'Inference' with all the inferred quantities obtained from running
            the method :math:'process_batch_imgs'
        draw_bg: If true the reconstructed background is added to the output images.
        draw_boxes: If true the object bounding boxes are added to the output images.
        draw_ideal_boxes: If true the optimal object bounding boxes are added to the output images.

    Returns:
        For each input image it returns 2 images. The reconstructed image with overlaid the bounding boxes
        and the input image with overlaid the bounding boxes. This second is useful for debugging and understand if
        the model has learned to localize the objects accurately.

    Note:
        Works for any number of leading dimensions. Each leading dimension will be processed independently

    Note:
        The inferred bounding boxes with p>0.5 are shown in red
        The inferred bounding boxes with p<0.5 are shown in blue
        The ideal bounding boxes are shown in green
    """

    bb_inferred = draw_bounding_boxes(bounding_box=inference.sample_bb_k,
                                      width=imgs_in.shape[-2],
                                      height=imgs_in.shape[-1],
                                      c=inference.sample_prob_k >= 0.5,
                                      color="red")

    bb_almost_inferred = draw_bounding_boxes(bounding_box=inference.sample_bb_k,
                                             width=imgs_in.shape[-2],
                                             height=imgs_in.shape[-1],
                                             c=inference.sample_prob_k < 0.5,
                                             color="blue")

    bb_ideal = draw_bounding_boxes(bounding_box=inference.sample_bb_ideal_k,
                                   width=imgs_in.shape[-2],
                                   height=imgs_in.shape[-1],
                                   c=torch.ones_like(inference.sample_prob_k).bool(),
                                   color="green")

    # Draw inferred and almost_inferred bb on top of the input image.
    # This is helpful to debug the recognition network
    bb_all = bb_inferred + bb_almost_inferred
    mask_no_bb_all = (torch.sum(bb_all, dim=-3, keepdim=True) == 0)
    imgs_in_with_all_bb = mask_no_bb_all * imgs_in + ~mask_no_bb_all * bb_all

    # Draw inferred and ideal bb on top of reconstructed image
    rec_imgs_no_bb = (inference.mixing_k1wh * inference.foreground_kcwh).sum(dim=-4)  # sum over boxes
    fg_mask = inference.mixing_k1wh.sum(dim=-4)  # sum over boxes
    bg_mask = torch.ones_like(fg_mask) - fg_mask

    background = bg_mask * inference.background_cwh if draw_bg else torch.zeros_like(rec_imgs_no_bb)
    bb1 = bb_inferred if draw_boxes else torch.zeros_like(rec_imgs_no_bb)
    bb2 = bb_almost_inferred if draw_boxes else torch.zeros_like(rec_imgs_no_bb)
    bb3 = bb_ideal if draw_ideal_boxes else torch.zeros_like(rec_imgs_no_bb)
    bb = bb1 + bb2 + bb3
    mask_no_bb = (torch.sum(bb, dim=-3, keepdim=True) == 0)
    imgs_rec_with_bb = mask_no_bb * (rec_imgs_no_bb + background) + ~mask_no_bb * bb

    return imgs_rec_with_bb, imgs_in_with_all_bb


def draw_bounding_boxes(bounding_box: BB,
                        width: int,
                        height: int,
                        c: torch.Tensor,
                        color: str = 'red') -> torch.Tensor:
    """
    Draw the bounding boxes.

    Args:
        bounding_box: BB of shape (*,K)
        width: width in pixel of the canvas
        height: height in pixel of the canvas
        c: boolean indicator of shape (*,K). Only the BB corrisponding to c=True are drawn
        color: color to use to draw the bounding box

    Returns:
        An image of size (*,3,:attr:`width`,:attr:`height`)

    Note:
        Works for any number of leading dimensions. Each leading dimension will be processed independently
    """
    assert c.dtype == torch.bool
    c_n, bx_n, by_n, bw_n, bh_n = broadcast_all(c,
                                                bounding_box.bx, bounding_box.by,
                                                bounding_box.bw, bounding_box.bh)

    # compute the coordinates of the bounding boxes and the probability of each box
    x1 = bx_n - 0.5 * bw_n
    x3 = bx_n + 0.5 * bw_n
    y1 = by_n - 0.5 * bh_n
    y3 = by_n + 0.5 * bh_n
    x1y1x3y3 = torch.stack((x1, y1, x3, y3), dim=-1)
    bxby = torch.stack((bx_n, by_n), dim=-1)

    # Reshape
    independet_dim = list(c_n.shape[:-1])
    canvas_numpy = torch.zeros(independet_dim + [height, width, 3]).flatten(end_dim=-4).numpy()  # shape (*,w,h,3)
    bxby = bxby.flatten(end_dim=-3)          # (b,k,2) -> (*,k,2)
    x1y1x3y3 = x1y1x3y3.flatten(end_dim=-3)  # (b,k,4) -> (*,k,2)
    c_n = c_n.flatten(end_dim=-2)            # (b,k)   -> (*,k)

    # draw the bounding boxes
    for n in range(c_n.shape[0]):
        # Draw on PIL
        img = PIL.Image.new(mode='RGB', size=(width, height), color=0)  # black canvas
        draw = PIL.ImageDraw.Draw(img)
        for k in range(c_n.shape[1]):
            if c_n[n, k]:
                draw.rectangle(x1y1x3y3[n, k, :].cpu().numpy(), outline=color, fill=None)
                draw.point(bxby[n, k, :].cpu().numpy(), fill=color)
        canvas_numpy[n, ...] = numpy.array(img.getdata(), numpy.uint8).reshape((height, width, 3))

    # Transform np to torch, rescale from [0,255] to (0,1)
    tmp = torch.from_numpy(canvas_numpy).permute(0, 3, 2, 1).float() / 255  # permute(0,3,2,1) is CORRECT
    return tmp.view(independet_dim + [3, width, height]).to(bounding_box.bx.device)





##########################
# FROM HERE
###########################




def movie_from_resolution_sweep(suggestion: Suggestion,
                                image: torch.Tensor,
                                contour_thickness: int = 2,
                                figsize: tuple = (8, 4)):
    assert torch.is_tensor(image)
    if len(image.shape) == 2:
        image = image.cpu().numpy()
    elif len(image.shape) == 3:
        image = image.permute(1, 2, 0).cpu().numpy()  # w,h, channel
    else:
        raise Exception("shape of image is not recognized")

    labels = suggestion.sweep_seg_mask[0].cpu().numpy()
    assert labels.shape[:2] == image.shape[:2]
    img_c = draw_contours_from_labels(image=image,
                                      labels=labels,
                                      contour_thickness=contour_thickness,
                                      contours_color='red')

    fig, ax = plt.subplots(ncols=3, figsize=figsize)
    ax_raw_image = ax[0]
    ax_contours = ax[1]
    ax_int_map = ax[2]

    ax_raw_image.imshow(image, cmap='gray')
    ax_raw_image.axis('off')
    ax_raw_image.set_title("raw image")

    ax_contours.imshow(img_c)
    ax_contours.axis('off')
    ax_contours.set_title("xxxxxx")

    ax_int_map.imshow(skimage.color.label2rgb(labels, image, bg_label=0, alpha=0.25))
    ax_int_map.axis('off')
    ax_int_map.set_title("xxxxx")
    plt.tight_layout()
    plt.close()

    def animate(i):
        labels = suggestion.sweep_seg_mask[i].cpu().numpy()
        img_c = draw_contours_from_labels(image=image,
                                          labels=labels,
                                          contour_thickness=contour_thickness,
                                          contours_color='red')

        title1 = 'frame={0:3d} res={1:.3f}'.format(i, suggestion.sweep_resolution[i])
        title2 = 'ncell={0:2d} iou={1:.3f}'.format(suggestion.sweep_n_cells[i], suggestion.sweep_iou[i])

        ax_contours.imshow(img_c)
        ax_contours.set_title(title1)

        ax_int_map.imshow(skimage.color.label2rgb(labels, image, bg_label=0, alpha=0.25))
        ax_int_map.set_title(title2)

    anim = animation.FuncAnimation(fig, animate, frames=suggestion.sweep_seg_mask.shape[0], interval=1000)

    return HTML(anim.to_html5_video())


def plot_grid(img,
              figsize: Optional[Tuple[float, float]] = None,
              experiment: Optional[neptune.run.Run] = None,
              neptune_name: Optional[str] = None):

    assert len(img.shape) == 3
    n_max = img.shape[-3]

    row_max = n_max // 4
    if row_max <= 1:
        fig, axes = plt.subplots(ncols=n_max, figsize=figsize)
        for n in range(n_max):
            axes[n].imshow(img[n])
    else:
        fig, axes = plt.subplots(ncols=4, nrows=row_max, figsize=figsize)
        for n in range(4 * row_max):
            row = n // 4
            col = n % 4
            axes[row, col].imshow(img[n])

    fig.tight_layout()
    if (neptune_name is not None) and (experiment is not None):
        # experiment[neptune_name].log(fig)
        # tmp_file_name = neptune_name.replace("/", "_")
        # fig.savefig(tmp_file_name + ".png")
        # experiment[neptune_name].log(File(tmp_file_name + ".png"))
        experiment[neptune_name].log(neptune.types.File.as_image(fig))
    plt.close(fig)
    return fig


def plot_img_and_seg(img: torch.Tensor,
                     seg: torch.Tensor,
                     figsize: Optional[Tuple[float, float]] = None,
                     experiment: Optional[neptune.run.Run] = None,
                     neptune_name: Optional[str] = None):

    assert len(img.shape) == len(seg.shape) == 4
    n_row = img.shape[-4]
    if n_row <= 1:
        fig, axes = plt.subplots(ncols=2, figsize=figsize)
        axes[0].imshow(img[0, 0], cmap='gray')
        axes[1].imshow(seg[0, 0], cmap='seismic', vmin=-0.5, vmax=10.5)
        axes[0].set_axis_off()
        axes[1].set_axis_off()

    else:
        fig, axes = plt.subplots(ncols=2, nrows=n_row, figsize=figsize)
        for n in range(n_row):
            axes[n, 0].imshow(img[n, 0], cmap='gray')
            axes[n, 1].imshow(seg[n, 0], cmap='seismic', vmin=-0.5, vmax=10.5)
            axes[n, 0].set_axis_off()
            axes[n, 1].set_axis_off()

    fig.tight_layout()
    if (neptune_name is not None) and (experiment is not None):
        experiment[neptune_name].log(neptune.types.File.as_image(fig))
    plt.close(fig)
    return fig


def plot_tiling(tiling,
                figsize: tuple = (12, 12),
                window: Optional[tuple] = None,
                experiment: Optional[neptune.run.Run] = None,
                neptune_name: Optional[str] = None):

    if window is None:
        window = [0, 0, tiling.integer_mask.shape[-2], tiling.integer_mask.shape[-1]]
    else:
        window = (max(0, window[0]),
                  max(0, window[1]),
                  min(tiling.integer_mask.shape[-2], window[2]),
                  min(tiling.integer_mask.shape[-1], window[3]))

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=figsize)
    axes[0, 0].imshow(skimage.color.label2rgb(label=tiling.integer_mask[0, 0, window[0]:window[2], 
                                                                        window[1]:window[3]].cpu().numpy(),
                                              image=numpy.zeros_like(tiling.integer_mask[0, 0, window[0]:window[2], 
                                                                                         window[1]:window[3]].cpu().numpy()),
                                              alpha=1.0,
                                              bg_label=0))
    
    axes[0, 1].imshow(skimage.color.label2rgb(label=tiling.integer_mask[0, 0, window[0]:window[2], 
                                                                        window[1]:window[3]].cpu().numpy(),
                                              image=tiling.raw_image[0, 0, window[0]:window[2], 
                                                                     window[1]:window[3]].cpu().numpy(),
                                              alpha=0.25,
                                              bg_label=0))
    axes[1, 0].imshow(tiling.fg_prob[0, 0, window[0]:window[2], 
                                     window[1]:window[3]].cpu().numpy(), cmap='gray')
    axes[1, 1].imshow(tiling.raw_image[0, :, window[0]:window[2], 
                                       window[1]:window[3]].cpu().permute(1, 2, 0).squeeze(-1).numpy(), cmap='gray')

    axes[0, 0].set_title("sample integer mask")
    axes[0, 1].set_title("sample integer mask")
    axes[1, 0].set_title("fg prob")
    axes[1, 1].set_title("raw image")
    fig.tight_layout()

    if (neptune_name is not None) and (experiment is not None):
        # experiment[neptune_name].log(fig)
        # fig.savefig(neptune_name + ".png")
        # experiment[neptune_name].log(File(neptune_name + ".png"))
        experiment[neptune_name].log(neptune.types.File.as_image(fig))
    plt.close(fig)
    return fig


def plot_concordance(concordance,
                     figsize: tuple = (12, 12),
                     experiment: Optional[neptune.run.Run] = None,
                     neptune_name: Optional[str] = None):
    fig, axes = plt.subplots(figsize=figsize)
    axes.imshow(concordance.intersection_mask.cpu(), cmap='gray')
    axes.set_title("intersection mask, iou=" + str(concordance.iou))

    fig.tight_layout()
    if (neptune_name is not None) and (experiment is not None):
        # experiment[neptune_name].log(fig)
        # tmp_file_name = neptune_name.replace("/", "_")
        # fig.savefig(tmp_file_name + ".png")
        # experiment[neptune_name].log(File(tmp_file_name + ".png"))
        experiment[neptune_name].log(neptune.types.File.as_image(fig))

    plt.close(fig)
    return fig


# ---------------------- show_batch and derivative


def show_batch(images: torch.Tensor,
               n_col: int = 4,
               n_padding: int = 10,
               n_mc_samples: int = 1,
               title: Optional[str] = None,
               pad_value: float = 1,
               normalize: bool = False,
               normalize_range: Optional[tuple] = None,
               figsize: Optional[Tuple[float, float]] = None,
               experiment: Optional[neptune.run.Run] = None,
               neptune_name: Optional[str] = None):
    """
    Visualize a torch tensor of shape: (*,  ch, width, height)
    It works for any number of leading dimensions
    """
    assert len(images.shape) >= 4  # *, ch, width, height

    if len(images.shape) > 4:
        images = images[:n_mc_samples].flatten(end_dim=-4)
    else:
        images = images.flatten(end_dim=-4)  # -1, ch, width, height

    if images.device != "cpu":
        images = images.cpu()

    # Always normalize the image in (0,1) either using min_max of tensor or normalize_range
    grid = utils.make_grid(images, n_col, n_padding, normalize=normalize, value_range=normalize_range,
                           scale_each=False, pad_value=pad_value)

    fig = plt.figure(figsize=figsize)
    plt.imshow(grid.detach().permute(1, 2, 0).squeeze(-1).numpy())
    # plt.axis("off")
    if isinstance(title, str):
        plt.title(title)
    fig.tight_layout()

    if (neptune_name is not None) and (experiment is not None):
        # experiment[neptune_name].log(fig)
        #tmp_file_name = neptune_name.replace("/", "_")
        #fig.savefig(tmp_file_name + ".png")
        #experiment[neptune_name].log(File(tmp_file_name + ".png"))
        experiment[neptune_name].log(neptune.types.File.as_image(fig))

    plt.close(fig)
    return fig


def plot_generation(output: Output,
                    epoch: int,
                    prefix: str = "",
                    experiment: Optional[neptune.run.Run] = None,
                    verbose: bool = False):

    if verbose:
        print("in plot_reconstruction_and_inference")

    _ = show_batch(output.imgs.clamp(min=0.0, max=1.0),
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='imgs, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/imgs")
    _ = show_batch(output.inference.sample_c_grid_before_nms.float(),
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='c_grid_before_nms, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/c_grid_before_nms")
    _ = show_batch(output.inference.sample_c_grid_after_nms.float(),
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='c_grid_after_nms, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/c_grid_after_nms")
    _ = show_batch(output.inference.foreground_kcwh.sum(dim=-4).clamp(min=0.0, max=1.0),
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='foreground, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/fg")
    _ = show_batch(output.inference.background_cwh.clamp(min=0.0, max=1.0),
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='background, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/bg")

    mixing_fg_b1wh = output.inference.mixing_k1wh.sum(dim=-4).clamp(min=0.0, max=1.0)
    _ = show_batch(mixing_fg_b1wh,
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='fg_mask, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/fg_mask")

    mixing_bg_b1wh = torch.ones_like(mixing_fg_b1wh) - mixing_fg_b1wh
    _ = show_batch(mixing_bg_b1wh,
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='bg_mask, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/bg_mask")

    _ = show_batch(output.bb_imgs,
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='bounding_box_selection, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/bb_selection")

    if verbose:
        print("leaving plot_generation")


def plot_reconstruction_and_inference(output: Output,
                                      epoch: int,
                                      prefix: str = "",
                                      experiment: Optional[neptune.run.Run] = None,
                                      verbose: bool = False):
    if verbose:
        print("in plot_reconstruction_and_inference")

    NMAX=10

    _ = show_batch(output.imgs[:NMAX].clamp(min=0.0, max=1.0),
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='imgs, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/imgs")

    _ = show_batch(output.inference.sample_c_grid_before_nms[:NMAX].float(),
                   n_col=5,
                   n_padding=4,
                   n_mc_samples=2,
                   normalize=False,
                   title='c_grid_before_nms, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/c_grid_before_nms")

    _ = show_batch(output.inference.sample_c_grid_after_nms[:NMAX].float(),
                   n_col=5,
                   n_padding=4,
                   n_mc_samples=2,
                   normalize=False,
                   title='c_grid_after_nms, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/c_grid_after_nms")

    _ = show_batch(output.inference.logit_grid[:NMAX],
                   n_col=5,
                   n_padding=4,
                   normalize=True,
                   normalize_range=(-3, 3),
                   title='logit_unet, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/logit_unet")

    _ = show_batch(torch.sigmoid(output.inference.logit_grid[:NMAX]),
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='prob_unet, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/prob_unet")

    _ = show_batch(output.inference.prob_from_ranking_grid[:NMAX],
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='prob_ranking, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/prob_ranking")

    _ = show_batch(output.inference.foreground_kcwh[:NMAX].sum(dim=-4).clamp(min=0.0, max=1.0),
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='foreground, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/fg")

    _ = show_batch(output.inference.background_cwh[:NMAX].clamp(min=0.0, max=1.0),
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='background, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/bg")

    _ = show_batch(output.inference.mask_overlap_1wh[:NMAX],
                   n_col=5,
                   n_padding=4,
                   normalize=True,
                   normalize_range=(0.0, 2.0),
                   title='overlap, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/overlap")

    mixing_fg_b1wh = output.inference.mixing_k1wh[:NMAX].sum(dim=-4).clamp(min=0.0, max=1.0)
    _ = show_batch(mixing_fg_b1wh,
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='fg_mask, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/fg_mask")

    _ = show_batch((torch.ones_like(mixing_fg_b1wh) - mixing_fg_b1wh)[:NMAX],
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='bg_mask, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/bg_mask")

    _ = show_batch(output.bb_imgs[:NMAX],
                   n_col=5,
                   n_padding=4,
                   normalize=False,
                   title='bounding_box_selection, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/bb_selection")

    # print(output.inference.feature_map.shape) --> batch_size, ch, w, h
    _ = show_batch(output.inference.feature_map[0].unsqueeze(-3),
                   n_col=5,
                   n_padding=4,
                   normalize=True,
                   title='feature map, epoch= {0:6d}'.format(epoch),
                   experiment=experiment,
                   neptune_name=prefix+"/f_map")

    # IoU histo
    c_bk = output.inference.sample_c_k
    iou_bk = output.inference.iou_boxes_k
    iou_on = iou_bk[c_bk]
    iou_off = iou_bk[~c_bk]
    fig = plt.figure(figsize=(12,12))
    plt.hist(iou_on.cpu().detach().numpy(), density=False, bins=20, label="iou_on", alpha=0.3, rwidth=0.8)
    plt.hist(iou_off.cpu().detach().numpy(), density=False, bins=20, label='iou_off', alpha=0.3, rwidth=0.8)
    plt.title("IoU histogram, epoch="+str(epoch))
    plt.legend()
    fig.tight_layout()
    if (prefix is not None) and (experiment is not None):
        experiment[prefix+"/iou_hist"].log(neptune.types.File.as_image(fig))
    plt.close(fig)

    # KL_where
    c_bk = output.inference.sample_c_k
    kl_box_bk = output.inference.kl_where_k
    kl_box_on = kl_box_bk[c_bk]
    kl_box_off = kl_box_bk[~c_bk]
    fig = plt.figure(figsize=(12, 12))
    plt.hist(kl_box_on.cpu().detach().numpy(), density=False, bins=20, label="kl_box_on", alpha=0.3, rwidth=0.8)
    plt.hist(kl_box_off.cpu().detach().numpy(), density=False, bins=20, label='kl_box_off', alpha=0.3, rwidth=0.8)
    plt.title("KL box histogram, epoch=" + str(epoch))
    plt.legend()
    fig.tight_layout()
    if (prefix is not None) and (experiment is not None):
        experiment[prefix + "/kl_box_hist"].log(neptune.types.File.as_image(fig))
    plt.close(fig)

    # KL_instance
    c_bk = output.inference.sample_c_k
    kl_fg_bk = output.inference.kl_instance_k
    kl_fg_on = kl_fg_bk[c_bk]
    kl_fg_off = kl_fg_bk[~c_bk]
    fig = plt.figure(figsize=(12, 12))
    plt.hist(kl_fg_on.cpu().detach().numpy(), density=False, bins=20, label="kl_fg_on", alpha=0.3, rwidth=0.8)
    plt.hist(kl_fg_off.cpu().detach().numpy(), density=False, bins=20, label='kl_bg_off', alpha=0.3, rwidth=0.8)
    plt.title("KL fg histogram, epoch=" + str(epoch))
    plt.legend()
    fig.tight_layout()
    if (prefix is not None) and (experiment is not None):
        experiment[prefix + "/kl_fg_hist"].log(neptune.types.File.as_image(fig))
    plt.close(fig)

    # KL_bg
    kl_bg = output.inference.kl_bg
    fig = plt.figure(figsize=(12, 12))
    plt.hist(kl_bg.cpu().detach().numpy(), density=False, bins=20, label="kl_bg", rwidth=0.8)
    plt.title("KL bg histogram, epoch=" + str(epoch))
    plt.legend()
    fig.tight_layout()
    if (prefix is not None) and (experiment is not None):
        experiment[prefix + "/kl_bg_hist"].log(neptune.types.File.as_image(fig))
    plt.close(fig)

    # KL_dpp
    kl_dpp = output.inference.kl_dpp
    fig = plt.figure(figsize=(12, 12))
    plt.hist(kl_dpp.cpu().detach().numpy(), density=False, bins=20, label="kl_dpp", rwidth=0.8)
    plt.title("KL dpp histogram, epoch=" + str(epoch))
    plt.legend()
    fig.tight_layout()
    if (prefix is not None) and (experiment is not None):
        experiment[prefix + "/kl_dpp_hist"].log(neptune.types.File.as_image(fig))
    plt.close(fig)

    if verbose:
        print("leaving plot_reconstruction_and_inference")


def plot_segmentation(segmentation: Segmentation,
                      epoch: Union[int, str] = "",
                      prefix: str = "",
                      experiment: Optional[neptune.run.Run] = None,
                      verbose: bool = False) -> tuple:
    if verbose:
        print("in plot_segmentation")

    if isinstance(epoch, int):
        title_postfix = 'epoch= {0:6d}'.format(epoch)
    elif isinstance(epoch, str):
        title_postfix = epoch
    else:
        raise Exception

    fig_a = show_batch(segmentation.integer_mask.float(),
                       n_padding=4,
                       normalize=True,
                       normalize_range=None,  # use min max of tensor to scale integer_mask into (0,1)
                       figsize=(12, 12),
                       title='integer_mask, '+title_postfix,
                       experiment=experiment,
                       neptune_name=prefix+"/integer_mask")
    fig_b = show_batch(segmentation.fg_prob.clamp(min=0.0, max=1.0),
                       n_padding=4,
                       normalize=False,
                       figsize=(12, 12),
                       title='fg_prob, '+title_postfix,
                       experiment=experiment,
                       neptune_name=prefix+"/fg_prob")

    if verbose:
        print("leaving plot_segmentation")

    return fig_a, fig_b


