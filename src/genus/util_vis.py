import PIL.Image
import PIL.ImageDraw
import torch
import numpy
import neptune
import skimage.color
import skimage.morphology
from typing import Tuple, Optional, Union
from torchvision import utils
from matplotlib import pyplot as plt
from IPython.display import HTML
from matplotlib import animation
from .namedtuple import BB, Output, Segmentation, Suggestion, Inference
from .util_logging import log_img_only

""" This module contains specialized plotting functions to visualize the segmentation results 
and the training process. In most cases if :attr:`experiment` (of :class:`neptune.experiments.Experiment`) and 
:attr:`neptune_name` of class str are specified then the corresponding image is logged into neptune """


#  Functions whose documentation should NOT be exposed

def contours_from_labels(labels: numpy.ndarray,
                         contour_thickness: int = 1) -> numpy.ndarray:
    assert isinstance(labels, numpy.ndarray)
    assert len(labels.shape) == 2
    assert contour_thickness >= 1
    contours = (skimage.morphology.dilation(labels) != labels)

    for i in range(1, contour_thickness):
        contours = skimage.morphology.binary_dilation(contours)
    return contours


def draw_contours(image: numpy.ndarray, contours: numpy.ndarray, contours_color: str = 'red') -> numpy.ndarray:
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
                              labels: numpy.ndarray,
                              contour_thickness: int = 1,
                              contours_color: str = 'red') -> numpy.ndarray:
    """ Document """
    contours = contours_from_labels(labels, contour_thickness)
    image_with_contours = draw_contours(image, contours, contours_color)
    return image_with_contours


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


def plot_label_contours(label: Union[torch.Tensor, numpy.ndarray],
                        image: Union[torch.Tensor, numpy.ndarray],
                        window: Optional[tuple] = None,
                        contour_thickness: int = 2,
                        contour_color: str = 'red',
                        figsize: tuple = (24, 24),
                        experiment: Optional[neptune.experiments.Experiment] = None,
                        neptune_name: Optional[str] = None):
    assert len(label.shape) == 2
    assert len(image.shape) == 2 or len(image.shape)==3
    
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

    print(window)
    if window is None:
        window = [0, 0, label.shape[-2], label.shape[-1]]
    else:
        window = (max(0, window[0]),
                  max(0, window[1]),
                  min(label.shape[-2], window[2]),
                  min(label.shape[-1], window[3]))

    contours = contours_from_labels(label[window[0]:window[2], window[1]:window[3]], contour_thickness)
    img_with_contours = draw_contours(image=image[window[0]:window[2], window[1]:window[3]],
                                      contours=contours,
                                      contours_color=contour_color)

    fig, ax = plt.subplots(ncols=3, figsize=figsize)
    ax[0].imshow(image[window[0]:window[2], window[1]:window[3]], cmap='gray')
    ax[1].imshow(img_with_contours)
    ax[2].imshow(skimage.color.label2rgb(label=label[window[0]:window[2], window[1]:window[3]],
                                         image=image[window[0]:window[2], window[1]:window[3]],
                                         alpha=0.25, 
                                         bg_label=0))

    fig.tight_layout()
    if (neptune_name is not None) and (experiment is not None):
        # log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
        log_img_only(name=neptune_name, fig=fig, experiment=experiment)
    plt.close(fig)
    return fig


def draw_img(inference: Inference,
             draw_bg: bool,
             draw_boxes: bool,
             draw_ideal_boxes: bool):

    rec_imgs_no_bb = (inference.mixing_kb1wh * inference.foreground_kbcwh).sum(dim=-5)  # sum over boxes
    fg_mask = inference.mixing_kb1wh.sum(dim=-5)  # sum over boxes
    background = (1 - fg_mask) * inference.background_bcwh if draw_bg else torch.zeros_like(rec_imgs_no_bb)

    bb = draw_bounding_boxes(bounding_box_kb=inference.sample_bb_kb,
                             width=rec_imgs_no_bb.shape[-2],
                             height=rec_imgs_no_bb.shape[-1],
                             c_kb=inference.sample_c_kb,
                             color="red") if draw_boxes else torch.zeros_like(fg_mask)

    bb_ideal = draw_bounding_boxes(bounding_box_kb=inference.sample_bb_ideal_kb,
                                   width=rec_imgs_no_bb.shape[-2],
                                   height=rec_imgs_no_bb.shape[-1],
                                   c_kb=inference.sample_c_kb,
                                   color="blue") if draw_ideal_boxes else torch.zeros_like(fg_mask)

    mask_no_bb = (torch.sum(bb + bb_ideal, dim=-3, keepdim=True) == 0)

    return mask_no_bb * (rec_imgs_no_bb + background) + ~mask_no_bb * (bb + bb_ideal)


def draw_bounding_boxes(bounding_box_kb: BB,
                        width: int,
                        height: int,
                        c_kb: torch.Tensor,
                        color: str = 'red') -> torch.Tensor:

    # prepare the storage
    assert bounding_box_kb.bx.shape == c_kb.shape
    assert len(bounding_box_kb.bx.shape) == 2
    n_boxes, batch_size = bounding_box_kb.bx.shape
    batch_bb_numpy = numpy.zeros((batch_size, width, height, 3))  # black canvas

    # compute the coordinates of the bounding boxes and the probability of each box
    x1 = bounding_box_kb.bx - 0.5 * bounding_box_kb.bw
    x3 = bounding_box_kb.bx + 0.5 * bounding_box_kb.bw
    y1 = bounding_box_kb.by - 0.5 * bounding_box_kb.bh
    y3 = bounding_box_kb.by + 0.5 * bounding_box_kb.bh
    x1y1x3y3 = torch.stack((x1, y1, x3, y3), dim=-1)

    # draw the bounding boxes
    for batch in range(batch_size):

        # Draw on PIL
        img = PIL.Image.new(mode='RGB', size=(width, height), color=0)  # black canvas
        draw = PIL.ImageDraw.Draw(img)
        for box in range(n_boxes):
            if c_kb[box, batch] > 0.5:
                draw.rectangle(x1y1x3y3[box, batch, :].cpu().numpy(), outline=color, fill=None)
        batch_bb_numpy[batch, ...] = numpy.array(img.getdata(), numpy.uint8).reshape((width, height, 3))

    # Transform np to torch, rescale from [0,255] to (0,1)
    batch_bb_torch = torch.from_numpy(batch_bb_numpy).permute(0, 3, 2, 1).float() / 255  # permute(0,3,2,1) is CORRECT
    return batch_bb_torch.to(bounding_box_kb.bx.device)


def plot_grid(img,
              figsize: Optional[Tuple[float, float]] = None,
              experiment: Optional[neptune.experiments.Experiment] = None,
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
        # log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
        log_img_only(name=neptune_name, fig=fig, experiment=experiment)
    plt.close(fig)
    return fig


def plot_img_and_seg(img: torch.Tensor,
                     seg: torch.Tensor,
                     figsize: Optional[Tuple[float, float]] = None,
                     experiment: Optional[neptune.experiments.Experiment] = None,
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
        # log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
        log_img_only(name=neptune_name, fig=fig, experiment=experiment)
    plt.close(fig)
    return fig


def show_batch(images: torch.Tensor,
               n_col: int = 4,
               n_padding: int = 10,
               title: Optional[str] = None,
               pad_value: int = 1,
               normalize_range: Optional[tuple] = None,
               figsize: Optional[Tuple[float, float]] = None,
               experiment: Optional[neptune.experiments.Experiment] = None,
               neptune_name: Optional[str] = None):

    """Visualize a torch tensor of shape: (batch x ch x width x height) """
    assert len(images.shape) == 4  # batch, ch, width, height
    if images.device != "cpu":
        images = images.cpu()

    # Always normalize the image in (0,1) either using min_max of tensor or normalize_range
    grid = utils.make_grid(images, n_col, n_padding, normalize=True, range=normalize_range,
                           scale_each=False, pad_value=pad_value)
        
    fig = plt.figure(figsize=figsize)
    plt.imshow(grid.detach().permute(1, 2, 0).squeeze(-1).numpy())
    # plt.axis("off")
    if isinstance(title, str):
        plt.title(title)
    fig.tight_layout()

    if (neptune_name is not None) and (experiment is not None):
        # log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
        log_img_only(name=neptune_name, fig=fig, experiment=experiment)

    plt.close(fig)
    return fig


def plot_tiling(tiling,
                figsize: tuple = (12, 12),
                window: Optional[tuple] = None,
                experiment: Optional[neptune.experiments.Experiment] = None,
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
        # log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
        log_img_only(name=neptune_name, fig=fig, experiment=experiment)
    plt.close(fig)
    return fig


def plot_generation(output: Output,
                    epoch: int,
                    prefix: str = "",
                    postfix: str = "",
                    experiment: Optional[neptune.experiments.Experiment] = None,
                    verbose: bool = False):

    if verbose:
        print("in plot_reconstruction_and_inference")

    fig_a = show_batch(output.imgs,
                       n_col=4,
                       n_padding=4,
                       normalize_range=(0.0, 1.0),
                       title='imgs, epoch= {0:6d}'.format(epoch),
                       experiment=experiment,
                       neptune_name=prefix + "imgs" + postfix)
    fig_b = show_batch(output.inference.sample_c_grid_before_nms.float(),
                       n_col=4,
                       n_padding=4,
                       normalize_range=(0.0, 1.0),
                       title='c_grid_before_nms, epoch= {0:6d}'.format(epoch),
                       experiment=experiment,
                       neptune_name=prefix + "c_grid_before_nms" + postfix)
    fig_c = show_batch(output.inference.sample_c_grid_after_nms.float(),
                       n_col=4,
                       n_padding=4,
                       normalize_range=(0.0, 1.0),
                       title='c_grid_after_nms, epoch= {0:6d}'.format(epoch),
                       experiment=experiment,
                       neptune_name=prefix + "c_grid_after_nms" + postfix)
    fig_d = show_batch(output.inference.background_bcwh,
                       n_col=4,
                       n_padding=4,
                       normalize_range=(0.0, 1.0),
                       title='background, epoch= {0:6d}'.format(epoch),
                       experiment=experiment,
                       neptune_name=prefix + "bg" + postfix)

    if verbose:
        print("leaving plot_generation")

    return fig_a, fig_b, fig_c, fig_d


def plot_reconstruction_and_inference(output: Output,
                                      epoch: int,
                                      prefix: str = "",
                                      postfix: str = "",
                                      experiment: Optional[neptune.experiments.Experiment] = None,
                                      verbose: bool = False):
    if verbose:
        print("in plot_reconstruction_and_inference")

    fig_a = show_batch(output.imgs,
                       n_col=4,
                       n_padding=4,
                       normalize_range=(0.0, 1.0),
                       title='imgs, epoch= {0:6d}'.format(epoch),
                       experiment=experiment,
                       neptune_name=prefix+"imgs"+postfix)
    fig_b = show_batch(output.inference.sample_c_grid_before_nms.float(),
                       n_col=4,
                       n_padding=4,
                       normalize_range=(0.0, 1.0),
                       title='c_grid_before_nms, epoch= {0:6d}'.format(epoch),
                       experiment=experiment,
                       neptune_name=prefix+"c_grid_before_nms"+postfix)
    fig_c = show_batch(output.inference.sample_c_grid_after_nms.float(),
                       n_col=4,
                       n_padding=4,
                       normalize_range=(0.0, 1.0),
                       title='c_grid_after_nms, epoch= {0:6d}'.format(epoch),
                       experiment=experiment,
                       neptune_name=prefix+"c_grid_after_nms"+postfix)
    fig_d = show_batch(torch.sigmoid(output.inference.logit_grid_unet),
                       n_col=4,
                       n_padding=4,
                       normalize_range=(0.0, 1.0),
                       title='logit_grid, epoch= {0:6d}'.format(epoch),
                       experiment=experiment,
                       neptune_name=prefix+"prob_unet_grid"+postfix)
    fig_e = show_batch(torch.sigmoid(output.inference.logit_grid),
                       n_col=4,
                       n_padding=4,
                       normalize_range=(0.0, 1.0),
                       title='prob_grid, epoch= {0:6d}'.format(epoch),
                       experiment=experiment,
                       neptune_name=prefix+"prob_grid"+postfix)
    fig_f = show_batch(output.inference.background_bcwh,
                       n_col=4,
                       n_padding=4,
                       normalize_range=(0.0, 1.0),
                       title='background, epoch= {0:6d}'.format(epoch),
                       experiment=experiment,
                       neptune_name=prefix+"bg"+postfix)
    if verbose:
        print("leaving plot_reconstruction_and_inference")

    return fig_a, fig_b, fig_c, fig_d, fig_e, fig_f


def plot_segmentation(segmentation: Segmentation,
                      epoch: Union[int, str] = "",
                      prefix: str = "",
                      postfix: str = "",
                      experiment: Optional[neptune.experiments.Experiment] = None,
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
                       normalize_range=None,  # use min_max of tensor
                       figsize=(12, 12),
                       title='integer_mask, '+title_postfix,
                       experiment=experiment,
                       neptune_name=prefix+"integer_mask"+postfix)
    fig_b = show_batch(segmentation.fg_prob,
                       n_padding=4,
                       normalize_range=(0.0, 1.0),
                       figsize=(12, 12),
                       title='fg_prob, '+title_postfix,
                       experiment=experiment,
                       neptune_name=prefix+"fg_prob"+postfix)

    if verbose:
        print("leaving plot_segmentation")

    return fig_a, fig_b


def plot_concordance(concordance,
                     figsize: tuple = (12, 12),
                     experiment: Optional[neptune.experiments.Experiment] = None,
                     neptune_name: Optional[str] = None):
    fig, axes = plt.subplots(figsize=figsize)
    axes.imshow(concordance.intersection_mask.cpu(), cmap='gray')
    axes.set_title("intersection mask, iou=" + str(concordance.iou))

    fig.tight_layout()
    if (neptune_name is not None) and (experiment is not None):
        log_img_only(name=neptune_name, fig=fig, experiment=experiment)
    plt.close(fig)
    return fig

