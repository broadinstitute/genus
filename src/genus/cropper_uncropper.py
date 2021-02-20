import torch
import torch.nn.functional as F
from .namedtuple import BB


class Cropper(object):
    """ Use STN to crop out a patch of the original images according to bounding_box.
        It uses REFLECTION padding """
    
    def __init__(self):
        super().__init__()

    @staticmethod
    def crop(bounding_box: BB, big_stuff: torch.Tensor, width_small: int, height_small: int) -> torch.Tensor:
        """ Document """

        # Prepare the shapes
        large_dependent_dim = list(big_stuff.shape[-3:])  # ch, width, height
        independent_dim = list(big_stuff.shape[:-3])  # this extract mc_samples, batch, k_boxes
        ch, width_raw, height_raw = big_stuff.shape[-3:]
        small_dependent_dim = [ch, width_small, height_small]
        assert independent_dim == list(bounding_box.bx.shape)

        # Compute the affine matrix
        affine: torch.Tensor = Cropper._compute_affine_cropper(bounding_box=bounding_box,
                                                               width_raw=width_raw,
                                                               height_raw=height_raw)

        # The cropped and uncropped imgs have:
        # a. same independent dimension (boxes, batch)
        # b. same channels
        # c. different width and height
        # I use reshape instead of using reshape and contigous().reshape([-1] + large_dependent_dims)
        big_stuff = big_stuff.reshape([-1] + large_dependent_dim)
        cropped_images = torch.zeros(independent_dim + small_dependent_dim,
                                     dtype=big_stuff.dtype,
                                     device=big_stuff.device).reshape([-1] + small_dependent_dim)

        grid = F.affine_grid(affine, list(cropped_images.shape), align_corners=True)
        cropped_images = F.grid_sample(big_stuff, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
        return cropped_images.reshape(independent_dim + small_dependent_dim)

    @staticmethod
    def _compute_affine_cropper(bounding_box: BB, width_raw: int, height_raw: int) -> torch.Tensor:
        """ Source is UNCROPPED (large) image
            Target is CROPPED (small) image.
            
            The equations are:
            | x_s |   | sx  0   kx | | x_t |   | sx  0  | | x_t |   | kx |
            |     | = |            | | y_t | = | 0   sy | | y_t | + | ky |
            | y_s |   | 0   sy  ky | | 1   |     
            We can evaluate the expression above at:
            a. target (0,0) <==> source (-1+2*bx_dimfull/WIDTH,-1+2*by_dimfull/HEIGHT)
            b. target (1,1) <==> source (-1+2*(bx_dimfull+0.5*bw_dimfull)/WIDTH,-1+2*(by_dimfull+0.5*bh_dimfull)/HEIGHT)
        
            This leads to:
            a. kx = -1+2*bx_dimfull/WIDTH
            b. ky = -1+2*by_dimfull/HEIGHT
            c. sx = bw_dimfull/WIDTH
            d. sy = bh_dimfull/HEIGHT
        """ 
        kx = (-1.0 + 2 * bounding_box.bx / width_raw).reshape(-1, 1)
        ky = (-1.0 + 2 * bounding_box.by / height_raw).reshape(-1, 1)
        sx = (bounding_box.bw / width_raw).reshape(-1, 1)
        sy = (bounding_box.bh / height_raw).reshape(-1, 1)
        zero = torch.zeros_like(kx)

        return torch.cat((sy, zero, ky, zero, sx, kx), dim=-1).reshape(-1, 2, 3)

    
class Uncropper(object):
    """ Use STN to uncrop (i.e. paste) an image according to bounding_box. It pastes on a zero background. """

    def __init__(self):
        super().__init__()

    @staticmethod
    def uncrop(bounding_box: BB, small_stuff: torch.Tensor, width_big: int, height_big: int) -> torch.Tensor:
        """ Document """

        # Check and prepare the sizes
        ch = small_stuff.shape[-3]  # this is the channels
        independent_dim = list(small_stuff.shape[:-3])  # this includes: mc_samples, batch, k_boxes
        small_dependent_dim = list(small_stuff.shape[-3:])  # this includes: ch, small_width, small_height
        large_dependent_dim = [ch, width_big, height_big]  # these are the dependent dimensions
        assert independent_dim == list(bounding_box.bx.shape)

        # Compute the affine matrix. Note that z_where has only independent dimensions
        affine_matrix = Uncropper._compute_affine_uncropper(bounding_box=bounding_box,
                                                            width_raw=width_big,
                                                            height_raw=height_big).reshape(independent_dim+[2, 3])

        # The cropped and uncropped imgs have:
        # a. same independent dimension (boxes, batch)
        # b. same channels
        # c. different width and height
        uncropped_stuff = torch.zeros(independent_dim + large_dependent_dim,
                                      dtype=small_stuff.dtype,
                                      device=small_stuff.device).reshape([-1] + large_dependent_dim)
        grid = F.affine_grid(affine_matrix.reshape(-1, 2, 3), list(uncropped_stuff.shape), align_corners=True)
        uncropped_stuff = F.grid_sample(small_stuff.reshape([-1] + small_dependent_dim), grid,
                                        mode='bilinear', padding_mode='zeros', align_corners=True)
        return uncropped_stuff.reshape(independent_dim + large_dependent_dim)

    @staticmethod
    def _compute_affine_uncropper(bounding_box: BB, width_raw: int, height_raw: int) -> torch.Tensor:
        """ Source is CROPPED (small) image
            Target is UNCROPPED (large) image.
            
            The equations are:
            | x_s |   | sx  0   kx | | x_t |   | sx  0  | | x_t |   | kx |
            |     | = |            | | y_t | = | 0   sy | | y_t | + | ky |
            | y_s |   | 0   sy  ky | | 1   |     
            We can evaluate the expression above at:
            a. source (0,0) <==> target (-1+2*bx_dimfull/WIDTH,-1+2*by_dimfull/HEIGHT)
            b. source (1,1) <==> target (-1+2*(bx_dimfull+0.5*bw_dimfull)/WIDTH,-1+2*(by_dimfull+0.5*bh_dimfull)/HEIGHT)
        
            This leads to:
            a. kx = (WIDTH-2*bx_dimfull)/bw_dimfull
            b. ky = (WIDTH-2*bx_dimfull)/bh_dimfull
            c. sx = WIDTH/bw_dimfull
            d. sy = HEIGHT/bh_dimfull
        """
        width_raw_tensor = width_raw * torch.ones_like(bounding_box.bx)
        height_raw_tensor = height_raw * torch.ones_like(bounding_box.bx)
        kx = ((width_raw_tensor - 2.0 * bounding_box.bx) / bounding_box.bw).reshape(-1, 1)
        ky = ((height_raw_tensor - 2.0 * bounding_box.by) / bounding_box.bh).reshape(-1, 1)
        sx = (width_raw_tensor / bounding_box.bw).reshape(-1, 1)
        sy = (height_raw_tensor / bounding_box.bh).reshape(-1, 1)
        zero = torch.zeros_like(kx)

        return torch.cat((sy, zero, ky, zero, sx, kx), dim=-1).reshape(-1, 2, 3)
