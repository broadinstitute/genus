import torch
from .conv import UnetUpBlock, UnetDownBlock, Mlp1by1, SameSpatialResolution
from collections import deque
from .namedtuple import UNEToutput
from typing import Optional


class UNet(torch.nn.Module):
    def __init__(self,
                 n_max_pool: int,
                 level_zwhere_and_logit_output: int,
                 level_background_output: int,
                 n_ch_output_features: int,
                 ch_after_preprocessing: int,
                 downsampling_factor_preprocessing: int,
                 dim_zbg: int,
                 dim_zwhere: int,
                 dim_logit: int,
                 ch_raw_image: int,
                 concatenate_raw_image_to_fmap: bool,
                 grad_logit_max: Optional[float] = None):
        super().__init__()

        # Parameters UNet
        self.n_max_pool = n_max_pool
        self.level_zwhere_and_logit_output = level_zwhere_and_logit_output
        self.level_background_output = level_background_output
        self.n_ch_output_features = n_ch_output_features
        self.ch_after_first_two_conv = ch_after_preprocessing
        self.dim_zbg = dim_zbg
        self.dim_zwhere = dim_zwhere
        self.dim_logit = dim_logit
        self.ch_raw_image = ch_raw_image
        self.concatenate_raw_image_to_fmap = concatenate_raw_image_to_fmap
        self.grad_logit_max = grad_logit_max
        self.downsampling_factor_preprocessing = downsampling_factor_preprocessing

        if self.downsampling_factor_preprocessing != 1:
            raise NotImplementedError("At the moment downsampling during preprocessing should be 1")

        # Initializations
        ch = self.ch_after_first_two_conv
        j = 1
        self.j_list = [j]
        self.ch_list = [ch]

        # Down path to center
        self.down_path = torch.nn.ModuleList([SameSpatialResolution(ch_in=self.ch_raw_image,
                                                                    ch_out=self.ch_list[-1],
                                                                    double_or_single="double",
                                                                    reflection_padding=True)])
        for i in range(0, self.n_max_pool):
            j = j * 2
            ch = ch * 2
            self.ch_list.append(ch)
            self.j_list.append(j)
            self.down_path.append(UnetDownBlock(ch_in=self.ch_list[-2], ch_out=self.ch_list[-1]))

        # Up path
        self.up_path = torch.nn.ModuleList()
        for i in range(0, self.n_max_pool):
            j = int(j // 2)
            ch = int(ch // 2)
            self.ch_list.append(ch)
            self.j_list.append(j)
            self.up_path.append(UnetUpBlock(ch_in=self.ch_list[-2], ch_out=self.ch_list[-1]))

        # Prediction maps
        # TODO: remove this concatenation
        #ch_out_fmap = self.n_ch_output_features - \
        #              self.ch_raw_image if self.concatenate_raw_image_to_fmap else self.n_ch_output_features
        #self.pred_features = Mlp1by1(ch_in=self.ch_list[-1],
        #                             ch_out=ch_out_fmap,
        #                             ch_hidden=-1)  # this means there is NO hidden layer
        # TODO: Remove this. I am cropping the raw image now
        ch_out_fmap = self.n_ch_output_features
        self.pred_features = Mlp1by1(ch_in=self.ch_raw_image,
                                     ch_out=ch_out_fmap,
                                     ch_hidden=-1)  # this means there is NO hidden layer

        self.ch_in_zwhere = self.ch_list[-self.level_zwhere_and_logit_output - 1]
        self.encode_zwhere = Mlp1by1(ch_in=self.ch_in_zwhere,
                                     ch_out=self.dim_zwhere,
                                     ch_hidden=(self.ch_in_zwhere + self.dim_zwhere)//2)

        self.ch_in_logit = self.ch_list[-self.level_zwhere_and_logit_output - 1]
        self.encode_logit = Mlp1by1(ch_in=self.ch_in_logit,
                                    ch_out=self.dim_logit,
                                    ch_hidden=(self.ch_in_logit + self.dim_logit) // 2)

        self.ch_in_bg = self.ch_list[-self.level_background_output - 1]
        self.encode_background = Mlp1by1(ch_in=self.ch_in_bg,
                                         ch_out=self.dim_zbg,
                                         ch_hidden=(self.ch_in_bg + self.dim_zbg) // 2)

    def forward(self, x: torch.Tensor, verbose: bool):
        # input_w, input_h = x.shape[-2:]
        if verbose:
            print("INPUT ---> shape ", x.shape)

        # Down path and save the tensor which will need to be concatenated
        raw_image = x
        to_be_concatenated = deque()
        for i, down in enumerate(self.down_path):
            x = down(x, verbose)
            if verbose:
                print("down   ", i, " shape ", x.shape)
            if i < self.n_max_pool:
                to_be_concatenated.append(x)
                if verbose:
                    print("appended")

        # During up path I need to concatenate with the tensor obtained during the down path
        # If distance is < self.n_prediction_maps I need to export a prediction map
        zwhere, logit, zbg = None, None, None
        for i, up in enumerate(self.up_path):
            dist_to_end_of_net = self.n_max_pool - i
            if dist_to_end_of_net == self.level_zwhere_and_logit_output:
                zwhere = self.encode_zwhere(x)
                logit = self.encode_logit(x)
                if self.training and self.grad_logit_max is not None:
                    logit.register_hook(lambda grad: grad.clamp(min=-self.grad_logit_max, max=self.grad_logit_max))

            if dist_to_end_of_net == self.level_background_output:
                zbg = self.encode_background(x)  # only few channels needed for predicting bg

            x = up(to_be_concatenated.pop(), x, verbose)
            if verbose:
                print("up     ", i, " shape ", x.shape)

        # always add a pred_map to the rightmost layer (which had distance 0 from the end of the net)
        # TODO: Remove this concatenation
        # if self.concatenate_raw_image_to_fmap:
        #     features = torch.cat((self.pred_features(x), raw_image), dim=-3)  # Here I am concatenating the raw image
        # else:
        #    features = self.pred_features(x)
        features = self.pred_features(raw_image)

        return UNEToutput(zwhere=zwhere,
                          logit=logit,
                          zbg=zbg,
                          features=features)

    def show_grid(self, ref_image):
        """ overimpose a grid the size of the corresponding resolution of each unet layer """

        assert len(ref_image.shape) == 4
        batch, ch, w_raw, h_raw = ref_image.shape

        nj = len(self.j_list)
        check_board = ref_image.new_zeros((nj, 1, 1, w_raw, h_raw))  # for each batch and channel the same check_board
        counter_w = torch.arange(w_raw)
        counter_h = torch.arange(h_raw)

        for k in range(nj):
            j = self.j_list[k]
            index_w = 1 + ((counter_w // j) % 2)  # either 1 or 2
            dx = index_w.float().view(w_raw, 1)
            index_h = 1 + ((counter_h // j) % 2)  # either 1 or 2
            dy = index_h.float().view(1, h_raw)
            check_board[k, 0, 0, :, :] = 0.25 * (dy * dx)  # dx*dy=1,2,4 multiply by 0.25 to have (0,1)

        assert check_board.shape == (nj, 1, 1, w_raw, h_raw)

        # I need to sum:
        # check_board of shape: --> levels, 1,      1, w_raw, h_raw
        # ref_image of shape ----->         batch, ch, w_raw, h_raw
        return ref_image + check_board

