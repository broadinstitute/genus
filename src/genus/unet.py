import torch
from .conv import UnetUpBlock, UnetDownBlock, SameSpatialResolution
from collections import deque
from .namedtuple import UNEToutput
import numpy
from typing import Optional


class UNetNew(torch.nn.Module):
    def __init__(self,
                 pre_processor: Optional[torch.nn.Module],
                 scale_factor_boundingboxes: int,
                 ch_in: int,
                 ch_out: int,
                 dim_zbg: int,
                 dim_zwhere: int,
                 dim_logit: int,
                 pretrained: bool):

        super().__init__()

        # Parameters UNet
        self.ch_raw_image = ch_in
        self.ch_output_features = ch_out
        self.dim_zbg = dim_zbg
        self.dim_zwhere = dim_zwhere
        self.dim_logit = dim_logit

        self.backbone = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                       in_channels=3, out_channels=1, init_features=32, pretrained=pretrained)

        # Some surgery last layer with feature map
        self.backbone.conv = torch.nn.Conv2d(32, self.ch_output_features, kernel_size=(1, 1), stride=(1, 1))

        # First layer with pre_processor if necessary
        if (pre_processor is None) and (self.ch_raw_image == 3):
            # no replacement is necessary
            scale_factor_initial_layer = 1.0
        else:
            pre_processor = torch.nn.Conv2d(self.ch_raw_image, 32,
                                            kernel_size=(3, 3),
                                            stride=(1, 1),
                                            padding=(1, 1),
                                            bias=False) if pre_processor is None else pre_processor
            x = torch.zeros((1, ch_in, 256, 256))
            y = pre_processor.forward(x)
            scale_factor_initial_layer = float(x.shape[-1]) / y.shape[-1]
            is_power_of_2 = numpy.log2(float(scale_factor_initial_layer)) % 1.0 == 0
            assert is_power_of_2, "The pre_processor need to reduce the spatial resolution by a power of 2. " \
                                  "Your reduction is {0}".format(scale_factor_initial_layer)
            assert y.shape[-3] == 32, "The pre_processor need to take {0} channels and " \
                                      "produce 32 channels".format(self.ch_raw_image)
            self.backbone.encoder1.enc1conv1 = pre_processor

        assert numpy.log2(float(scale_factor_boundingboxes)) % 1.0 == 0, \
            "scale_factor_boundingboxes {0} needs to be a power of 2".format(scale_factor_boundingboxes)

        scale_factor_boundingboxes_min = scale_factor_initial_layer
        scale_factor_boundingboxes_max = scale_factor_initial_layer * 2 ** 4
        assert (scale_factor_boundingboxes >= scale_factor_boundingboxes_min) and \
               (scale_factor_boundingboxes <= scale_factor_boundingboxes_max), \
            "scale_factor_boundingboxes {0} needs to in the range ({1},{2})".format(scale_factor_boundingboxes,
                                                                                    scale_factor_boundingboxes_min,
                                                                                    scale_factor_boundingboxes_max)

        # This is preparation for surgey
        self.hooked_handles_bg = None
        self.hooked_handles_logit = None
        self.hooked_bg = None
        self.hooked_logit = None

        def capture_bg_fn(module, input, output):
            self.hooked_bg = output
            # print("captured bg", self.hooked_bg.shape)

        def capture_logit_fn(module, input, output):
            self.hooked_logit = output
            # print("captured logit", self.hooked_logit.shape)

        self.hook_handles_bg = self.backbone.bottleneck.bottleneckrelu2.register_forward_hook(capture_bg_fn)
        ch_in_bg = 512

        level_logit_output = int(numpy.log2(float(scale_factor_boundingboxes) / scale_factor_initial_layer))

        if level_logit_output == 0:
            self.hook_handles_logit = self.backbone.decoder1.dec1relu2.register_forward_hook(capture_logit_fn)
            ch_in_logit = 32
        elif level_logit_output == 1:
            self.hook_handles_logit = self.backbone.decoder2.dec2relu2.register_forward_hook(capture_logit_fn)
            ch_in_logit = 64
        elif level_logit_output == 2:
            self.hook_handles_logit = self.backbone.decoder3.dec3relu2.register_forward_hook(capture_logit_fn)
            ch_in_logit = 128
        elif level_logit_output == 3:
            self.hook_handles_logit = self.backbone.decoder4.dec4relu2.register_forward_hook(capture_logit_fn)
            ch_in_logit = 256
        elif level_logit_output == 4:
            self.hook_handles_logit = self.backbone.bottleneck.bottleneckrelu2.register_forward_hook(capture_logit_fn)
            ch_in_logit = 512
        else:
            raise Exception("level_logit_output is wrong!")

        self.encode_logit = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in_logit, out_channels=ch_in_logit // 2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=ch_in_logit // 2, out_channels=self.dim_logit, kernel_size=1, padding=0)
        )

        ch_in_zwhere = ch_in_logit
        self.encode_zwhere = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in_zwhere, out_channels=ch_in_zwhere // 2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=ch_in_zwhere // 2, out_channels=2 * self.dim_zwhere, kernel_size=1, padding=0)
        )

        self.encode_background = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in_bg, out_channels=ch_in_bg // 2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=ch_in_bg // 2, out_channels=2 * self.dim_zbg, kernel_size=1, padding=0)
        )

    def forward(self, x, verbose: bool):
        fmap = self.backbone(x)
        zbg = self.encode_background(self.hooked_bg)
        zwhere = self.encode_zwhere(self.hooked_logit)
        logit = self.encode_logit(self.hooked_logit)

        if verbose:
            print("INPUT ---> shape ", x.shape)
            print("FMAP ----> shape ", fmap.shape)
            print("LOGIT ---> shape ", logit.shape)
            print("ZWHERE --> shape ", zwhere.shape)
            print("ZBG -----> shape ", zbg.shape)

        return UNEToutput(zwhere=zwhere,
                          logit=logit,
                          zbg=zbg,
                          features=fmap)

    def show_grid(self, ref_image):
        """ overimpose a grid the size of the corresponding resolution of each unet layer """

        assert len(ref_image.shape) == 4
        batch, ch, w_raw, h_raw = ref_image.shape

        j_list = 2**torch.arange(4)

        nj = len(j_list)
        check_board = ref_image.new_zeros((nj, 1, 1, w_raw, h_raw))  # for each batch and channel the same check_board
        counter_w = torch.arange(w_raw)
        counter_h = torch.arange(h_raw)

        for k in range(nj):
            j = j_list[k]
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


#----------------------------------------------------------------------------------


class UNet(torch.nn.Module):
    def __init__(self,
                 scale_factor_initial_layer: int,
                 scale_factor_background: int,
                 scale_factor_boundingboxes: int,
                 ch_in: int,
                 ch_out: int,
                 ch_before_first_maxpool: int,
                 dim_zbg: int,
                 dim_zwhere: int,
                 dim_logit: int):

        super().__init__()

        # Parameters UNet
        assert numpy.log2(float(scale_factor_initial_layer)) % 1.0 == 0
        assert numpy.log2(float(scale_factor_background)) % 1.0 == 0
        assert numpy.log2(float(scale_factor_boundingboxes)) % 1.0 == 0

        self.n_max_pool = int(numpy.log2(float(scale_factor_background)))
        self.level_zwhere_and_logit_output = int(numpy.log2(float(scale_factor_boundingboxes)))
        self.scale_factor_initial_layer = scale_factor_initial_layer
        self.level_background_output = self.n_max_pool
        self.ch_raw_image = ch_in
        self.ch_output_features = ch_out
        self.ch_before_maxpool = ch_before_first_maxpool
        self.dim_zbg = dim_zbg
        self.dim_zwhere = dim_zwhere
        self.dim_logit = dim_logit

        if self.scale_factor_initial_layer != 1:
            raise NotImplementedError("At the moment scale_factor_initial_layer should be 1")

        # Initializations
        ch = self.ch_before_maxpool
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

        # Prediction Heads
        ch_in_features = self.ch_list[-1]
        self.pred_features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in_features, out_channels=ch_in_features//2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=ch_in_features//2, out_channels=self.ch_output_features, kernel_size=1, padding=0)
        )

        ch_in_logit = self.ch_list[-self.level_zwhere_and_logit_output - 1]
        self.encode_logit = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in_logit, out_channels=ch_in_logit//2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=ch_in_logit//2, out_channels=self.dim_logit, kernel_size=1, padding=0)
        )

        # These two have a factor of 2 in the out_channels b/c I need to predict both mu,std
        ch_in_zwhere = self.ch_list[-self.level_zwhere_and_logit_output - 1]
        self.encode_zwhere = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in_zwhere, out_channels=ch_in_zwhere//2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=ch_in_zwhere//2, out_channels=2*self.dim_zwhere, kernel_size=1, padding=0)
        )

        ch_in_bg = self.ch_list[-self.level_background_output - 1]
        self.encode_background = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=ch_in_bg, out_channels=ch_in_bg//2, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=ch_in_bg//2, out_channels=2*self.dim_zbg, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, verbose: bool):
        # raw_image = x
        # input_w, input_h = raw_image.shape[-2:]
        if verbose:
            print("INPUT ---> shape ", x.shape)

        # Down path and save the tensor which will need to be concatenated
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

            # print("DEBUG", dist_to_end_of_net, self.level_zwhere_and_logit_output)

            if dist_to_end_of_net == self.level_zwhere_and_logit_output:
                zwhere = self.encode_zwhere(x)
                logit = self.encode_logit(x)

            if dist_to_end_of_net == self.level_background_output:
                zbg = self.encode_background(x)  # only few channels needed for predicting bg

            x = up(to_be_concatenated.pop(), x, verbose)
            if verbose:
                print("up     ", i, " shape ", x.shape)

        # always add a pred_map to the rightmost layer (which had distance 0 from the end of the net)
        features = self.pred_features(x)

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

