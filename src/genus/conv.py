import torch
import torch.nn as nn
import numpy

CH_BG_MAP = 32
LOW_RESOLUTION_BG = 5

class DoubleSpatialResolution(nn.Module):
    """
    Helper Function:
    Tiny wrapper around ConvTranspose2D which doubles the spatial resolution of a tensor.
    """
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv_double = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, x, verbose=False):
        y = self.conv_double(x)
        if verbose:
            print("input -> output", x.shape, y.shape)
        return y


class SameSpatialResolution(nn.Module):
    """
    Helper Function:
    Implements [conv] or [conv + relu + conv]
    The output has the same spatial resolution of the input
    """
    def __init__(self, ch_in: int, ch_out: int, double_or_single: str = "double", reflection_padding: bool = True):
        """
        Args:
            ch_in: int, channels input
            ch_out: int, channels output
            double_or_single: str, whether to use one or two convolution layers
            reflection_padding: bool, if true use reflection padding, if false pad with zero
        """
        super().__init__()
        if double_or_single == "single":
            if reflection_padding:
                self.conv_same = nn.Sequential(
                    nn.ReflectionPad2d(padding=1),
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=0, bias=True)
                )
            else:
                self.conv_same = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)

        elif double_or_single == "double":
            if reflection_padding:
                self.conv_same = nn.Sequential(
                    nn.ReflectionPad2d(padding=2),
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=0, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=0, bias=True)
                )
            else:
                self.conv_same = nn.Sequential(
                    nn.ConstantPad2d(padding=2, value=0.0),
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=0, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=0, bias=True)
                )

    def forward(self, x, verbose=False):
        y = self.conv_same(x)
        if verbose:
            print("input -> output", x.shape, y.shape)
        return y


class UnetDownBlock(nn.Module):
    """
    Helper function:
    Performs the sequence max_pool(2x2) + SameSpatialResolution
    The spatial extension of the tensor is reduced in half.
    """
        
    def __init__(self, ch_in: int, ch_out: int):
        """
        Args:
            ch_in: int, input channels
            ch_out: int, output channels
        """
        super().__init__()
        self.max_pool_layer = nn.MaxPool2d(2, 2)
        self.same_conv = SameSpatialResolution(ch_in, ch_out, reflection_padding=True, double_or_single="double")

    def forward(self, x0, verbose=False):

        x1 = self.max_pool_layer.forward(x0)
        x2 = self.same_conv.forward(x1)

        if verbose:
            print("input -> output", x0.shape, x2.shape)

        return x2


class UnetUpBlock(nn.Module):
    """
    Helper function:

    Performs: up_conv + concatenation + SameSpatialResolution
    During upconv the channels go from ch_in to ch_in/2
    Since I am concatenating with something which has ch_in/2
    During the SameSpatialResolution the channels go from ch_in,ch_out
    Therefore during initialization only ch_in,ch_out of the SameSpatialResolution need to be specified.
    The forward function takes two tensors: x_from_contracting_path , x_to_upconv
    """
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.ch_in = ch_in
        self.up_conv_layer = DoubleSpatialResolution(ch_in, int(ch_in/2))
        self.same_conv = SameSpatialResolution(ch_in, ch_out)

    def forward(self, x_from_compressing_path, x_to_upconv, verbose=False):
        x = self.up_conv_layer.forward(x_to_upconv)
        x1 = torch.cat((x_from_compressing_path, x), dim=-3)  # concatenate along the channel dimension

        if verbose:
            print("x_from_compressing_path", x_from_compressing_path.shape)
            print("x_to_upconv", x_to_upconv.shape)
            print("after_upconv", x.shape)
            print("after concat", x1.shape)
            print("ch_in", self.ch_in)

        x2 = self.same_conv.forward(x1)
        return x2


class Mlp1by1(nn.Module):
    """ Helper function """

    def __init__(self, ch_in: int, ch_out: int, ch_hidden: int):
        """
        Args:
            ch_in: int input channels
            ch_out: int output channels
            ch_hidden: int hidden layer channels, if :attr:`ch_hidden` <=0 there is NO hidden layer

        Returns:
            A tensor with the same dimensions as input but a different number of channels specified by :attr:`ch_out`
        """
        super().__init__()
        if ch_hidden <= 0:
            self.mlp_1by1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.mlp_1by1 = nn.Sequential(
                nn.Conv2d(ch_in, ch_hidden, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_hidden, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp_1by1(x.flatten(end_dim=-4))
        return y.view(list(x.shape[:-3]) + list(y.shape[-3:]))

#---- Logit

class EncoderLogit(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.encode_logit = Mlp1by1(ch_in=ch_in, ch_out=ch_out, ch_hidden= ch_in // 2)

    def forward(self, x, verbose=False):
        return self.encode_logit(x)

#----- Zwhere

class EncoderWhere(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.encode_zwhere = Mlp1by1(ch_in=ch_in, ch_out=ch_out, ch_hidden=ch_in // 2)

    def forward(self, x, verbose=False):
        return self.encode_zwhere(x)

class DecoderWhere(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.decode_zwhere = torch.nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1, groups=4)

    def forward(self, x):
        return self.decode_zwhere(x)

#--------- INSTANCE -------

class EncoderInstance(nn.Module):
    def __init__(self, glimpse_size: int, ch_in: int, dim_z: int):
        super().__init__()
        self.ch_in = ch_in
        assert glimpse_size == 28
        self.dim_z = dim_z

        self.conv = nn.Sequential(
            torch.nn.Conv2d(in_channels=self.ch_in, out_channels=32, kernel_size=4, stride=1, padding=2),  # 28,28
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 14,14
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),  # 7,7
        )
        self.linear = nn.Linear(64 * 7 * 7, 2 * self.dim_z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x.flatten(end_dim=-4)).view(-1, 64 * 7 * 7)  # flatten the dependent dimension
        x2 = self.linear(x1).view(list(x.shape[:-3]) + [2*self.dim_z])
        return x2

class DecoderInstance(nn.Module):
    def __init__(self, glimpse_size: int, dim_z: int, ch_out: int):
        super().__init__()
        self.glimpse_size = glimpse_size
        assert self.glimpse_size == 28
        self.dim_z = dim_z
        self.ch_out = ch_out
        self.upsample = nn.Linear(self.dim_z, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  64,  14,  14
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),  # B,  32, 28, 28
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, self.ch_out, 4, 1, 2)  # B, ch, 28, 28
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        independent_dim = list(z.shape[:-1])
        x1 = self.upsample(z.view(-1, self.dim_z)).view(-1, 64, 7, 7)
        x2 = self.decoder(x1).view(independent_dim + [self.ch_out, self.glimpse_size, self.glimpse_size])
        return x2

#----- Zbg ------
# TODO: Make zbg fully connected not on 2D grid
# zbg can be either:
# 1. batch, ch, small_w, small_height (if using convolutional encoder/decoder)
# 2. batch, ch, 1, 1  (if using fully connected layer)

class EncoderBg(nn.Module):

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.linear = Mlp1by1(ch_in=ch_in, ch_out=CH_BG_MAP, ch_hidden=(ch_in + CH_BG_MAP) // 2)
        self.adaptive_avg_2D = nn.AdaptiveAvgPool2d(output_size=LOW_RESOLUTION_BG) # 5x5
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=CH_BG_MAP, out_channels=2*CH_BG_MAP, kernel_size=3, padding=0),  # 3x3
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2*CH_BG_MAP, out_channels=4*CH_BG_MAP, kernel_size=3),  # 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=4 * CH_BG_MAP, out_channels=ch_out, kernel_size=1)  # 1x1
        )

    def forward(self, x, verbose=False):
        x1 = torch.nn.ReLU()(self.linear(x))
        x2 = self.adaptive_avg_2D(x1)
        x3 = self.convolutional(x2)
        return x3


class DecoderBg(nn.Module):
    """
    Decode a small patch into a larger patch.

    Note:
        It relies on :class:`DoubleSpatialResolution` which doubles the spatial resolution of the tensor
        at each application.
    """
    CH_MIN_DECODER = 16

    def __init__(self, ch_in: int, ch_out: int, scale_factor: int):
        """
        Args:
            ch_in: int, number of channels in the input
            ch_out: int, number of channels in the output
            scale_factor: integer factor of 2, describes the increase in the spatial extension from input to output
        """
        super().__init__()
        n_levels = numpy.log2(float(scale_factor))
        assert (n_levels % 1.0 == 0) and (n_levels >= 1) # scale_factor is a power of 2 and larger than 2
        self.ch_in = ch_in
        self.ch_out = ch_out

        self.upsample = nn.Conv2d(in_channels=self.ch_in,
                                  out_channels=CH_BG_MAP * LOW_RESOLUTION_BG * LOW_RESOLUTION_BG,
                                  kernel_size=1, padding=0)

        ch = CH_BG_MAP
        ch_half = max(ch // 2, self.CH_MIN_DECODER)
        self.decoder = torch.nn.ModuleList()
        for n in range(0, int(n_levels)-1):
            self.decoder.append(DoubleSpatialResolution(ch, ch))
            self.decoder.append(nn.ReLU(inplace=True))
            self.decoder.append(Mlp1by1(ch_in=ch, ch_out=ch_half, ch_hidden=-1))
            self.decoder.append(nn.ReLU(inplace=True))
            ch = ch_half
            ch_half = max(ch // 2, self.CH_MIN_DECODER)
        self.decoder.append(DoubleSpatialResolution(ch, ch))
        self.decoder.append(nn.ReLU(inplace=True))
        self.decoder.append(Mlp1by1(ch_in=ch, ch_out=self.ch_out, ch_hidden=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4
        y = self.upsample(x).view(-1, CH_BG_MAP, LOW_RESOLUTION_BG, LOW_RESOLUTION_BG)
        for i, module in enumerate(self.decoder):
            y = module(y)
        return y



