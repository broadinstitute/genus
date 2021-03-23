import torch
import torch.nn as nn

CH_BG_MAP = 32


class DoubleSpatialResolution(nn.Module):
    """
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


class HalvesSpatialResolution(nn.Module):
    """
    Tiny wrapper around Conv2D which halves the spatial resolution of a tensor.
    """
    def __init__(self, ch_in: int, ch_out: int, reflection_padding: bool = True):
        super().__init__()
        if reflection_padding:
            self.conv_half = nn.Sequential(
                nn.ReflectionPad2d(padding=1),
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=0, bias=True)
            )
        else:
            self.conv_half = nn.Sequential(
                nn.ConstantPad2d(padding=1, value=0.0),
                nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=0, bias=True)
            )

    def forward(self, x, verbose=False):
        y = self.conv_half(x)
        if verbose:
            print("input -> output", x.shape, y.shape)
        return y


class SameSpatialResolution(nn.Module):
    """
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
                self.conv_same = nn.Sequential(
                    nn.ConstantPad2d(padding=1, value=0.0),
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
                )

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
    """ Performs: up_conv + concatenation + SameSpatialResolution
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


class DecoderBackground(nn.Module):
    """
    Decode zbg to background.
    The corresponding encoder is the contracting path of the UNET

    Note:
        It relies on :class:`DoubleSpatialResolution` which doubles the spatial resolution of the tensor
        at each application.
    """

    def __init__(self, ch_in: int, ch_out: int, n_up_conv: int):
        """
        Args:
             ch_in: int, number of channels of the latent code
             ch_out: int, number of channels of the output
             n_up_conv: int, number of application of :class:`DoubleSpatialResolution`.
                Each application double the spatial resolution
        """

        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.upconv = torch.nn.ModuleList()

        input_ch = self.ch_in
        for n in range(0, n_up_conv-1):
            self.upconv.append(DoubleSpatialResolution(ch_in=input_ch, ch_out=CH_BG_MAP))
            self.upconv.append(torch.nn.ReLU(inplace=True))
            input_ch = CH_BG_MAP
        self.upconv.append(DoubleSpatialResolution(ch_in=CH_BG_MAP, ch_out=self.ch_out))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: tensor of shape (*, ch_in, small_w, small_h)

        Returns:
            background of shape (*, ch_out, W, H) where :math:`W = small_W \times 2**n_up_conv`

        Note:
            Works for any number of leading dimensions

        """
        x = z.flatten(end_dim=-4)
        for i, module in enumerate(self.upconv):
            x = module(x)
        return x.view(list(z.shape[:-3]) + list(x.shape[-3:]))


class EncoderConv(nn.Module):
    """
    Encode the glimpse into zinstance.

    Note:
        It relies on :class:`HalvesSpatialResolution` which halves the spatial resolution of the tensor
        at each application.
    """

    def __init__(self, size: int, ch_in: int, ch_out: int):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.width = size
        assert self.width == 32

        self.conv = nn.Sequential(
            HalvesSpatialResolution(ch_in=self.ch_in, ch_out=2*self.ch_in),  # 16 x 16
            nn.ReLU(inplace=True),
            HalvesSpatialResolution(ch_in=2*self.ch_in, ch_out=4*self.ch_in),  # 8 x 8
            nn.ReLU(inplace=True),
            HalvesSpatialResolution(ch_in=4*self.ch_in, ch_out=8*self.ch_in),  # 4 x 4
            nn.ReLU(inplace=True),
            HalvesSpatialResolution(ch_in=8*self.ch_in, ch_out=self.ch_out)  # 2 x 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.flatten(end_dim=-4))
        return y.view(list(x.shape[:-3]) + list(y.shape[-3:]))


class DecoderConv(nn.Module):
    """
    Decode zinstance into a glimpse.

    Note:
        It relies on :class:`DoubleSpatialResolution` which doubles the spatial resolution of the tensor
        at each application.
    """

    def __init__(self, size: int, ch_in: int, ch_out: int):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.width = size
        assert self.width == 32

        self.decoder = nn.Sequential(
            DoubleSpatialResolution(self.ch_in, self.ch_in//2),  # 4 x 4
            nn.ReLU(inplace=True),
            DoubleSpatialResolution(self.ch_in//2, self.ch_in//4),  # 8 x 8
            nn.ReLU(inplace=True),
            DoubleSpatialResolution(self.ch_in//4, self.ch_in//8),  # 16 x 16
            nn.ReLU(inplace=True),
            DoubleSpatialResolution(self.ch_in//8, self.ch_out)  # 32 x 32
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.decoder(z.flatten(end_dim=-4))
        return y.view(list(z.shape[:-3]) + list(y.shape[-3:]))


