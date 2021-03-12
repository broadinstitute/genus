import torch
import torch.nn as nn
import torch.nn.functional as F
from .namedtuple import ZZ

EPS_STD = 1E-3  # standard_deviation = F.softplus(x) + EPS_STD >= EPS_STD
LOW_RESOLUTION_BG = (5, 5)
CH_BG_MAP = 32


class MLP_1by1(nn.Module):
    """ Use 1x1 convolution, if ch_hidden <= 0 there is NO hidden layer """
    def __init__(self, ch_in: int, ch_out: int, ch_hidden: int):
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
        return self.mlp_1by1(x)


class Encoder1by1(nn.Module):
    def __init__(self, ch_in: int, dim_z: int, ch_hidden: int):
        super().__init__()
        self.dim_z = dim_z
        self.predict = MLP_1by1(ch_in=ch_in, ch_out=2*self.dim_z, ch_hidden=ch_hidden)

    def forward(self, x: torch.Tensor) -> ZZ:
        mu, std = torch.split(self.predict(x), self.dim_z, dim=-3)
        return ZZ(mu=mu, std=F.softplus(std) + EPS_STD)


class Decoder1by1Linear(nn.Module):
    """ Decode z with 1x1 convolutions. z can have any number of leading dimensions """
    def __init__(self, dim_z: int, ch_out: int, groups: int = 1):
        super().__init__()
        # if groups=1 all inputs convolved to produce all outputs
        # if groups=in_channels each channel is convolved with its set of filters
        self.predict = nn.Conv2d(dim_z,
                                 ch_out,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True,
                                 groups=groups)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.predict(z.flatten(end_dim=-4))
        return x.view(list(z.shape[:-3]) + list(x.shape[-3:]))


class EncoderBackground(nn.Module):
    """ Encode bg_map into -> bg_mu, bg_std
        Use  adaptive_avg_2D adaptive_max_2D so that any input spatial resolution can work
    """

    def __init__(self, ch_in: int, dim_z: int):
        super().__init__()
        self.ch_in = ch_in
        self.dim_z = dim_z

        ch_hidden = (CH_BG_MAP + dim_z)//2

        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=CH_BG_MAP, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=CH_BG_MAP, out_channels=ch_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch_hidden, out_channels=2*self.dim_z, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> ZZ:
        y = self.convolutional(x)  # B, 2*dim_z, small_w, small_h
        mu, std = torch.split(y, self.dim_z, dim=-3)  # (B, dim_z, small_w, small_h) , (B, dim_z, small_w, small_h)
        return ZZ(mu=mu, std=F.softplus(std) + EPS_STD)


class DecoderBackground(nn.Module):
    """
    Decode z to background

    Note:
        Observation ConvTranspose2D with:
        1. k=4, s=2, p=1 -> double the spatial dimension
    """
    def __init__(self, ch_out: int, dim_z: int, n_up_conv: int):
        super().__init__()
        self.dim_z = dim_z
        self.ch_out = ch_out
        self.upconv = torch.nn.ModuleList()

        input_ch = self.dim_z
        for n in range(0, n_up_conv):
            self.upconv.append(torch.nn.ConvTranspose2d(in_channels=input_ch, out_channels=CH_BG_MAP,
                                                        kernel_size=4, stride=2, padding=1))
            self.upconv.append(torch.nn.ReLU(inplace=True))
            input_ch = CH_BG_MAP
        self.upconv.append(torch.nn.ConvTranspose2d(in_channels=CH_BG_MAP, out_channels=self.ch_out, kernel_size=1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: tensor of shape (*, CH_BG_MAP, small_w, small_h)

        Returns:
            background of shape (*, self.ch_out, W, H)

        Note:
            Works for any number of leading dimensions
        """
        independent_dim = list(z.shape[:-3])
        x = z.flatten(end_dim=-4)
        for i, module in enumerate(self.upconv):
            x = module(x)
        dependent_dim = list(x.shape[-3:])
        return x.view(independent_dim + dependent_dim)


class DecoderConv(nn.Module):
    def __init__(self, size: int, dim_z: int, ch_out: int):
        super().__init__()
        self.width = size
        assert self.width == 28
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
        x2 = self.decoder(x1).view(independent_dim + [self.ch_out, self.width, self.width])
        return x2


class EncoderConv(nn.Module):
    def __init__(self, size: int, ch_in: int, dim_z: int):
        super().__init__()
        self.ch_in: int = ch_in
        self.width: int = size
        assert (self.width == 28 or self.width == 56)
        self.dim_z = dim_z

        self.conv = nn.Sequential(
            torch.nn.Conv2d(in_channels=self.ch_in, out_channels=32, kernel_size=4, stride=1, padding=2),  # 28,28
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 14,14
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),  # 7,7
        )

        self.compute_mu = nn.Linear(64 * 7 * 7, self.dim_z)
        self.compute_std = nn.Linear(64 * 7 * 7, self.dim_z)

    def forward(self, x: torch.Tensor) -> ZZ:  # this is right

        independent_dim = list(x.shape[:-3])  # this might includes: enumeration, n_boxes, batch_size
        x1 = self.conv(x.flatten(end_dim=-4)).view(-1, 64 * 7 * 7)  # flatten the dependent dimension
        mu = self.compute_mu(x1).view(independent_dim + [self.dim_z])
        std = F.softplus(self.compute_std(x1)).view(independent_dim + [self.dim_z])
        return ZZ(mu=mu, std=std + EPS_STD)
