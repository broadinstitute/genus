import torch
import torch.nn as nn
import torch.nn.functional as F
from .namedtuple import ZZ

EPS_STD = 1E-3  # standard_deviation = F.softplus(x) + EPS_STD >= EPS_STD
LOW_RESOLUTION_BG = (5, 5)
CH_BG_MAP = 32


class DecoderBackground(nn.Module):
    """
    Decode zbg to background.

    """
    def __init__(self, ch_out: int, dim_z: int, n_up_conv: int):
        """
        Args:
             ch_out
        Note:
            It relies on the observation that ConvTranspose2D with (k=4, s=2, p=1) -> double the spatial dimension
        """

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
        x = z.flatten(end_dim=-4)
        for i, module in enumerate(self.upconv):
            x = module(x)
        return x.view(list(z.shape[:-3]) + list(x.shape[-3:]))


class DecoderConv(nn.Module):
    def __init__(self, size: int, dim_z: int, ch_out: int):
        super().__init__()
        self.width = size
        assert self.width == 32
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
