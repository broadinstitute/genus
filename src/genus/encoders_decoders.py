import torch
import torch.nn as nn
import torch.nn.functional as F
from .namedtuple import ZZ, BB
from typing import Tuple
from .util import convert_to_box_list

EPS_STD = 1E-3  # standard_deviation = F.softplus(x) + EPS_STD >= EPS_STD
LOW_RESOLUTION_BG = (5, 5)
CH_BG_MAP = 32


# --------- HELPER FUNCTION ---------------------


class Mlp1by1(nn.Module):
    """ FROM DAPI """
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

# ---------- ENCODERS -------------------------------


class EncoderWhereLogit(nn.Module):
    """ QUITE SIMILAR TO DAPI """
    def __init__(self, ch_in: int, dim_z: int, dim_logit: int, ch_hidden: int):
        super().__init__()
        self.dim_z = dim_z
        self.dim_logit = dim_logit
        self.predict = Mlp1by1(ch_in=ch_in, ch_out=2*self.dim_z+self.dim_logit, ch_hidden=ch_hidden)

    def forward(self, x: torch.Tensor) -> Tuple[ZZ, torch.Tensor]:
        logit, mu, std = torch.split(self.predict(x), (self.dim_logit, self.dim_z, self.dim_z), dim=-3)
        return ZZ(mu=mu, std=F.softplus(std) + EPS_STD), logit


class EncoderBackground(nn.Module):
    """ TAKEN FROM DAPI """

    def __init__(self, ch_in: int, dim_z: int):
        super().__init__()
        self.ch_in = ch_in
        self.dim_z = dim_z

        ch_hidden = (CH_BG_MAP + dim_z)//2

        self.bg_map_before = nn.Conv2d(self.ch_in, CH_BG_MAP, kernel_size=1, stride=1, padding=0, bias=True)
        self.adaptive_avg_2D = nn.AdaptiveAvgPool2d(output_size=LOW_RESOLUTION_BG)
        self.adaptive_max_2D = nn.AdaptiveAvgPool2d(output_size=LOW_RESOLUTION_BG)

        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=2*CH_BG_MAP, out_channels=CH_BG_MAP, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=CH_BG_MAP, out_channels=CH_BG_MAP, kernel_size=3),  # 3x3
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=CH_BG_MAP, out_channels=CH_BG_MAP, kernel_size=3),  # 1x1
            nn.ReLU(inplace=True))

        self.linear = nn.Sequential(
            nn.Linear(in_features=CH_BG_MAP, out_features=ch_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=ch_hidden, out_features=2*self.dim_z))

    def forward(self, x: torch.Tensor) -> ZZ:
        # TODO: see how fast.ai does UNET
        y1 = self.bg_map_before(x)  # B, 32, small_w, small_h
        y2 = torch.cat((self.adaptive_avg_2D(y1), self.adaptive_max_2D(y1)), dim=-3)  # 2*ch_bg_map , low_res, low_res
        y3 = self.convolutional(y2)  # B, 32, 1, 1
        mu, std = torch.split(self.linear(y3.flatten(start_dim=-3)), self.dim_z, dim=-1)  # B, dim_z
        return ZZ(mu=mu, std=F.softplus(std) + EPS_STD)


class EncoderInstance(nn.Module):
    """ TAKEN FROM DAPI """
    def __init__(self, size: int, ch_in: int, dim_z: int):
        super().__init__()
        self.ch_in: int = ch_in
        self.width: int = size
        assert self.width == 28
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
        dependent_dim = list(x.shape[-3:])  # this includes: ch, width, height
        # assert dependent_dim == [self.ch_raw_image, self.width, self.width]
        x1 = x.view([-1] + dependent_dim)  # flatten the independent dimensions
        x2 = self.conv(x1).view(-1, 64 * 7 * 7)  # flatten the dependent dimension
        mu = self.compute_mu(x2).view(independent_dim + [self.dim_z])
        std = F.softplus(self.compute_std(x2)).view(independent_dim + [self.dim_z])
        return ZZ(mu=mu, std=std + EPS_STD)


# ------ DECODER --------------------

class DecoderWhere(nn.Module):
    def __init__(self, dim_z: int):
        super().__init__()
        self.dim_z = dim_z

        self.predict = nn.Conv2d(dim_z,
                                 4,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True,
                                 groups=1)

    @staticmethod
    def tmaps_to_bb(tmaps, width_raw_image: int, height_raw_image: int, min_box_size: float, max_box_size: float):
        tx_map, ty_map, tw_map, th_map = torch.split(tmaps, 1, dim=-3)
        n_width, n_height = tx_map.shape[-2:]
        ix_array = torch.arange(start=0, end=n_width, dtype=tx_map.dtype, device=tx_map.device)
        iy_array = torch.arange(start=0, end=n_height, dtype=tx_map.dtype, device=tx_map.device)
        ix_grid, iy_grid = torch.meshgrid([ix_array, iy_array])

        bx_map: torch.Tensor = width_raw_image * (ix_grid + tx_map) / n_width
        by_map: torch.Tensor = height_raw_image * (iy_grid + ty_map) / n_height
        bw_map: torch.Tensor = min_box_size + (max_box_size - min_box_size) * tw_map
        bh_map: torch.Tensor = min_box_size + (max_box_size - min_box_size) * th_map
        return BB(bx=convert_to_box_list(bx_map).squeeze(-1),
                  by=convert_to_box_list(by_map).squeeze(-1),
                  bw=convert_to_box_list(bw_map).squeeze(-1),
                  bh=convert_to_box_list(bh_map).squeeze(-1))

    def forward(self, z: torch.Tensor,
                width_raw_image: int,
                height_raw_image: int,
                min_box_size: int,
                max_box_size: int) -> BB:

        return self.tmaps_to_bb(tmaps=torch.sigmoid(self.predict(z)),
                                width_raw_image=width_raw_image,
                                height_raw_image=height_raw_image,
                                min_box_size=min_box_size,
                                max_box_size=max_box_size)


class DecoderBackground(nn.Module):
    """ TAKEN FROM DAPI """
    def __init__(self, ch_out: int, dim_z: int):
        super().__init__()
        self.dim_z = dim_z
        self.ch_out = ch_out
        self.upsample = nn.Linear(self.dim_z, 5 * 5 * 32)
        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 10,10
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),  # 20,20
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),  # 40,40
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels=16, out_channels=self.ch_out, kernel_size=4, stride=2, padding=1),  # 80,80
        )

    def forward(self, z: torch.Tensor, high_resolution: tuple) -> torch.Tensor:
        # From (B, dim_z) to (B, ch_out, 28, 28) to (B, ch_out, w_raw, h_raw)
        x0 = self.upsample(z).view(-1, 32, 5, 5)
        x1 = self.decoder(x0)  # B, ch_out, 80, 80
        return F.interpolate(x1, size=high_resolution, mode='bilinear', align_corners=True)


class DecoderInstance(nn.Module):
    """ TAKEN FROM DAPI """
    def __init__(self, size: int, dim_z: int, ch_out: int):
        super().__init__()
        self.width = size
        self.dim_z: int = dim_z
        self.ch_out: int = ch_out
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
        return self.decoder(x1).view(independent_dim + [self.ch_out, self.width, self.width])
