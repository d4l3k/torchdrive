from dataclasses import replace
from typing import Tuple

import torch
from torch import nn
from torchvision.models.resnet import resnet18

from torchworld.structures.grid import GridImage, Grid3d
from torchworld.models.resnet_3d import resnet3d18

class Resnet18FPNImage(nn.Module):
    """
    Implements a resnet FPN based off of resnet18.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1: nn.Module = backbone.bn1
        self.relu: nn.Module = backbone.relu

        self.layer1: nn.Module = backbone.layer1
        self.layer2: nn.Module = backbone.layer2
        self.layer3: nn.Module = backbone.layer3

        self.up3_skip = UpsamplingAdd2d(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd2d(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd2d(64, in_channels, scale_factor=2)

    def forward(self, grid: GridImage) -> Tuple[GridImage, GridImage, GridImage, GridImage]:
        """
        Args:
            x: [BS, in_channels, H, W]
        Returns:
            4 grids with the fpn features
            * (in_channels, H, W)
            * (64, H/2, W/2)
            * (128, H/4, W/4)
            * (256, H/8, W/8)
        """
        x = grid.data

        # (H, W)
        skip_x = {"1": x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x["2"] = x
        x = self.layer2(x)
        skip_x["3"] = x

        # (H/8, W/8)
        x4 = self.layer3(x)

        # First upsample to (H/4, W/4)
        x3 = self.up3_skip(x4, skip_x["3"])

        # Second upsample to (H/2, W/2)
        x2 = self.up2_skip(x3, skip_x["2"])

        # Third upsample to (H, W)
        x1 = self.up1_skip(x2, skip_x["1"])

        return (
            replace(grid, data=x1),
            replace(grid, data=x2),
            replace(grid, data=x3),
            replace(grid, data=x4),
        )


class Resnet18FPN3d(nn.Module):
    """
    Implements a resnet FPN based off of resnet18 but in 3d.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        backbone = resnet3d18(
            zero_init_residual=True, final_channels=in_channels*16
        )
        self.first_conv = nn.Conv3d(
            in_channels,
            in_channels*2,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1: nn.Module = backbone.bn1
        self.relu: nn.Module = backbone.relu

        self.layer1: nn.Module = backbone.layer1
        self.layer2: nn.Module = backbone.layer2
        self.layer3: nn.Module = backbone.layer3

        self.up3_skip = UpsamplingAdd3d(
            in_channels*8, in_channels*4, scale_factor=2
        )
        self.up2_skip = UpsamplingAdd3d(
            in_channels*4, in_channels*2, scale_factor=2
        )
        self.up1_skip = UpsamplingAdd3d(
            in_channels*2, in_channels, scale_factor=2
        )

    def forward(self, grid: Grid3d) -> Tuple[GridImage, GridImage, GridImage, GridImage]:
        """
        Args:
            x: [BS, in_channels, Z, Y, X]
        Returns:
            4 grids with the fpn features
            * (in_channels, Z, Y, X)
            * (final_channels/4, Z/2, Y/2, X/2)
            * (final_channels/2, Z/4, Y/4, X/4)
            * (final_channels, Z/8, Y/8, Z/8)
        """
        x = grid.data

        # (H, W)
        skip_x = {"1": x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x["2"] = x
        x = self.layer2(x)
        skip_x["3"] = x

        # (H/8, W/8)
        x4 = self.layer3(x)

        # First upsample to (H/4, W/4)
        x3 = self.up3_skip(x4, skip_x["3"])

        # Second upsample to (H/2, W/2)
        x2 = self.up2_skip(x3, skip_x["2"])

        # Third upsample to (H, W)
        x1 = self.up1_skip(x2, skip_x["1"])

        return (
            replace(grid, data=x1),
            replace(grid, data=x2),
            replace(grid, data=x3),
            replace(grid, data=x4),
        )

class UpsamplingAdd2d(nn.Module):
    """
    Upsamples a 2d input and adds it to the skip connection value.
    """

    def __init__(
        self, in_channels: int, out_channels: int, scale_factor: int = 2
    ) -> None:
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=False
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, h, w]
            x_skip: [batch_size, out_channels, h*2, w*2]
        """
        x = self.upsample_layer(x)
        return x + x_skip


class UpsamplingAdd3d(nn.Module):
    """
    Upsamples a 3d input and adds it to the skip connection value.
    """

    def __init__(
        self, in_channels: int, out_channels: int, scale_factor: int = 2
    ) -> None:
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor, mode="trilinear", align_corners=False
            ),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm3d(out_channels),
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, h, w, d]
            x_skip: [batch_size, out_channels, h*2, w*2, d*2]
        """
        x = self.upsample_layer(x)
        return x + x_skip

