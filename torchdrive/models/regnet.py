import math
from typing import Protocol, Tuple, Type

import torch

from torch import nn
from torchvision import models

from torchdrive.positional_encoding import positional_encoding


def resnet_init(module: nn.Module) -> None:
    """
    Helper method for initializing resnet style model weights.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # Note that there is no bias due to BN
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)


class RegNetConstructor(Protocol):
    def __call__(self, pretrained: bool = False) -> models.RegNet:
        ...


class RegNetEncoder(nn.Module):
    """
    A RegNet based encoder with a positional encoding designed for use with a
    transformer.
    """

    def __init__(
        self,
        cam_shape: Tuple[int, int],
        dim: int,
        trunk: RegNetConstructor = models.regnet_x_800mf,
    ) -> None:
        super().__init__()

        self.model: models.RegNet = trunk(pretrained=True)
        assert len(self.model.trunk_output) == 4
        if trunk == models.regnet_x_1_6gf:
            self.num_ch_enc: Tuple[int, ...] = (32, 72, 168, 408, 912)
        elif trunk == models.regnet_x_800mf:
            self.num_ch_enc = (32, 64, 128, 288, 672)
        elif trunk == models.regnet_x_400mf:
            self.num_ch_enc = (32, 32, 64, 160, 400)
        elif trunk == models.regnet_y_400mf:
            self.num_ch_enc = (32, 48, 104, 208, 440)
        else:
            raise ValueError(f"unknown trunk type {trunk}")

        self.output_shape: Tuple[int, int] = (cam_shape[0] // 16, cam_shape[1] // 16)

        self.f3_encoder = nn.Sequential(
            nn.Conv2d(self.num_ch_enc[3] + 6, dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        resnet_init(self.f3_encoder)
        self.register_buffer(
            "positional_encoding",
            positional_encoding(*self.output_shape),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BS = x.shape[0]
        # adapted from torchvision.models.Resnet
        f0 = self.model.stem(x)
        f1 = self.model.trunk_output[0](f0)
        f2 = self.model.trunk_output[1](f1)
        f3 = self.model.trunk_output[2](f2)
        # f4 = self.model.trunk_output[3](f3)
        # print(f0.shape, f1.shape, f2.shape, f3.shape, f4.shape)

        pos_enc = self.positional_encoding.expand(BS, -1, -1, -1)
        x3 = self.f3_encoder(torch.cat((f3, pos_enc), dim=1))
        return x3


class ConvPEBlock(nn.Module):
    """
    regnet x anystage block with positional encoding added.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bev_shape: Tuple[int, int],
        norm: Type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        self.register_buffer(
            "positional_encoding", positional_encoding(*bev_shape), persistent=False
        )
        self.decode = models.regnet.AnyStage(
            in_ch + 6,
            out_ch,
            stride=1,
            depth=4,
            block_constructor=models.regnet.ResBottleneckBlock,
            norm_layer=norm,
            activation_layer=nn.ReLU,
            group_width=out_ch,  # regnet_x_3_2gf
            bottleneck_multiplier=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            (
                self.positional_encoding.expand(len(x), -1, -1, -1),
                x,
            ),
            dim=1,
        )
        return self.decode(x)
