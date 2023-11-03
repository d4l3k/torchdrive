import math
from collections import OrderedDict
from typing import Callable, Optional, Protocol, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torchvision.ops.misc import Conv3dNormActivation

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
        elif isinstance(m, nn.Conv3d):
            # Note that there is no bias due to BN
            fan_out = (
                m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            )
            nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
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
        use_f4: bool = True,
    ) -> None:
        super().__init__()

        self.use_f4 = use_f4

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

        if use_f4:
            self.f4_proj: nn.Module = nn.Conv2d(self.num_ch_enc[4], dim, 1)
            resnet_init(self.f4_proj)
        self.f3_proj = nn.Conv2d(self.num_ch_enc[3], dim, 1)
        resnet_init(self.f3_proj)

        self.pos_encoding = nn.Parameter(
            torch.zeros(dim, *self.output_shape, dtype=torch.float),
            requires_grad=True,
        )
        nn.init.normal_(self.pos_encoding, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BS = x.shape[0]
        # adapted from torchvision.models.Resnet
        f0 = self.model.stem(x)
        f1 = self.model.trunk_output[0](f0)
        f2 = self.model.trunk_output[1](f1)
        f3 = self.model.trunk_output[2](f2)

        out = self.f3_proj(f3)

        if self.use_f4:
            f4 = self.model.trunk_output[3](f3)
            f4 = F.interpolate(f4.float(), size=f3.shape[-2:])
            out = out + self.f4_proj(f4)

        out = out + self.pos_encoding

        return out


class ConvPEBlock(nn.Module):
    """
    regnet x anystage block with positional encoding added.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        input_shape: Tuple[int, int],
        norm: Type[nn.Module] = nn.BatchNorm2d,
        depth: int = 4,
    ) -> None:
        super().__init__()

        self.register_buffer(
            "positional_encoding", positional_encoding(*input_shape), persistent=False
        )
        self.decode = models.regnet.AnyStage(
            in_ch + 6,
            out_ch,
            stride=1,
            depth=depth,
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


class UpsamplePEBlock(nn.Module):
    """
    ConvPEBlock with upsampling added.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        input_shape: Tuple[int, int],
        norm: Type[torch.nn.modules.batchnorm.BatchNorm2d] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=(2, 2))
        self.decode = ConvPEBlock(
            in_ch=in_ch,
            out_ch=out_ch,
            input_shape=(
                input_shape[0] * 2,
                input_shape[1] * 2,
            ),
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.decode(x)


class BottleneckTransform3d(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1.
    Uses 3D convolutions.
        BSD-3-Clause license
    Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
    """

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers["a"] = Conv3dNormActivation(
            width_in,
            w_b,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        layers["b"] = Conv3dNormActivation(
            w_b,
            w_b,
            kernel_size=3,
            stride=stride,
            groups=g,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        assert not se_ratio

        layers["c"] = Conv3dNormActivation(
            w_b,
            width_out,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=None,
        )
        super().__init__(layers)


class ResBottleneckBlock3d(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform.
        Uses 3D convolutions.
        BSD-3-Clause license
    Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
    """

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj: Optional[Conv3dNormActivation] = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = Conv3dNormActivation(
                width_in,
                width_out,
                kernel_size=1,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        self.f = BottleneckTransform3d(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation: nn.Module = activation_layer(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)
