import torch
from torch import nn
from torchvision import models

from torchdrive.models.regnet import ResBottleneckBlock3d


class ResUpsample3d(nn.Module):
    """
    This upsamples a 3D grid using a regnet X style convolution blocks but in
    3d.
    """

    def __init__(
        self, num_upsamples: int, dim: int, min_dim: int, depth: int = 4
    ) -> None:
        super().__init__()

        blocks = []
        in_ch = dim
        for i in range(num_upsamples):
            out_ch = max(in_ch // 2, min_dim)
            blocks += [
                models.regnet.AnyStage(
                    in_ch,
                    out_ch,
                    stride=1,
                    depth=depth,
                    block_constructor=ResBottleneckBlock3d,
                    norm_layer=nn.BatchNorm3d,
                    activation_layer=nn.ReLU,
                    group_width=out_ch,  # disable groups
                    bottleneck_multiplier=1.0,
                ),
                nn.Upsample(scale_factor=(2, 2, 2)),
            ]
            in_ch = out_ch
        self.upsample = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)
