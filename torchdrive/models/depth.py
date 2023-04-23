from typing import Tuple

import torch
from torch import nn

from torchdrive.models.bev import BEVUpsampler


class DepthDecoder(nn.Module):
    """
    Used for decoding disparity from a camera embedding.
    """

    def __init__(
        self,
        num_upsamples: int,
        cam_shape: Tuple[int, int],
        dim: int,
        num_classes: int = 0,
        final_dim: int = 32,
    ) -> None:
        super().__init__()

        self.upsample = nn.Sequential(
            BEVUpsampler(
                num_upsamples=num_upsamples,
                bev_shape=cam_shape,
                dim=dim,
                output_dim=final_dim,
            ),
            nn.Conv2d(final_dim, 1 + 3 + num_classes, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [BS, dim, *cam_shape]
        Returns:
            disp: [BS, *(cam_shape * 2**num_upsamples)]
            vel: [BS, 3, *(cam_shape * 2**num_upsamples)]
        """
        out = self.upsample(x)
        disp = out[:, 0]
        vel = out[:, 1:4]
        semantic = out[:, 4:]
        return disp, vel, semantic
