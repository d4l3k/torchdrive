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
            nn.Conv2d(final_dim, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x).squeeze(1)
