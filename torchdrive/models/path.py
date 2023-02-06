from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchdrive.models.transformer import TransformerDecoder
from torchdrive.positional_encoding import positional_encoding


def rel_dists(series: torch.Tensor) -> torch.Tensor:
    """
    rel_dists returns the distances between each point in the series.
    """
    a = series[..., 1:]
    b = series[..., :-1]
    dists = torch.linalg.vector_norm(a - b, dim=1)
    return F.pad(dists, (1, 0))


class PathTransformer(nn.Module):
    def __init__(
        self,
        bev_shape: Tuple[int, int],
        bev_dim: int,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 3,
        pos_dim: int = 3,
    ) -> None:
        super().__init__()

        self.dim = dim

        self.register_buffer(
            "positional_encoding", positional_encoding(*bev_shape), persistent=False
        )
        self.bev_encoder = nn.Conv2d(bev_dim + 6, dim, 1)

        self.pos_encoder = nn.Conv1d(pos_dim, dim, 1)
        self.pos_decoder = nn.Conv1d(dim, pos_dim, 1)

        self.transformer = TransformerDecoder(
            dim=dim, layers=num_layers, num_heads=num_heads
        )

    def forward(self, bev: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        BS = len(bev)
        device = bev.device
        dtype = bev.dtype

        bev = torch.cat(
            (
                self.positional_encoding.expand(BS, -1, -1, -1),
                bev,
            ),
            dim=1,
        )
        bev = self.bev_encoder(bev).reshape(BS, self.dim, -1).permute(0, 2, 1)
        positions = self.pos_encoder(positions).permute(0, 2, 1)

        positions = self.transformer(positions, bev).permute(0, 2, 1)
        return self.pos_decoder(positions)
