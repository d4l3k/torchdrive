import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchdrive.amp import autocast
from torchdrive.models.mlp import ConvMLP
from torchdrive.models.regnet import ConvPEBlock
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


def pos_to_bucket(pos: torch.Tensor, buckets: int) -> torch.Tensor:
    angle = torch.atan2(pos[:, 0], pos[:, 1])
    angle /= math.pi * 2
    angle += 0.5
    return (angle * buckets).floor().int() % buckets


class PathTransformer(nn.Module):
    def __init__(
        self,
        bev_shape: Tuple[int, int],
        bev_dim: int,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 3,
        pos_dim: int = 3,
        direction_buckets: int = 14,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.direction_buckets = direction_buckets

        self.register_buffer(
            "positional_encoding", positional_encoding(*bev_shape), persistent=False
        )
        self.bev_encoder = nn.Conv2d(bev_dim + 6, dim, 1)

        self.bev_project = ConvPEBlock(bev_dim, bev_dim, bev_shape, depth=1)

        self.pos_encoder = nn.Conv1d(pos_dim, dim, 1)
        self.pos_decoder = nn.Conv1d(dim, pos_dim, 1)

        static_features = 3
        self.static_encoder = ConvMLP(static_features, dim, dim)
        self.direction_embedding = nn.Embedding(direction_buckets, dim)

        self.transformer = TransformerDecoder(
            dim=dim, layers=num_layers, num_heads=num_heads
        )

    def forward(
        self, bev: torch.Tensor, positions: torch.Tensor, final_pos: torch.Tensor
    ) -> torch.Tensor:
        BS = len(bev)
        device = bev.device
        dtype = bev.dtype

        with autocast():
            # direction buckets
            direction_bucket = pos_to_bucket(final_pos, buckets=self.direction_buckets)
            direction_feats = self.direction_embedding(direction_bucket).unsqueeze(1)

            # static features (speed)
            speed = positions[:, :, 1] - positions[:, :, 0]
            static = self.static_encoder(speed.unsqueeze(-1)).permute(0, 2, 1)

            # bev features
            bev = self.bev_project(bev)
            bev = torch.cat(
                (
                    self.positional_encoding.expand(BS, -1, -1, -1),
                    bev,
                ),
                dim=1,
            )
            bev = self.bev_encoder(bev).reshape(BS, self.dim, -1).permute(0, 2, 1)

            # cross attention features to decode
            cross_feats = torch.cat((bev, static, direction_feats), dim=1)

            positions = self.pos_encoder(positions).permute(0, 2, 1)

            out_positions = self.transformer(positions, cross_feats).permute(0, 2, 1)

        return self.pos_decoder(out_positions.float())
