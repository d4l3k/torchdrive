import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from torchdrive.amp import autocast
from torchdrive.models.mlp import ConvMLP
from torchdrive.models.regnet import ConvPEBlock
from torchdrive.models.transformer import StockTransformerDecoder, transformer_init
from torchdrive.positional_encoding import apply_sin_cos_enc2d


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

        self.bev_encoder = nn.Conv2d(bev_dim, dim, 1)

        self.bev_project = ConvPEBlock(bev_dim, bev_dim, bev_shape, depth=1)

        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, dim, bias=False),
        )
        self.pos_decoder = nn.Linear(dim, pos_dim, bias=False)

        static_features = 3
        self.static_encoder = ConvMLP(static_features, dim, dim)
        self.direction_embedding = nn.Embedding(direction_buckets, dim)

        self.transformer = StockTransformerDecoder(
            dim=dim, layers=num_layers, num_heads=num_heads
        )

        transformer_init(self)

    def forward(
        self, bev: torch.Tensor, positions: torch.Tensor, final_pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        BS = len(bev)
        device = bev.device
        dtype = bev.dtype

        position_emb = self.pos_encoder(positions.permute(0, 2, 1))
        ae_pos = self.pos_decoder(position_emb).permute(0, 2, 1)

        # direction buckets
        direction_bucket = pos_to_bucket(final_pos, buckets=self.direction_buckets)
        direction_feats = self.direction_embedding(direction_bucket).unsqueeze(1)

        # static features (speed)
        speed = positions[:, :, 1] - positions[:, :, 0]

        with autocast():
            static = self.static_encoder(speed.unsqueeze(-1)).permute(0, 2, 1)

            # bev features
            bev = self.bev_project(bev)
            bev = apply_sin_cos_enc2d(bev)
            bev = self.bev_encoder(bev).flatten(-2, -1).permute(0, 2, 1)

            # cross attention features to decode
            cross_feats = torch.cat((bev, static, direction_feats), dim=1)

            out_positions = self.transformer(position_emb, cross_feats)

        pred_pos = self.pos_decoder(out_positions.float()).permute(0, 2, 1)

        return pred_pos, ae_pos

    @staticmethod
    def infer(
        m: nn.Module,
        bev: torch.Tensor,
        seq: torch.Tensor,
        final_pos: torch.Tensor,
        n: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        infer runs the inference in an autoregressive manner.
        """
        for i in range(n):
            out, _ = m(bev, seq, final_pos)
            seq = torch.cat((seq, out[..., -1:]), dim=-1)
        return seq
