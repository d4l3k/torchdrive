import math
from typing import Callable, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from torchdrive.amp import autocast
from torchdrive.models.mlp import MLP
from torchdrive.models.regnet import ConvPEBlock
from torchdrive.models.transformer import StockTransformerDecoder, transformer_init
from torchdrive.positional_encoding import apply_sin_cos_enc2d

MAX_POS = 100  # meters from origin


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
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        super().__init__()

        self.dim = dim

        self.bev_encoder = nn.Conv2d(bev_dim, dim, 1)

        self.bev_project = compile_fn(ConvPEBlock(bev_dim, bev_dim, bev_shape, depth=1))

        self.pos_encoder = compile_fn(
            nn.Sequential(
                nn.Linear(pos_dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
            )
        )
        self.pos_decoder = compile_fn(
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, pos_dim),
            )
        )

        static_features = 3 * 3
        self.static_encoder = compile_fn(MLP(static_features, dim, dim))

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

        num_pos = positions.size(2)
        # static features (speed)
        if num_pos > 1:
            speed = positions[:, :, 1] - positions[:, :, 0]
        else:
            speed = positions[:, :, 0] * 0

        start_position = positions[:, :, 0]

        # feed it the target end position with random jitter added to avoid
        # overfitting
        end_jitter = 5
        end_position = positions[:, :, -1]
        end_jitter = (torch.rand_like(end_position) * 2 - 1) * end_jitter
        end_position = end_position + end_jitter

        with autocast():
            static_feats = torch.cat(
                (speed, start_position, end_position), dim=1
            ).unsqueeze(-1)
            static = self.static_encoder(static_feats).permute(0, 2, 1)

            # bev features
            bev = self.bev_project(bev)
            bev = apply_sin_cos_enc2d(bev)
            bev = self.bev_encoder(bev).flatten(-2, -1).permute(0, 2, 1)

            # cross attention features to decode
            cross_feats = torch.cat((bev, static), dim=1)

            out_positions = self.transformer(position_emb, cross_feats)

        pred_pos = self.pos_decoder(out_positions.float())
        pred_pos = pred_pos.permute(0, 2, 1)  # [bs, 3, n]
        # convert to bounded x/y/z coords
        pred_pos = (pred_pos.sigmoid() * 2 - 1) * MAX_POS

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
