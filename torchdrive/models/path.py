import math
from typing import Callable, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from torchworld.positional_encoding import (
    LearnedPositionalEncoding2d,
    LearnedPositionalEncodingSeq,
)

from torchdrive.models.mlp import MLP
from torchdrive.models.transformer import StockTransformerDecoder, transformer_init


class XYEncoder(nn.Module):
    def __init__(self, num_buckets: int, max_dist: float) -> None:
        super().__init__()

        self.num_buckets = num_buckets
        self.max_dist = max_dist

    def encode_labels(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments
        ---------
        xy: [bs, 2, seq_len]

        Returns
        -------
        Tuple of x coordinates and y coordinates with their discretized labels.
        """
        x = xy[:, 0]
        y = xy[:, 1]

        x = (
            (((x / (2 * self.max_dist)) + 0.5) * self.num_buckets)
            .long()
            .clamp(0, self.num_buckets - 1)
        )
        y = (
            (((y / (2 * self.max_dist)) + 0.5) * self.num_buckets)
            .long()
            .clamp(0, self.num_buckets - 1)
        )
        return x, y

    def encode_one_hot(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = self.encode_labels(xy)
        x = F.one_hot(x, self.num_buckets).float()
        y = F.one_hot(y, self.num_buckets).float()
        return torch.cat((x, y), dim=2).permute(0, 2, 1)

    def decode(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Decodes from logit/probabilities one hot encoding.

        Returns
        -------
        xy: [bs, 2, seq_len]
        """
        x = xy[:, : self.num_buckets]
        y = xy[:, self.num_buckets :]
        x = ((x.argmax(dim=1).float() / self.num_buckets) - 0.5) * (2 * self.max_dist)
        y = ((y.argmax(dim=1).float() / self.num_buckets) - 0.5) * (2 * self.max_dist)
        return torch.stack((x, y), dim=1)

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = predicted[:, : self.num_buckets]
        y = predicted[:, self.num_buckets :]
        xl, yl = self.encode_labels(target)
        return F.cross_entropy(x, xl, reduction="none") + F.cross_entropy(
            y, yl, reduction="none"
        )


class PathSigmoidMeters(nn.Module):
    def __init__(self, max_pos: float = 100.0) -> None:
        super().__init__()

        self.max_pos = max_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # convert to bounded x/y/z coords
        return (x.float().sigmoid() * 2 - 1) * self.max_pos


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


class PathAutoRegressiveTransformer(nn.Module):
    def __init__(
        self,
        bev_shape: Tuple[int, int],
        bev_dim: int,
        dim: int,
        max_seq_len: int,
        num_heads: int = 8,
        num_layers: int = 3,
        pos_dim: int = 3,
        final_jitter: float = 0.0,
        dropout: float = 0.1,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.final_jitter = final_jitter

        self.bev_encoder = nn.Sequential(
            models.regnet.AnyStage(
                bev_dim,
                bev_dim,
                stride=1,
                depth=4,
                block_constructor=models.regnet.ResBottleneckBlock,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU,
                group_width=bev_dim,
                bottleneck_multiplier=1.0,
            ),
            nn.Conv2d(bev_dim, dim, 1),
            LearnedPositionalEncoding2d(bev_shape, dim),
        )

        # self.bev_project = compile_fn(ConvPEBlock(bev_dim, bev_dim, bev_shape, depth=1))

        # pyre-fixme[4]: Attribute must be annotated.
        self.pos_encoder = compile_fn(
            nn.Sequential(
                nn.Linear(pos_dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim),
                LearnedPositionalEncodingSeq(max_seq_len, dim),
            )
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.pos_decoder = compile_fn(
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, pos_dim),
            )
        )

        static_features = 2 * pos_dim
        # pyre-fixme[4]: Attribute must be annotated.
        self.static_encoder = compile_fn(MLP(static_features, dim, dim, num_layers=3))

        self.transformer = StockTransformerDecoder(
            dim=dim,
            layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        transformer_init(self)

    def forward(
        self,
        bev: torch.Tensor,
        positions: torch.Tensor,
        final_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        BS = len(bev)
        device = bev.device
        dtype = bev.dtype

        position_emb = self.pos_encoder(positions.permute(0, 2, 1))
        ae_pos = self.pos_decoder(position_emb).permute(0, 2, 1)

        num_pos = positions.size(2)
        start_position = positions[:, :, 0]

        # feed it the target end position with random jitter added to avoid
        # overfitting
        # if self.training:
        #    final_jitter = (torch.rand_like(final_pos) * 2 - 1) * self.final_jitter
        #    final_pos = final_pos + final_jitter

        static_feats = torch.cat((start_position, final_pos), dim=1).unsqueeze(-1)
        static = self.static_encoder(static_feats).permute(0, 2, 1)

        # bev features
        bev = self.bev_encoder(bev).flatten(-2, -1).permute(0, 2, 1)

        # cross attention features to decode
        cross_feats = torch.cat((bev, static), dim=1)

        out_positions = self.transformer(position_emb, cross_feats)

        pred_pos = self.pos_decoder(out_positions.float()).permute(
            0, 2, 1
        )  # [bs, 3, n]

        return pred_pos, ae_pos

    @staticmethod
    def infer(
        xy_encoder: XYEncoder,
        m: nn.Module,
        bev: torch.Tensor,
        seq: torch.Tensor,
        final_pos: torch.Tensor,
        n: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """
        infer runs the inference in an autoregressive manner.
        """
        final_pos_one_hot = xy_encoder.encode_one_hot(final_pos.unsqueeze(2)).squeeze(2)
        seq = seq[:, :2]  # switch to xy
        for i in range(n):
            # add dummy item on end
            inp = torch.cat((seq, torch.zeros_like(seq[..., -1:])), dim=-1)
            inp_one_hot = xy_encoder.encode_one_hot(inp)
            out, _ = m(bev, inp_one_hot, final_pos_one_hot)
            out = xy_encoder.decode(out)
            seq = torch.cat((seq, out[..., -1:]), dim=-1)
        return seq

    def set_dropout(self, dropout: float) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.MultiheadAttention)):
                m.dropout = dropout


class PathOneShotTransformer(nn.Module):
    def __init__(
        self,
        bev_shape: Tuple[int, int],
        bev_dim: int,
        dim: int,
        max_seq_len: int,
        static_features: int,
        num_heads: int = 8,
        num_layers: int = 3,
        pos_dim: int = 3,
        final_jitter: float = 0.0,
        dropout: float = 0.1,
        num_queries: int = 100,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.final_jitter = final_jitter
        self.pos_dim = pos_dim
        self.max_seq_len = max_seq_len

        self.query_embed = nn.Embedding(num_queries, dim)

        self.bev_encoder = nn.Sequential(
            models.regnet.AnyStage(
                bev_dim,
                bev_dim,
                stride=1,
                depth=4,
                block_constructor=models.regnet.ResBottleneckBlock,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU,
                group_width=bev_dim,
                bottleneck_multiplier=1.0,
            ),
            nn.Conv2d(bev_dim, dim, 1),
            LearnedPositionalEncoding2d(bev_shape, dim),
        )

        decoder_dim = num_queries * dim
        out_dim = max_seq_len * pos_dim
        # pyre-fixme[4]: Attribute must be annotated.
        self.pos_decoder = compile_fn(
            nn.Sequential(
                # nn.Linear(decoder_dim, decoder_dim),
                # nn.ReLU(inplace=True),
                # nn.Linear(decoder_dim, decoder_dim),
                # nn.ReLU(inplace=True),
                nn.Linear(decoder_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim),
            )
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            dim_feedforward=dim * 4,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # pyre-fixme[4]: Attribute must be annotated.
        self.static_encoder = compile_fn(MLP(static_features, dim, dim, num_layers=3))

        transformer_init(self)

    def forward(
        self,
        bev: torch.Tensor,
        positions: torch.Tensor,
        static_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        BS = len(bev)
        device = bev.device
        dtype = bev.dtype

        num_pos = positions.size(2)

        static = self.static_encoder(static_features.unsqueeze(-1)).permute(0, 2, 1)

        queries = self.query_embed.weight + static

        # bev features
        bev = self.bev_encoder(bev).flatten(-2, -1).permute(0, 2, 1)

        # cross attention features to decode
        # cross_feats = torch.cat((bev, static), dim=1)

        out_positions = self.transformer(tgt=queries, memory=bev)

        flattened = out_positions.flatten(1, 2).float()
        pred_pos = self.pos_decoder(flattened).unflatten(
            1, (self.pos_dim, self.max_seq_len)
        )
        pred_pos = pred_pos[..., :num_pos]
        assert num_pos <= self.max_seq_len, (num_pos, self.max_seq_len)

        return pred_pos, torch.tensor(0)

    @staticmethod
    def infer(
        xy_encoder: XYEncoder,
        m: nn.Module,
        bev: torch.Tensor,
        seq: torch.Tensor,
        static_features: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """
        infer runs the inference in an autoregressive manner.
        """
        seq = seq[:, :2]  # switch to xy
        bs = seq.size(0)
        # add dummy item on end
        inp = torch.cat(
            (seq, torch.zeros(bs, 2, n - 1, device=bev.device, dtype=bev.dtype)), dim=-1
        )
        inp_one_hot = xy_encoder.encode_one_hot(inp)
        out, _ = m(bev, inp_one_hot, static_features)
        out = xy_encoder.decode(out)
        return torch.cat((seq, out), dim=-1)
