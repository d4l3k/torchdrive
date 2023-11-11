from typing import List, Tuple

import torch
from torch import nn
from torchvision import models

from torchworld.models.deformable_transformer import (
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
)
from torchworld.ops.ms_deformable_attention import prepare_src
from torchworld.positional_encoding import LearnedPositionalEncoding2d

from torchdrive.models.mlp import MLP


class DetDeformableTransformerDecoder(nn.Module):
    """
    BEV based detection decoder. Consumes a BEV grid and generates detections
    using a deformable detr style transformer.
    """

    def __init__(
        self,
        dim: int,
        num_queries: int,
        bev_shape: Tuple[int, int],
        num_heads: int = 8,
        num_classes: int = 10,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_levels = num_levels

        self.num_queries = num_queries

        self.query_embed = nn.Embedding(num_queries, dim)
        self.reference_points_project = nn.Linear(dim, 2)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=dim,
            d_ffn=dim_feedforward,
            n_heads=num_heads,
            n_levels=num_levels,
            n_points=num_points,
            dropout=dropout,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.bbox_decoder = MLP(dim, dim, 9, num_layers=4)
        self.class_decoder = nn.Conv1d(dim, num_classes + 1, 1)

        bev_encoders = []
        for i in range(num_levels - 1, -1, -1):
            h, w = bev_shape
            bev_encoders.append(
                nn.Sequential(
                    models.regnet.AnyStage(
                        dim,
                        dim,
                        stride=1,
                        depth=4,
                        block_constructor=models.regnet.ResBottleneckBlock,
                        norm_layer=nn.BatchNorm2d,
                        activation_layer=nn.ReLU,
                        group_width=dim,
                        bottleneck_multiplier=1.0,
                    ),
                    nn.Conv2d(dim, dim, 1),
                    LearnedPositionalEncoding2d((h * 2**i, w * 2**i), dim),
                )
            )
        self.bev_encoders = nn.ModuleList(bev_encoders)

    def forward(self, grids: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            feats: [(BS, dim, x, y), ...]
        Returns:
            classes: logits (BS, num_queries, 11)
            bboxes: 0-1 (BS, num_queries, 9)
        """
        BS = len(grids[0])

        grids = [self.bev_encoders[i](grid) for i, grid in enumerate(grids)]

        src, src_spatial_shapes, src_level_start_index, src_valid_ratios = prepare_src(
            grids
        )

        target = self.query_embed.weight.expand(BS, -1, -1)

        # reference points are in range [0, 1], fixed position per query
        reference_points = self.reference_points_project(self.query_embed.weight)
        reference_points = reference_points.sigmoid()
        reference_points = reference_points.expand(BS, -1, -1)

        out, _ = self.decoder(
            tgt=target,
            reference_points=reference_points,
            src=src,
            src_spatial_shapes=src_spatial_shapes,
            src_level_start_index=src_level_start_index,
            src_valid_ratios=src_valid_ratios,
        )

        out = out.permute(0, 2, 1)  # (BS, ch, num_queries)

        bboxes = self.bbox_decoder(out).permute(0, 2, 1)
        classes = self.class_decoder(out).permute(0, 2, 1)  # logits

        bboxes = bboxes.float().sigmoid()  # normalized 0 to 1

        return classes, bboxes
