from typing import List, Tuple

import torch
from torch import nn
from torchvision import models

from torchdrive.models.mlp import MLP


class DetConvDecoder(nn.Module):
    """
    BEV based detection decoder. Consumes a BEV grid and generates detections
    using a 2D BEV grid.
    """

    def __init__(
        self,
        dim: int,
        bev_shape: Tuple[int, int],
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_classes = num_classes

        self.bev_encoder = nn.Sequential(
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
        )

        self.bbox_decoder = MLP(dim, dim, 9, num_layers=3, dropout=dropout)
        self.class_decoder = nn.Sequential(
            nn.Conv1d(dim, num_classes + 1, 1),
        )

    def forward(self, grids: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            feats: [(BS, dim, x, y), ...]
        Returns:
            classes: logits (BS, num_queries, 11)
            bboxes: 0-1 (BS, num_queries, 9)
        """
        BS = len(grids[0])

        # target 32x32
        out = self.bev_encoder(grids[-2])

        # flatten to 1d
        out = out.flatten(2, 3)

        bboxes = self.bbox_decoder(out).permute(0, 2, 1)
        classes = self.class_decoder(out).permute(0, 2, 1)  # logits

        bboxes = bboxes.float().sigmoid()  # normalized 0 to 1

        return classes, bboxes

    def decoder_params(self) -> List[object]:
        return list(self.bbox_decoder.parameters()) + list(
            self.class_decoder.parameters()
        )
