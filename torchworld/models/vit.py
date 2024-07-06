from typing import Tuple

import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class MaskViT(nn.Module):
    def __init__(
        self,
        attention_dropout: float,
        cam_shape: Tuple[int, int],
        dim: int,
    ) -> None:
        super().__init__()

        self.cam_shape = cam_shape
        self.encoder = vit_b_16(
            weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,
            progress=True,
            attention_dropout=attention_dropout,
        )
        self.encoder.encoder.pos_embedding = nn.Parameter(
            torch.empty(
                1,
                self.encoder.hidden_dim,
                cam_shape[0] // self.encoder.patch_size,
                cam_shape[1] // self.encoder.patch_size,
            ).normal_(std=0.02)
        )
        self.project = nn.Linear(self.encoder.hidden_dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Reshape and permute the input tensor
        n, c, h, w = x.shape
        p = self.encoder.patch_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.encoder.conv_proj(x)

        x = x + self.encoder.encoder.pos_embedding

        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, mask.sum())
        x = x[:, :, mask]

        # (n, hidden_dim, mask_elems) -> (n, mask_elems, hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        x = self.encoder.encoder.ln(self.encoder.encoder.layers(self.encoder.encoder.dropout(x)))
        return self.project(x)
