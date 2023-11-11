import math

import torch
from torch import nn

from torchdrive.attention import attention
from torchdrive.positional_encoding import sequence_encoding


class MultiHeadAttention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int = 8, dropout_p: float = 0.1, causal: bool = False
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.dropout_p = dropout_p
        self.causal = causal

        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim),
        )
        self.kv_encoder = nn.Sequential(
            nn.Linear(dim, 2 * dim),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        BS, seq_len, dim = q.shape
        q = self.query_encoder(q)
        kv = self.kv_encoder(kv)

        x = attention(
            q,
            kv,
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=self.causal,
        )
        return self.out_proj(x)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout_p: float = 0.1) -> None:
        super().__init__()

        self.dim = dim
        hidden_dim = 4 * dim

        self.self_attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout_p=dropout_p,
            causal=True,
        )
        self.ln1 = nn.LayerNorm(dim)

        self.cross_attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout_p=dropout_p,
            causal=False,
        )
        self.ln2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_p),
        )
        self.ln3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.self_attn(x, x))
        x = self.ln2(x + self.cross_attn(x, cross))
        x = self.ln3(x + self.ffn(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim: int, layers: int, num_heads: int) -> None:
        super().__init__()

        self.dim = dim
        self.blocks = nn.ModuleList(
            [TransformerDecoderBlock(dim, num_heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        x = sequence_encoding(x)
        for block in self.blocks:
            x = block(x, cross)
        return x


class StockTransformerDecoder(nn.Module):
    def __init__(
        self, dim: int, layers: int, num_heads: int, dropout: float = 0.1
    ) -> None:
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            dim_feedforward=dim * 4,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=layers
        )

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        x = sequence_encoding(x)

        device = x.device
        sz = x.size(1)

        mask = nn.Transformer.generate_square_subsequent_mask(sz, device=device)
        return self.transformer_decoder(
            x, cross, tgt_mask=mask, tgt_is_causal=True, memory_is_causal=False
        )


def transformer_init(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            # Note that there is no bias due to BN
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
        elif isinstance(m, nn.Conv1d):
            # Note that there is no bias due to BN
            fan_out = m.kernel_size[0] * m.out_channels
            nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
