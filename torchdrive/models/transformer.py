import torch
from torch import nn

from torchdrive.attention import attention
from torchdrive.models.regnet import resnet_init
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
        resnet_init(self.query_encoder)
        resnet_init(self.kv_encoder)
        resnet_init(self.out_proj)

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
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_p),
        )
        self.ln3 = nn.LayerNorm(dim)

        resnet_init(self.ffn)

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
