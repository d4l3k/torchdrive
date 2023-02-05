import torch
from torch import nn

from torchdrive.attention import attention
from torchdrive.models.regnet import resnet_init
from torchdrive.positional_encoding import sequence_encoding


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout_p: float = 0.1) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.dropout_p = dropout_p

        self.query_encoder = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
        )
        self.kv_encoder = nn.Sequential(
            nn.Conv1d(dim, 2 * dim, 1),
        )
        resnet_init(self.query_encoder)
        resnet_init(self.kv_encoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BS, seq_len, dim = x.shape
        x = x.permute(0, 2, 1)
        kv = self.kv_encoder(x).permute(0, 2, 1)
        q = self.query_encoder(x).permute(0, 2, 1)

        return attention(
            q,
            kv,
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=True,
        )


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout_p: float = 0.1) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.dropout_p = dropout_p

        self.query_encoder = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
        )
        self.kv_encoder = nn.Sequential(
            nn.Conv1d(dim, 2 * dim, 1),
        )
        resnet_init(self.query_encoder)
        resnet_init(self.kv_encoder)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        kv = self.kv_encoder(kv.permute(0, 2, 1)).permute(0, 2, 1)
        q = self.query_encoder(q.permute(0, 2, 1)).permute(0, 2, 1)

        return attention(
            q,
            kv,
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_p=self.dropout_p if self.training else 0.0,
        )


class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout_p: float = 0.1) -> None:
        super().__init__()

        self.dim = dim
        hidden_dim = 2 * dim

        self.self_attn = SelfAttention(
            dim=dim, num_heads=num_heads, dropout_p=dropout_p
        )
        self.ln1 = nn.LayerNorm(dim)

        self.cross_attn = CrossAttention(
            dim=dim, num_heads=num_heads, dropout_p=dropout_p
        )
        self.ln2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, dim, 1),
        )
        self.ln3 = nn.LayerNorm(dim)

        resnet_init(self.ffn)

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.self_attn(x))
        x = self.ln2(x + self.cross_attn(x, cross))
        x = self.ln3(x + self.ffn(x.permute(0, 2, 1)).permute(0, 2, 1))
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
