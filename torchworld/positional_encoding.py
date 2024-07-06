import math
from typing import Optional, Tuple

import torch
from torch import nn


def sin_cos_enc(
    seq_len: int, dim: int, device: torch.device, dtype: torch.dtype = torch.float
) -> torch.Tensor:
    """
    Creates a 1d sin/cos position encoding.

    Returns: (seq_len, dim)
    """

    if dim % 2 != 0:
        raise ValueError(f"dim must be a multiple of 2, got {dim}")

    position = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=dtype, device=device) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim, dtype=dtype, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def apply_sin_cos_enc1d(x: torch.Tensor) -> torch.Tensor:
    """
    Applies a 2d sin/cos position encoding to the tensor.

    Input: (bs, seq_len, dim)
    Returns: (bs, seq_len, dim)
    """
    bs, seq_len, dim = x.shape
    return x + sin_cos_enc(seq_len, dim=dim, device=x.device, dtype=x.dtype)


def sin_cos_enc2d(
    h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype = torch.float
) -> torch.Tensor:
    """
    Creates a 2d sin/cos position encoding.

    Returns: (dim, h, w)
    """
    assert dim % 2 == 0, "dimensions must be a multiple of 2 {dim}"
    y_enc = sin_cos_enc(h, dim // 2, device=device, dtype=dtype)
    x_enc = sin_cos_enc(w, dim // 2, device=device, dtype=dtype)

    y_enc = y_enc.permute(1, 0).unsqueeze(2).expand(-1, -1, w)
    x_enc = x_enc.permute(1, 0).unsqueeze(1).expand(-1, h, -1)
    return torch.cat((x_enc, y_enc), dim=0)


def apply_sin_cos_enc2d(x: torch.Tensor) -> torch.Tensor:
    """
    Applies a 2d sin/cos position encoding to the tensor.

    Input: (bs, dim, h, w)
    Returns: (bs, dim, h, w)
    """
    bs, dim, h, w = x.shape
    return x + sin_cos_enc2d(h=h, w=w, dim=dim, device=x.device, dtype=x.dtype)


def sequence_encoding(x: torch.Tensor) -> torch.Tensor:
    """
    Simple fixed sin/cos encoding added to the sequence. Good for 1d language
    style tasks.

    Adapted from:
    https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/sine.py
    BSD style license.
    Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
    """

    BS, seq_len, dim_model = x.shape
    pos = sin_cos_enc(seq_len, dim_model, device=x.device, dtype=x.dtype)

    output = x.unsqueeze(-1) if x.ndim == 2 else x

    return output + pos.unsqueeze(0)


class LearnedPositionalEncodingSeq(nn.Module):
    def __init__(self, max_seq_len: int, dim: int) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, ch = x.shape

        assert (
            seq_len <= self.max_seq_len
        ), f"got sequence longer than max_seq_len {x.shape} {self.max_seq_len}"

        return x + self.embedding.weight[None, :seq_len, :]


class LearnedPositionalEncoding1d(nn.Module):
    def __init__(self, max_seq_len: int, dim: int) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, ch, seq_len = x.shape

        assert (
            seq_len <= self.max_seq_len
        ), f"got sequence longer than max_seq_len {x.shape} {self.max_seq_len}"

        return x + self.embedding.weight[None, :, :seq_len]


class LearnedPositionalEncoding2d(nn.Module):
    def __init__(self, shape: Tuple[int, int], dim: int) -> None:
        super().__init__()

        self.h_embedding = nn.Embedding(dim, shape[0])
        self.w_embedding = nn.Embedding(dim, shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x
            + self.h_embedding.weight[None, :, :, None]
            + self.w_embedding.weight[None, :, None, :]
        )
