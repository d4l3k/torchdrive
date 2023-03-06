import math
from typing import Optional

import torch


def positional_encoding(
    x: int, y: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    This positional is designed to encode x/y positions for use in encoding
    images and BEV grids.

    This is a combination of sin/cos position encodings as well as alternating
    1/-1 encoding on x/y axis which is an attempt to encourage distinctness when upsampling.

    Channels:
    * sin(x)
    * cos(x)
    * sin(y)
    * cos(y)
    * alternating 1/-1 x-axis
    * alternating 1/-1 y-axis
    """
    positional_encoding = torch.zeros((1, 6, x, y), device=device)
    x_range = torch.arange(0, x, device=device) / (x * 2 * math.pi)
    y_range = torch.arange(0, y, device=device) / (y * 2 * math.pi)
    positional_encoding[0, 0, :, :] = torch.sin(x_range).unsqueeze(1)
    positional_encoding[0, 1, :, :] = torch.cos(x_range).unsqueeze(1)
    positional_encoding[0, 2, :, :] = torch.sin(y_range).unsqueeze(0)
    positional_encoding[0, 3, :, :] = torch.cos(y_range).unsqueeze(0)
    positional_encoding[0, 4, ::2, :] = 1
    positional_encoding[0, 4, 1::2, :] = -1
    positional_encoding[0, 5, :, ::2] = 1
    positional_encoding[0, 5, :, 1::2] = -1
    return positional_encoding


def sin_cos_enc(
    seq_len: int, dim: int, device: torch.device, dtype: torch.dtype = torch.float
) -> torch.Tensor:
    """
    Creates a 1d sin/cos position encoding.

    Returns: (seq_len, dim)
    """
    pos = (
        torch.arange(0, seq_len, device=device, dtype=dtype).unsqueeze(1).repeat(1, dim)
    )
    dim_arr = (
        torch.arange(0, dim, device=device, dtype=dtype).unsqueeze(0).repeat(seq_len, 1)
    )
    # pyre-fixme[6]: expected Tensor but got float
    div = torch.exp(-math.log(10000) * (2 * (dim_arr // 2) / dim))
    pos *= div
    pos[:, 0::2] = torch.sin(pos[:, 0::2])
    pos[:, 1::2] = torch.cos(pos[:, 1::2])
    return pos


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
