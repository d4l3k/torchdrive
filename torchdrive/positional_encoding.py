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


def sequence_encoding(x: torch.Tensor) -> torch.Tensor:
    """
    Simple fixed sin/cos encoding added to the sequence. Good for 1d language
    style tasks.

    Adapted from:
    https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/sine.py
    BSD style license.
    Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
    """

    dtype = x.dtype
    BS, seq_len, dim_model = x.shape
    pos = (
        torch.arange(0, seq_len, device=x.device, dtype=dtype)
        .unsqueeze(1)
        .repeat(1, dim_model)
    )
    dim = (
        torch.arange(0, dim_model, device=x.device, dtype=dtype)
        .unsqueeze(0)
        .repeat(seq_len, 1)
    )
    # pyre-fixme[6]: expected Tensor but got float
    div = torch.exp(-math.log(10000) * (2 * (dim // 2) / dim_model))
    pos *= div
    pos[:, 0::2] = torch.sin(pos[:, 0::2])
    pos[:, 1::2] = torch.cos(pos[:, 1::2])

    output = x.unsqueeze(-1) if x.ndim == 2 else x

    return output + pos.unsqueeze(0)
