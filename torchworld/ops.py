
import os.path
from typing import List, Any, Tuple, Optional

import torch
from torch.autograd.function import once_differentiable
from torch.autograd import Function
from torch.utils.cpp_extension import load
import torch.nn.functional as F

TORCHWORLD_DIR: str = os.path.dirname(__file__)

CPP_SRCS: List[str] = [
    "csrc/ms_deform_attn/vision.cpp",
    "csrc/ms_deform_attn/cpu/ms_deform_attn_cpu.cpp",
    "csrc/ms_deform_attn/cuda/ms_deform_attn_cuda.cu",
]
INCLUDE_PATHS: List[str] = [
    "csrc/ms_deform_attn"
]

def _add_dir(paths: List[str]) -> List[str]:
    return [
        os.path.join(TORCHWORLD_DIR, file) for file in paths
    ]

torchworld_cpp: object = load(
    name="torchworld_cpp",
    sources=_add_dir(CPP_SRCS),
    extra_include_paths=_add_dir(INCLUDE_PATHS),
    verbose=True,
)

def ms_deformable_attention(
    value: torch.Tensor, spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor, sampling_locations:
                torch.Tensor, attention_weights: torch.Tensor, im2col_step: int) -> torch.Tensor:
    """
    Multiscale deformable attention

    Arguments
    ---------
    value:
    spatial_shapes:
    level_start_index:
    sampling_locations:
    attention_weights:
    im2col_step:
    """
    if value.is_cuda:
        return _ms_deformable_attention_cuda.apply(value, spatial_shapes,
                                            level_start_index,
                                            sampling_locations,
                                            attention_weights, im2col_step)
    return _ms_deformable_attention_pytorch(value, spatial_shapes, sampling_locations,
                                            attention_weights)

# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
class _ms_deformable_attention_cuda(Function):
    @staticmethod
    # pyre-fixme[2]: Any
    # pyre-fixme[14]: overriden signature
    def forward(ctx: Any, value: torch.Tensor, spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor, sampling_locations:
                torch.Tensor, attention_weights: torch.Tensor, im2col_step:
                int) -> torch.Tensor:
        ctx.im2col_step = im2col_step
        output = torch.ops.torchworld.ms_deform_attn_forward(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    # pyre-fixme[2]: Any
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = ctx.saved_tensors

        grad_value, grad_sampling_loc, grad_attn_weight = torch.ops.torchworld.ms_deform_attn_backward(
                value, spatial_shapes, level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def _ms_deformable_attention_pytorch(value: torch.Tensor, spatial_shapes:
                                torch.Tensor, sampling_locations: torch.Tensor,
                                attention_weights: torch.Tensor) -> torch.Tensor:
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
