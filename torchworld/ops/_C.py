import os.path
from typing import List

import torch
from torch.utils.cpp_extension import load

TORCHWORLD_DIR: str = os.path.join(os.path.dirname(__file__), "..")

CPP_SRCS: List[str] = [
    "csrc/ms_deform_attn/vision.cpp",
    "csrc/ms_deform_attn/cpu/ms_deform_attn_cpu.cpp",
]
if torch.cuda.is_available():
    CPP_SRCS += [
        "csrc/ms_deform_attn/cuda/ms_deform_attn_cuda.cu",
    ]
INCLUDE_PATHS: List[str] = ["csrc/ms_deform_attn"]


def _add_dir(paths: List[str]) -> List[str]:
    return [os.path.join(TORCHWORLD_DIR, file) for file in paths]


torchworld_cpp: object = load(
    name="torchworld_cpp",
    sources=_add_dir(CPP_SRCS),
    extra_include_paths=_add_dir(INCLUDE_PATHS),
    verbose=True,
)
