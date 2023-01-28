from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class Batch:
    # example weight
    weight: torch.Tensor
    # per frame distance traveled in meters
    distances: torch.Tensor
    # per frame world to car translation matrix
    cam_T: torch.Tensor
    # per frame car relative translation matrix
    frame_T: torch.Tensor
    # per camera intrinsics, normalized
    K: Dict[str, torch.Tensor]
    # car to camera local translation matrix, extrinsics
    T: Dict[str, torch.Tensor]
    # per camera and frame color data
    color: Dict[Tuple[str, int], torch.Tensor]
    # per camera mask
    mask: Dict[str, torch.Tensor]


def dummy_batch() -> Batch:
    color = {}
    for cam in ["left", "right"]:
        for i in range(2):
            color[cam, i] = torch.rand(2, 3, 48, 64)
    return Batch(
        weight=torch.rand(0),
        distances=torch.rand(5),
        cam_T=torch.rand(5, 4, 4),
        frame_T=torch.rand(5, 4, 4),
        K={},
        T={},
        color=color,
        mask={},
    )
