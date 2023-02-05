from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple, TypeVar

import torch
from torch.utils.data import default_collate


@dataclass(frozen=True)
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
    # sequential cam_T only aligned with the start frames extending into the
    # future
    long_cam_T: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            **{
                field.name: transfer(field.name, getattr(self, field.name), device)
                for field in fields(Batch)
            }
        )


def dummy_batch() -> Batch:
    N = 2
    BS = 2
    color = {}
    cams = ["left", "right"]
    for cam in cams:
        for i in range(N):
            color[cam, i] = torch.rand(BS, 3, 48, 64)
    return Batch(
        weight=torch.rand(BS),
        distances=torch.rand(BS, N),
        cam_T=torch.rand(BS, N, 4, 4),
        long_cam_T=torch.rand(BS, 9 * 3, 4, 4),
        frame_T=torch.rand(BS, N, 4, 4),
        K={cam: torch.rand(BS, 4, 4) for cam in cams},
        T={cam: torch.rand(BS, 4, 4) for cam in cams},
        color=color,
        mask={cam: torch.rand(BS, 1, 48, 64) for cam in cams},
    )


def _collate_long_cam_T(tensors: List[torch.Tensor]) -> torch.Tensor:
    seq_len = min(t.size(0) for t in tensors)
    return default_collate([t[:seq_len] for t in tensors])


_COLLATE_FIELDS = {
    "long_cam_T": _collate_long_cam_T,
}


def collate(batch: List[Optional[Batch]], strict: bool = True) -> Optional[Batch]:
    """
    collate merges a provided set of single example batches and allows some
    examples to be discarded if there's corrupted data.
    """
    BS = len(batch)
    batch = [item for item in batch if item is not None]
    if len(batch) <= BS / 2:
        if strict:
            raise RuntimeError("not enough data in batch")
        return None

    return Batch(
        **{
            field.name: _COLLATE_FIELDS.get(field.name, default_collate)(
                [getattr(b, field.name) for b in batch]
            )
            for field in fields(Batch)
        }
    )


def nonstrict_collate(batch: List[Optional[Batch]]) -> Optional[Batch]:
    """
    collate with strict=False so it returns empty batches if the batch size is
    too small.
    """
    return collate(batch, strict=False)


T = TypeVar("T")


def transfer(k: str, x: T, device: torch.device) -> T:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, list):
        return [transfer(k, i, device=device) for i in x]
    if isinstance(x, dict):
        return {key: transfer(k, value, device=device) for key, value in x.items()}
    return x
