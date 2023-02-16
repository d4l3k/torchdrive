from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple, TypeVar, Union

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
    long_cam_T: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    def batch_size(self) -> int:
        return self.weight.numel()

    def to(self, device: torch.device) -> "Batch":
        """
        returns a copy of batch that's been transferred to the specified device.
        """
        return Batch(
            **{
                field.name: transfer(field.name, getattr(self, field.name), device)
                for field in fields(Batch)
            }
        )

    def split(self, split_size: int) -> List["Batch"]:
        """
        Splits the batch into `split_size` sized pieces.
        """
        out = []
        for field in fields(Batch):
            original = getattr(self, field.name)
            parts = split(original, split_size)
            print(
                field.name,
                len(parts),
                len(original),
                original.__class__,
                parts.__class__,
            )
            for i, p in enumerate(parts):
                if len(out) <= i:
                    out.append({})
                out[i][field.name] = p
        return [Batch(*g) for g in out]


def dummy_item() -> Batch:
    N = 2
    color = {}
    cams = ["left", "right"]
    for cam in cams:
        for i in range(N):
            color[cam, i] = torch.rand(3, 48, 64)
    return Batch(
        weight=torch.rand(1)[0],
        distances=torch.rand(N),
        cam_T=torch.rand(N, 4, 4),
        long_cam_T=torch.rand(9 * 3, 4, 4),
        frame_T=torch.rand(N, 4, 4),
        K={cam: torch.rand(4, 4) for cam in cams},
        T={cam: torch.rand(4, 4) for cam in cams},
        color=color,
        mask={cam: torch.rand(1, 48, 64) for cam in cams},
    )


def dummy_batch() -> Batch:
    BS = 2
    out = collate([dummy_item()] * BS)
    assert out is not None
    return out


def _collate_long_cam_T(
    tensors: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lens = torch.tensor([t.size(0) for t in tensors])
    # pyre-fixme[9]: int
    max_len: int = lens.amax().item()
    orig_max_len = max_len
    out = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    mask = torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
    assert out.shape[:2] == mask.shape, (out.shape, mask.shape)

    return (out, mask, lens)


_COLLATE_FIELDS = {
    "long_cam_T": _collate_long_cam_T,
}


def collate(
    batch: Union[List[Optional[Batch]], List[Batch]], strict: bool = True
) -> Optional[Batch]:
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
    """
    transfers the provided object to the specified device.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, list):
        return [transfer(k, i, device=device) for i in x]
    if isinstance(x, tuple):
        return tuple(transfer(k, i, device=device) for i in x)
    if isinstance(x, dict):
        return {key: transfer(k, value, device=device) for key, value in x.items()}
    return x


def split(x: T, split_size: int) -> List[T]:
    """
    split split_size the object into `split_size` pieces.
    """
    if isinstance(x, torch.Tensor):
        return torch.split(x, split_size)
    elif isinstance(x, dict):
        groups = []
        for key, value in x.items():
            parts = split(value, split_size)
            for i, v in enumerate(parts):
                if len(groups) <= i:
                    groups.append({})
                groups[i][key] = v
        return groups
    elif isinstance(x, tuple):
        groups = []
        for value in x:
            parts = split(value, split_size)
            for i, v in enumerate(parts):
                if len(groups) <= i:
                    groups.append([])
                groups[i].append(v)
        return [tuple(g) for g in groups]
    raise ValueError(f"can't split {x}")
