from dataclasses import dataclass, fields
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader, default_collate


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
    # time for each frame in seconds, monotonically increasing, can be starting at any point
    frame_time: torch.Tensor
    # per camera intrinsics, normalized
    K: Dict[str, torch.Tensor]
    # car to camera local translation matrix, extrinsics
    T: Dict[str, torch.Tensor]
    # per camera and frame color data [BS, N, 3, H, W]
    color: Dict[str, torch.Tensor]
    # per camera mask
    mask: Dict[str, torch.Tensor]
    # sequential cam_T only aligned with the start frames extending into the
    # future
    long_cam_T: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    global_batch_size: int = 1

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
        BS = self.batch_size()
        parts = BS // split_size
        if BS % split_size != 0:
            parts += 1
        for i in range(parts):
            out.append({"global_batch_size": self.global_batch_size})
        for field in fields(Batch):
            name = field.name
            if name == "global_batch_size":
                continue
            original = getattr(self, name)
            parts = split(original, split_size)
            for i, p in enumerate(parts):
                out[i][name] = p
        return [Batch(**g) for g in out]


def dummy_item() -> Batch:
    N = 3
    color = {}
    cams = ["left", "right"]
    for cam in cams:
        color[cam] = torch.rand(N, 3, 48, 64)
    return Batch(
        weight=torch.rand(1)[0],
        distances=torch.rand(N),
        cam_T=torch.rand(N, 4, 4),
        long_cam_T=torch.rand(9 * 3, 4, 4),
        frame_T=torch.rand(N, 4, 4),
        frame_time=torch.arange(N, dtype=torch.float),
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


def _collate_weight(
    tensors: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.stack(tensors)
    # normalize to sum to 1
    weights /= weights.sum() + 1e-8

    return weights


_COLLATE_FIELDS: Mapping[str, Callable[[object], object]] = {
    "long_cam_T": _collate_long_cam_T,
    "weight": _collate_weight,
    "global_batch_size": sum,
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
            raise RuntimeError(f"not enough data in batch, BS={BS}")
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


class TransferCollator:
    """
    TransferCollator takes in a torch DataLoader with a batch size of 1 and
    buffers, transfers and collates into larger batches.

    This overlaps the data transfer of the batches with compute by
    starting the async transfer of the next batch before returning the current
    one. This should produce better utilization since it overlaps the H2D and
    compute channels.

    Unlike the normal dataloader collate behavior, collate runs on the target
    device after data transfer.

    If the last batch is smaller than batch_size it is discarded.
    """

    def __init__(
        self,
        dataloader: DataLoader[Batch],
        batch_size: int,
        device: torch.device,
        buffer_factor: int = 2,
    ) -> None:
        self.dataloader = dataloader
        self.futures: List[Batch] = []
        self.device = device
        self.batch_size = batch_size
        self.buffer_factor = buffer_factor
        self.iter: Optional[Iterator[Batch]] = None

        self.pool = ThreadPoolExecutor(max_workers=1)

    def __iter__(self) -> "TransferCollator":
        self.iter = iter(self.dataloader)
        self.futures = []
        return self

    @contextmanager
    def _stream_sync(self):
        """
        _stream_sync creates a new CUDA stream to run and then synchronizes at
        the end.
        """
        if self.device.type == "cuda":
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                yield
            s.synchronize()
        else:
            yield

    def _get_batch(self) -> Batch:
        with self._stream_sync():
            frames = []
            while len(frames) < self.batch_size:
                frame = next(self.iter)
                if frame is None:
                    continue
                frame = frame.to(self.device)
                frames.append(frame)
            return collate(frames)

    def __next__(self) -> Batch:
        it = self.iter
        assert it is not None
        while len(self.futures) < self.buffer_factor:
            self.futures.append(self.pool.submit(self._get_batch))

        batch: Optional[Batch] = self.futures.pop(0).result()

        assert batch is not None, "collate returned None"
        return batch

    def __len__(self) -> int:
        return len(self.dataloader) // self.batch_size
