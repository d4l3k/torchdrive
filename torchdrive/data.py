import io
import random
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from typing import (
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch

import zstd
from torch.utils.data import DataLoader, default_collate

from torchworld.structures.cameras import CamerasBase
from torchworld.structures.grid import GridImage
from torchworld.structures.points import Points

from torchdrive.render.raymarcher import CustomPerspectiveCameras


@dataclass(frozen=True)
class Batch:
    # per frame unique token, must be unique across the entire dataset
    token: List[List[str]]
    # example weight [BS]
    weight: torch.Tensor
    # per frame distance traveled in meters  [BS, num_frames]
    distances: torch.Tensor
    # per frame world to car translation matrix  [BS, num_frames, 4, 4]
    cam_T: torch.Tensor
    # per frame car relative translation matrix [BS, num_frames, 4, 4]
    frame_T: torch.Tensor
    # time for each frame in seconds, monotonically increasing, can be starting
    # at any point [BS, num_frames]
    frame_time: torch.Tensor
    # per camera intrinsics, normalized [BS, 4, 4]
    K: Dict[str, torch.Tensor]
    # per cam, camera to car translation matrix, extrinsics [BS, 4, 4]
    T: Dict[str, torch.Tensor]
    # per camera and frame color data [BS, N, 3, H, W]
    color: Dict[str, torch.Tensor]
    # per camera mask [BS, 1, h, w]
    mask: Dict[str, torch.Tensor]
    # sequential cam_T only aligned with the start frames extending into the
    # future (out, mask, lens) [BS, long_num_frames, 4, 4]
    long_cam_T: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    # Lidar transformation matrix -- car to Lidar coordinates [4, 4],
    # relative to first cam_T position
    lidar_T: torch.Tensor
    # Lidar data [BS, 4, n], channel format is [x, y, z, intensity]
    lidar: Optional[torch.Tensor] = None

    # AutoLabeler fields
    # semantic segmentation for each camera and the frames
    sem_seg: Optional[Dict[str, torch.Tensor]] = None
    # object detections for each camera and frames
    # [cam, batch, frame, class]: tensor[x1, y1, x2, y2, prob]
    det: Optional[Dict[str, List[List[List[torch.Tensor]]]]] = None

    global_batch_size: int = 1

    def batch_size(self) -> int:
        return self.weight.numel()

    def device(self) -> torch.device:
        return self.weight.device

    def camera_names(self) -> Tuple[str, ...]:
        return tuple(self.color.keys())

    def camera(self, cam: str, frame: int) -> CamerasBase:
        color = self.color[cam]
        bs = len(color)
        device = color.device
        image_size = torch.tensor([color.shape[-2:]], device=device).expand(bs, -1)
        return CustomPerspectiveCameras(
            T=self.world_to_cam(cam, frame),
            K=self.K[cam],
            image_size=image_size,
            device=device,
        )

    def grid_image(self, cam: str, frame: int) -> GridImage:
        """
        Return the camera color for the specified frame.

        Returns
        -------
        tensor [bs, nframes, 3, H, W]
        """
        color = self.color[cam][:, frame]
        return GridImage(
            data=color,
            camera=self.camera(cam, frame),
            time=self.frame_time[:, frame],
            mask=self.mask[cam],
        )

    def lidar_points(self) -> Points:
        """
        Returns the lidar data as a Points object in world coordinates.
        """
        lidar = self.lidar
        if lidar is None:
            raise ValueError("lidar must not be None")

        BS, ch, num_points = lidar.shape
        lidar_data = torch.cat((lidar[:, :3], torch.ones(BS, 1, num_points)), dim=1)
        lidar_data = self.lidar_to_world().matmul(lidar_data)
        lidar_data = lidar_data[:, :3] / lidar_data[:, 3:]
        lidar_data = torch.cat((lidar_data, lidar[:, 3:]), dim=1)

        return Points(data=lidar_data)

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
        num_parts = BS // split_size
        if BS % split_size != 0:
            num_parts += 1
        for i in range(num_parts):
            out.append({"global_batch_size": self.global_batch_size})
        for field in fields(Batch):
            name = field.name
            if name == "global_batch_size":
                continue

            original = getattr(self, name)

            if original is None:
                for i in range(num_parts):
                    out[i][name] = None
                continue

            if name == "token":
                for i in range(num_parts):
                    out[i][name] = original[i * split_size : (i + 1) * split_size]
                continue
            elif name == "det":
                for i in range(num_parts):
                    out[i][name] = {
                        cam: v[i * split_size : (i + 1) * split_size]
                        for cam, v in original.items()
                    }
                continue

            parts = split(original, split_size)
            for i, p in enumerate(parts):
                out[i][name] = p
        return [Batch(**g) for g in out]

    def world_to_car(self, frame: int) -> torch.Tensor:
        """
        Get the world space to car transformation matrix.
        [batch_size, 4, 4]
        """
        return self.cam_T[:, frame]

    def car_to_world(self, frame: int) -> torch.Tensor:
        """
        Get the car to world space transformation matrix.
        [batch_size, 4, 4]
        """
        return self.world_to_car(frame).inverse()

    def world_to_cam(self, cam: str, frame: int) -> torch.Tensor:
        """
        Get the world space to camera space transformation matrix.
        [batch_size, 4, 4]
        """
        return torch.linalg.solve(self.T[cam], self.world_to_car(frame))
        # return self.T[cam].pinverse().matmul(self.world_to_car(frame))

    def cam_to_world(self, cam: str, frame: int) -> torch.Tensor:
        """
        Get the camera space to world space transformation matrix.
        [batch_size, 4, 4]
        """
        return self.world_to_cam(cam, frame).inverse()

    def lidar_to_world(self) -> torch.Tensor:
        """
        Get the lidar to world space transformation matrix.
        [batch_size, 4, 4]
        """
        return self.car_to_world(0).matmul(self.lidar_T)

    def save(self, path: str, compress_level: int = 3, threads: int = 1) -> None:
        """
        Saves the batch to the specified path.
        """
        data = asdict(self)

        buffer = io.BytesIO()
        torch.save(data, buffer)
        buffer.seek(0)
        buf = buffer.read()
        if path.endswith(".zst") or path.endswith(".zstd"):
            buf = zstd.compress(buf, compress_level, threads)
        with open(path, "wb") as f:
            f.write(buf)

    @classmethod
    def load(cls, path: str) -> None:
        with open(path, "rb") as f:
            buf = f.read()

        if path.endswith(".zst") or path.endswith(".zstd"):
            buf = zstd.uncompress(buf)
        buffer = io.BytesIO(buf)
        data = torch.load(buffer, weights_only=True)

        return cls(**data)

    def positions(self) -> torch.Tensor:
        """
        Returns the XY positions of the batch.

        You likely want to normalize the batch first.

        Returns:
            tensor [bs, long_cam_T, 2]
        """
        device = self.device()

        world_to_car, mask, lengths = self.long_cam_T
        car_to_world = torch.zeros_like(world_to_car)
        car_to_world[mask] = world_to_car[mask].inverse()

        assert mask.int().sum() == lengths.sum(), (mask, lengths)

        zero_coord = torch.zeros(1, 4, device=device, dtype=torch.float)
        zero_coord[:, -1] = 1

        positions = torch.matmul(car_to_world, zero_coord.T).squeeze(-1)
        positions /= positions[..., -1:] + 1e-8  # perspective warp

        return positions[..., :3]


def _rand_det_target() -> torch.Tensor:
    t = torch.rand(2, 5)
    t[:, :2] *= -1
    return t


def dummy_item() -> Batch:
    N = 3
    color = {}
    cams = ["left", "right"]
    for cam in cams:
        color[cam] = torch.rand(N, 3, 48, 64)

    long_cam_T = torch.rand(random.randint(3, 9) * 3, 4, 4)
    return Batch(
        token=[[f"dummy{i}" for i in range(N)]],
        weight=torch.rand(1)[0],
        distances=torch.rand(N),
        cam_T=long_cam_T[:N],
        long_cam_T=long_cam_T,
        frame_T=torch.rand(N, 4, 4),
        frame_time=torch.arange(N, dtype=torch.float),
        K={cam: torch.rand(4, 4) for cam in cams},
        T={cam: torch.rand(4, 4) for cam in cams},
        color=color,
        mask={cam: torch.rand(1, 48, 64) for cam in cams},
        lidar_T=torch.rand(4, 4),
        lidar=torch.rand(4, random.randint(6, 10)),
        sem_seg={cam: torch.rand(N, 19, 24, 32) for cam in cams},
        det={cam: [[[_rand_det_target()]] * 10 for i in range(N)] for cam in cams},
    )


def dummy_batch() -> Batch:
    BS = 2
    out = collate([dummy_item() for i in range(BS)])
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


def _collate_lidar(
    tensors: List[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    if len(tensors) == 0 or tensors[0] is None:
        return None
    # pyre-fixme[16]: `Optional` has no attribute `size`.
    min_dim = min(x.size(1) for x in tensors)
    assert min_dim > 5, f"min dimension must not be empty {tensors[0].shape}"
    # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
    return torch.stack([x[:, :min_dim] for x in tensors])


def _collate_token(
    tokens: List[List[List[object]]],
) -> Optional[torch.Tensor]:
    out = []
    for token in tokens:
        out += token
    # pyre-fixme[7]: Expected `Optional[Tensor]` but got `List[typing.Any]`.
    return out


def _collate_det(
    dets: List[Dict[str, List[object]]],
) -> Optional[torch.Tensor]:
    if dets[0] is None:
        return None
    out = {cam: [] for cam in dets[0].keys()}
    for det in dets:
        for cam, frames in det.items():
            out[cam] += frames
    # pyre-fixme[7]: Expected `Optional[Tensor]` but got `Dict[str, List[typing.Any]]`.
    return out


def _collate_optional(items: List[Optional[object]]) -> Optional[List[object]]:
    if len(items) == 0 or items[0] is None:
        return None
    return default_collate(items)


_COLLATE_FIELDS: Mapping[str, Callable[[object], object]] = {
    "long_cam_T": _collate_long_cam_T,
    "weight": _collate_weight,
    "global_batch_size": sum,
    "lidar": _collate_lidar,
    "token": _collate_token,
    "sem_seg": _collate_optional,
    "lidar_T": _collate_optional,
    "det": _collate_det,
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

    kwargs = {}
    for field in fields(Batch):
        try:
            kwargs[field.name] = _COLLATE_FIELDS.get(field.name, default_collate)(
                [getattr(b, field.name) for b in batch]
            )
        except Exception as e:
            print(f"failed to collate {field.name}: {e}")
            raise

    return Batch(**kwargs)


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
        self.futures: List[Future[Optional[Batch]]] = []
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
    def _stream_sync(self) -> Generator[None, None, None]:
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

    def _get_batch(self) -> Optional[Batch]:
        it = self.iter
        assert it, "must have iterator"

        with self._stream_sync():
            frames = []
            while len(frames) < self.batch_size:
                frame = next(it)
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
