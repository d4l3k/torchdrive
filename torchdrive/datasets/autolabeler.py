import dataclasses
import os.path
import sys
from enum import Enum
from typing import Dict, List, Optional

import safetensors.torch
import torch

import zstd

from torchdrive.data import Batch
from torchdrive.datasets.dataset import Dataset, Datasets

ZSTD_COMPRESS_LEVEL = 3
ZSTD_THREADS = 1


class LabelType(str, Enum):
    SEM_SEG = "sem_seg"
    DET = "det"


def save_tensors(path: str, data: Dict[str, torch.Tensor]) -> None:
    buf = safetensors.torch.save(data)
    buf = zstd.compress(buf, ZSTD_COMPRESS_LEVEL, ZSTD_THREADS)
    with open(path, "wb") as f:
        f.write(buf)


_SIZE = {
    torch.int64: 8,
    torch.float32: 4,
    torch.int32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int16: 2,
    torch.uint8: 1,
    torch.int8: 1,
    torch.bool: 1,
    torch.float64: 8,
}

_TYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    # "U64": torch.uint64,
    "I32": torch.int32,
    # "U32": torch.uint32,
    "I16": torch.int16,
    # "U16": torch.uint16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _getdtype(dtype_str: str) -> torch.dtype:
    return _TYPES[dtype_str]


def _view2torch(safeview) -> Dict[str, torch.Tensor]:
    result = {}
    for k, v in safeview:
        dtype = _getdtype(v["dtype"])
        if len(v["data"]) == 0:
            arr = torch.zeros(*v["shape"], dtype=dtype)
            assert arr.numel() == 0
        else:
            arr = torch.frombuffer(v["data"], dtype=dtype).reshape(v["shape"])
        if sys.byteorder == "big":
            arr = torch.from_numpy(arr.numpy().byteswap(inplace=False))
        result[k] = arr

    return result


def load_tensors(path: str) -> object:
    with open(path, "rb") as f:
        buf = f.read()
    buf = zstd.uncompress(buf)
    raw = safetensors.deserialize(buf)
    return _view2torch(raw)


class AutoLabeler(Dataset):
    """
    Autolabeler takes in a dataset and a cache location and automatically loads
    the autolabeled data based off of the batch tokens.
    """

    def __init__(self, dataset: Dataset, path: str) -> None:
        self.dataset = dataset
        self.path = path

    def __len__(self) -> int:
        return len(self.dataset)

    def _sem_seg(self, batch: Batch) -> Optional[Dict[str, torch.Tensor]]:
        try:
            tokens = batch.token[0]
            out = {cam: [] for cam in self.dataset.cameras}
            for token in tokens:
                path = os.path.join(
                    self.path,
                    self.dataset.NAME,
                    LabelType.SEM_SEG,
                    f"{token}.safetensors.zstd",
                )
                data = load_tensors(path)
                for cam, frame in data.items():
                    out[cam].append(frame.bfloat16() / 255)
            return {cam: torch.stack(frames) for cam, frames in out.items()}
        except FileNotFoundError as e:
            print("autolabeler", e)
            return None

    def _det(self, batch: Batch) -> Optional[Dict[str, List[List[List[torch.Tensor]]]]]:
        try:
            tokens = batch.token[0]
            out = {cam: [[]] for cam in self.dataset.cameras}
            for token in tokens:
                path = os.path.join(
                    self.path,
                    self.dataset.NAME,
                    LabelType.DET,
                    f"{token}.safetensors.zstd",
                )
                data = load_tensors(path)
                for cam in self.dataset.cameras:
                    labels = [data[f"{cam}/{i}"] for i in range(10)]  # 10 labels
                    out[cam][0].append(labels)
            return out
        except FileNotFoundError as e:
            print("autolabeler", e)
            return None

    def __getitem__(self, idx: int) -> Optional[Batch]:
        batch = self.dataset[idx]
        if batch is None:
            return None

        return dataclasses.replace(
            batch,
            sem_seg=self._sem_seg(batch),
            det=self._det(batch),
        )

    @property
    def NAME(self) -> Datasets:
        return self.dataset.NAME

    @property
    def cameras(self) -> Datasets:
        return self.dataset.cameras

    @property
    def CAMERA_OVERLAP(self) -> Datasets:
        return self.dataset.CAMERA_OVERLAP
