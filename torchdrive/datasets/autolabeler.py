import dataclasses
import os.path
from enum import Enum
from typing import Dict, Optional

import safetensors.torch
import torch

import zstd

from torchdrive.data import Batch
from torchdrive.datasets.dataset import Dataset, Datasets

ZSTD_COMPRESS_LEVEL = 3
ZSTD_THREADS = 1


class LabelType(str, Enum):
    SEM_SEG = "sem_seg"


def save_tensors(path: str, data: Dict[str, torch.Tensor]) -> None:
    buf = safetensors.torch.save(data)
    buf = zstd.compress(buf, ZSTD_COMPRESS_LEVEL, ZSTD_THREADS)
    with open(path, "wb") as f:
        f.write(buf)


def load_tensors(path: str) -> object:
    with open(path, "rb") as f:
        buf = f.read()
    buf = zstd.uncompress(buf)
    return safetensors.torch.load(buf)


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

    def __getitem__(self, idx: int) -> Optional[Batch]:
        batch = self.dataset[idx]
        if batch is None:
            return None
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
        return dataclasses.replace(
            batch, sem_seg={cam: torch.stack(frames) for cam, frames in out.items()}
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
