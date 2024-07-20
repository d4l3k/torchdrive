import os
from dataclasses import replace
from typing import List

import torch

from PIL import Image
from torchdrive.data import Batch

from torchdrive.datasets.dataset import Dataset, Datasets
from torchdrive.datasets.rice import MultiCamDataset
from torchvision.transforms import v2
from tqdm import tqdm


class ExportedDataset(Dataset):
    NAME: str = Datasets.EXPORTED

    cameras: List[str] = []
    CAMERA_OVERLAP = {}

    def __init__(self, path: str, cameras: List[str]) -> None:
        path = os.path.expanduser(path)

        self.cameras = cameras
        self.path = path
        self.files = []

        raw_files = os.scandir(path)
        for file in tqdm(raw_files, "scanning files"):
            name = file.name
            path = file.path
            if name.endswith(".pt") or name.endswith(".pt.zstd"):
                self.files.append(path)

        print(f"found {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Batch:
        file_path = self.files[idx]
        batch = Batch.load(file_path)
        if batch.color is None:
            batch = replace(batch, color={})

            for camera in batch.mask:
                frames = []
                for token in batch.token[0]:
                    frame_path = os.path.join(
                        self.path,
                        f"{token}_{camera}.jpg",
                    )
                    img = Image.open(frame_path)
                    img = v2.functional.to_image(img)
                    img = img.bfloat16() / 255.0
                    frames.append(img)

                batch.color[camera] = torch.stack(frames, dim=0)
        return batch


if __name__ == "__main__":
    dataset = ExportedDataset(
        path="~/rice_export/rice/",
        cameras=["main"],
    )

    for batch in dataset:
        print(batch)
        break
