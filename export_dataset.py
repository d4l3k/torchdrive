import argparse
import dataclasses
import os
from multiprocessing.pool import ThreadPool
from typing import Dict

import zstd

from tqdm import tqdm

# set device before loading CUDA/PyTorch
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(LOCAL_RANK))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchdrive.data import Batch, TransferCollator
from torchdrive.datasets.autolabeler import AutoLabeler, LabelType, save_tensors
from torchdrive.datasets.dataset import Dataset
from torchdrive.train_config import create_parser, TrainConfig
from torchvision.transforms import v2
from torchvision.utils import save_image
from torchworld.transforms.img import normalize_img

# pyre-fixme[5]: Global expression must be annotated.
parser = create_parser()
parser.add_argument("--num_workers", type=int, required=True)
args: argparse.Namespace = parser.parse_args()


config: TrainConfig = args.config

# overrides
config.num_frames = 1

if "RANK" in os.environ:
    WORLD_SIZE: int = int(os.environ["WORLD_SIZE"])
    RANK: int = int(os.environ["RANK"])
else:
    WORLD_SIZE = 1
    RANK = 0

if torch.cuda.is_available():
    assert torch.cuda.device_count() <= 1
    device_id = 0
    device = torch.device(device_id)
else:
    device = torch.device("cpu")

torch.set_float32_matmul_precision("high")

dataset, _ = config.create_dataset(smoke=args.smoke)


def transform_img(t: torch.Tensor):
    t = normalize_img(t)
    t.clamp_(min=0.0, max=1.0)

    return [v2.functional.to_pil_image(frame) for frame in t]


dataset.transform = transform_img

if isinstance(dataset, AutoLabeler):
    dataset = dataset.dataset

sampler: DistributedSampler[Dataset] = DistributedSampler(
    dataset,
    num_replicas=WORLD_SIZE,
    rank=RANK,
    shuffle=False,
    drop_last=False,
    # seed=1,
)
dataloader = DataLoader[Batch](
    dataset,
    batch_size=None,
    num_workers=args.num_workers,
    pin_memory=False,
    sampler=sampler,
)

assert os.path.exists(args.output), "output dir must exist"

pool = ThreadPool(args.num_workers or 4)

# pyre-fixme[5]: Global expression must be annotated.
handles = []


def run(f, *args):
    handles.append(pool.apply_async(f, args))


output_path = os.path.join(args.output, config.dataset)
index_path = os.path.join(output_path, "index.txt")

os.makedirs(output_path, exist_ok=True)

# with open(index_path, "wta") as index_file:
for batch in tqdm(dataloader, "export"):
    if batch is None:
        continue

    token = batch.token[0][0]
    assert len(token) > 5
    token_path = os.path.join(output_path, f"{token}.pt")

    # index_file.write(token+"\n")
    # index_file.flush()
    if os.path.exists(token_path):
        continue

    for cam, frames in batch.color.items():
        for i, frame in enumerate(frames):
            frame_token = batch.token[0][i]
            frame_path = os.path.join(output_path, f"{frame_token}_{cam}.jpg")
            if not os.path.exists(frame_path):
                run(lambda path, frame: frame.save(path), frame_path, frame)

    # clear color data
    batch = dataclasses.replace(batch, color=None)
    run(
        lambda path, batch: torch.save(dataclasses.asdict(batch), path),
        token_path,
        batch,
    )

    while len(handles) > args.num_workers * 2:
        handles.pop(0).get()

for handle in handles:
    handle.get()
pool.terminate()
pool.join()
# print(i, len(buf), type(buf), len(compressed), pred.dtype)
# break
