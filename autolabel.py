import argparse
import importlib
import os
from multiprocessing.pool import ThreadPool
from typing import Dict

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
from torchdrive.datasets.autolabeler import LabelType, save_tensors
from torchdrive.datasets.dataset import Dataset
from torchdrive.train_config import create_parser

parser = create_parser()
parser.add_argument("--num_workers", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
args: argparse.Namespace = parser.parse_args()


config_module = importlib.import_module("configs." + args.config)
config = config_module.CONFIG

# overrides
config.num_frames = 1

if "RANK" in os.environ:
    dist.init_process_group("nccl")
    WORLD_SIZE: int = dist.get_world_size()
    RANK: int = dist.get_rank()
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

dataset: Dataset = config.create_dataset(smoke=args.smoke)

sampler: DistributedSampler[Dataset] = DistributedSampler(
    dataset,
    num_replicas=WORLD_SIZE,
    rank=RANK,
    shuffle=True,
    drop_last=False,
    seed=0,
)
dataloader = DataLoader[Batch](
    dataset,
    batch_size=None,
    num_workers=args.num_workers,
    pin_memory=True,
    sampler=sampler,
)
collator = TransferCollator(dataloader, batch_size=args.batch_size, device=device)

compile_fn = torch.compile if args.compile else lambda x: x


class LabelSemSeg(nn.Module):
    TYPE = LabelType.SEM_SEG

    def __init__(self) -> None:
        super().__init__()

        from torchdrive.models.semantic import BDD100KSemSeg

        # model_config = "upernet_convnext-t_fp16_512x1024_80k_sem_seg_bdd100k.py" # 1.02s/it bs12
        # model_config = "upernet_convnext-s_fp16_512x1024_80k_sem_seg_bdd100k.py" # 1.20s/it bs12
        model_config = (
            "upernet_convnext-b_fp16_512x1024_80k_sem_seg_bdd100k.py"  # 1.39s/it bs12
        )
        self.model = BDD100KSemSeg(
            device=device,
            compile_fn=compile_fn,
            mmlab=True,
            half=True,
            config=model_config,
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        pred = self.model(img)
        pred = F.interpolate(pred, scale_factor=1 / 2, mode="bilinear")
        # uses 1 byte to represent probabilities between 0-1 (0:0.0, 255:1.0)
        pred = (pred.sigmoid() * 255).byte()
        return pred


class LabelDet(nn.Module):
    TYPE = LabelType.DET

    def __init__(self) -> None:
        super().__init__()

        from torchdrive.models.det import BDD100KDet

        model_config = "cascade_rcnn_convnext-s_fpn_fp16_3x_det_bdd100k.py"
        self.model = BDD100KDet(
            config=model_config,
            device=device,
            half=True,
            compile_fn=compile_fn,
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        pred = self.model(img)
        return pred


TASKS = [
    LabelDet(),
    # LabelSemSeg(),
]


def flatten(v: object) -> Dict[str, torch.Tensor]:
    out = {}

    def _flatten(prefix: str, v: object):
        if isinstance(v, torch.Tensor):
            out[prefix] = v
        elif isinstance(v, np.ndarray):
            out[prefix] = torch.from_numpy(v)
        elif isinstance(v, dict):
            for k, v2 in v.items():
                _flatten(os.path.join(prefix, k), v2)
        elif isinstance(v, list):
            for i, v2 in enumerate(v):
                _flatten(os.path.join(prefix, str(i)), v2)
        else:
            raise TypeError(f"unknown type of {type(v)} - {v}")

    _flatten("", v)
    return out


def get_task_path(task: nn.Module) -> str:
    return os.path.join(args.output, config.dataset, task.TYPE)


assert os.path.exists(args.output), "output dir must exist"

for task in TASKS:
    task_path = get_task_path(task)
    print(f"{task.TYPE}: writing to {task_path}")
    os.makedirs(task_path, exist_ok=True)

pool = ThreadPool(args.batch_size)

handles = []

for batch in tqdm(collator):
    for task in TASKS:
        task_path = get_task_path(task)
        token_paths = []
        idxs = []
        for i in range(batch.batch_size()):
            token = batch.token[i][0]
            assert len(token) > 5
            token_path = os.path.join(task_path, f"{token}.safetensors.zstd")
            token_paths.append(token_path)
            if not os.path.exists(token_path):
                idxs.append(i)

        if len(idxs) == 0:
            continue

        cam_data = {}
        for cam, frames in batch.color.items():
            squashed = frames[idxs].squeeze(1)
            cam_data[cam] = task(squashed)

        for j, i in enumerate(idxs):
            frame_data = {}
            for cam, pred in cam_data.items():
                frame_data[cam] = pred[j]

            path = token_paths[i]
            handles.append(
                pool.apply_async(
                    save_tensors,
                    (
                        path,
                        flatten(frame_data),
                    ),
                )
            )

    while len(handles) > args.batch_size * 2:
        handles.pop(0).get()

for handle in handles:
    handle.get()
pool.terminate()
pool.join()
# print(i, len(buf), type(buf), len(compressed), pred.dtype)
# break
