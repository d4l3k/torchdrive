import argparse
import importlib
import os
from multiprocessing.pool import ThreadPool

from tqdm import tqdm

# set device before loading CUDA/PyTorch
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(LOCAL_RANK))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchdrive.data import Batch, TransferCollator
from torchdrive.datasets.autolabeler import LabelType, save_tensors
from torchdrive.datasets.dataset import Dataset
from torchdrive.models.semantic import BDD100KSemSeg
from torchdrive.train_config import create_parser

parser = create_parser()
parser.add_argument("--num_workers", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--smoke", action="store_true")
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

# model_config = "upernet_convnext-t_fp16_512x1024_80k_sem_seg_bdd100k.py" # 1.02s/it bs12
# model_config = "upernet_convnext-s_fp16_512x1024_80k_sem_seg_bdd100k.py" # 1.20s/it bs12
model_config = (
    "upernet_convnext-b_fp16_512x1024_80k_sem_seg_bdd100k.py"  # 1.39s/it bs12
)
model = BDD100KSemSeg(
    device=device,
    compile_fn=torch.compile if args.compile else lambda x: x,
    mmlab=True,
    half=True,
    config=model_config,
)

"""
print("Quantizing...")

model_fp32 = model.orig_model
print(model_fp32)
model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
#model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'relu']])
model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)

print("Feeding data...")

for batch in tqdm(collator):
    cam_data = {}
    for cam, frames in batch.color.items():
        frames = frames.squeeze(1)
        frames = model.normalize(frames)
        frames = model.transform(frames)
        model_fp32_prepared(frames)

    break

print("Converting...")
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
"""

assert os.path.exists(args.output), "output dir must exist"
sem_seg_path = os.path.join(args.output, config.dataset, LabelType.SEM_SEG)
print(f"writing to {sem_seg_path}")
os.makedirs(sem_seg_path, exist_ok=True)

pool = ThreadPool(args.batch_size)


handles = []

for batch in tqdm(collator):
    cam_data = {}
    for cam, frames in batch.color.items():
        squashed = frames.squeeze(1)
        pred = model(squashed)
        pred = F.interpolate(pred, scale_factor=1 / 2, mode="bilinear")
        # pred = pred.argmax(dim=1).byte()
        pred = (pred.sigmoid() * 255).byte()
        cam_data[cam] = pred

    for i in range(args.batch_size):
        frame_data = {}
        for cam, pred in cam_data.items():
            frame_data[cam] = pred[i]

        token = batch.token[i][0]
        path = os.path.join(sem_seg_path, f"{token}.safetensors.zstd")
        handles.append(
            pool.apply_async(
                save_tensors,
                (
                    path,
                    frame_data,
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
