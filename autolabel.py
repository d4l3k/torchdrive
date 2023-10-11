import os
from tqdm import tqdm
import argparse
import importlib

# set device before loading CUDA/PyTorch
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(LOCAL_RANK))

import torch
from torchdrive.train_config import create_parser
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchdrive.checkpoint import remap_state_dict
from torchdrive.data import Batch, transfer, TransferCollator
from torchdrive.datasets.dataset import Dataset
from torchdrive.models.semantic import BDD100KSemSeg

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
    shuffle=False,
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

model = BDD100KSemSeg(
    device=device,
    compile_fn=torch.compile if args.compile else lambda x: x,
)

for batch in tqdm(collator):
    for cam, frames in batch.color.items():
        squashed = frames.squeeze(1)
        out = model(squashed)
        print(squashed.shape, out.shape)

    break
