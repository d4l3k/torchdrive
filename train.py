import argparse
import binascii
import json
import os
import os.path
from collections import defaultdict
from typing import cast, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
from torch import optim
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from torchdrive.checkpoint import remap_state_dict
from torchdrive.data import Batch, nonstrict_collate, transfer
from torchdrive.datasets.rice import MultiCamDataset
from torchdrive.tasks.ae import AETask
from torchdrive.tasks.bev import BEVTask, BEVTaskVan
from torchdrive.tasks.det import DetTask
from torchdrive.tasks.path import PathTask
from torchdrive.tasks.voxel import VoxelTask
from tqdm import tqdm


def tuple_str(s: str) -> Tuple[str, ...]:
    return tuple(s.split(","))


def tuple_int(s: str) -> Tuple[int, ...]:
    return tuple(int(v) for v in s.split(","))


parser = argparse.ArgumentParser(description="train")
parser.add_argument("--output", required=True, type=str, default="out")
parser.add_argument("--load", type=str)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--masks", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--step_size", type=int, default=15)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument(
    "--cameras",
    default="main,narrow,fisheye,leftpillar,rightpillar,leftrepeater,rightrepeater,backup",
    type=tuple_str,
)
parser.add_argument("--dim", type=int, required=True)
parser.add_argument("--hr_dim", type=int)
parser.add_argument("--bev_shape", type=tuple_int, required=True)
parser.add_argument("--cam_shape", type=tuple_int, required=True)
parser.add_argument("--skip_load_optim", default=False, action="store_true")
parser.add_argument("--anomaly-detection", default=False, action="store_true")
parser.add_argument("--limit_size", type=int)
parser.add_argument("--grad_clip", type=float, default=35)

# tasks
parser.add_argument("--det", default=False, action="store_true")
parser.add_argument("--ae", default=False, action="store_true")
parser.add_argument("--voxel", default=False, action="store_true")
parser.add_argument("--path", default=False, action="store_true")

args: argparse.Namespace = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

if "RANK" in os.environ:
    dist.init_process_group("nccl")
    world_size: int = dist.get_world_size()
    rank: int = dist.get_rank()
else:
    world_size = 1
    rank = 0

if rank == 0:
    writer: Optional[SummaryWriter] = SummaryWriter(
        log_dir=os.path.join(args.output, "tb"),
        max_queue=500,
        flush_secs=60,
    )
    writer.add_text("args", json.dumps(vars(args), indent=4))

    import git

    repo = git.Repo(search_parent_directories=True)
    writer.add_text("git/sha", repo.head.object.hexsha)
    t: str = repo.head.commit.tree
    writer.add_text("git/diff", repo.git.diff(t))
else:
    writer = None

device_id: int = rank % torch.cuda.device_count()
device = torch.device(device_id)
torch.cuda.set_device(device)

BS: int = args.batch_size
NUM_EPOCHS: int = args.epochs

dataset = MultiCamDataset(
    index_file=args.dataset,
    mask_dir=args.masks,
    cameras=args.cameras,
    dynamic=True,
    cam_shape=args.cam_shape,
    # 3 encode frames, 3 decode frames, overlap last frame
    nframes_per_point=5,
    limit_size=args.limit_size,
)
if rank == 0:
    print(f"trainset size {len(dataset)}")

sampler: DistributedSampler[MultiCamDataset] = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=True,
    seed=binascii.crc32((args.load or args.output).encode("utf-8")),
)
dataloader = DataLoader[Batch](
    dataset,
    batch_size=BS,
    num_workers=args.num_workers,
    drop_last=True,
    collate_fn=nonstrict_collate,
    pin_memory=False,
    sampler=sampler,
)

if args.anomaly_detection:
    torch.set_anomaly_enabled(True)

tasks: Dict[str, BEVTask] = {}
hr_tasks: Dict[str, BEVTask] = {}
if args.det:
    tasks["det"] = DetTask(
        cameras=args.cameras,
        cam_shape=args.cam_shape,
        bev_shape=args.bev_shape,
        dim=args.dim,
        device=device,
    )
if args.ae:
    tasks["ae"] = AETask(
        cameras=args.cameras,
        cam_shape=args.cam_shape,
        bev_shape=args.bev_shape,
        dim=args.dim,
    )
if args.path:
    tasks["path"] = PathTask(
        bev_shape=args.bev_shape,
        bev_dim=args.dim,
    )
if args.voxel:
    hr_tasks["voxel"] = VoxelTask(
        cameras=args.cameras,
        cam_shape=args.cam_shape,
        dim=args.hr_dim,
        height=12,
    )

model = BEVTaskVan(
    tasks=tasks,
    hr_tasks=hr_tasks,
    bev_shape=args.bev_shape,
    cam_shape=args.cam_shape,
    cameras=args.cameras,
    dim=args.dim,
    hr_dim=args.hr_dim,
    writer=writer,
    output=args.output,
)

model = model.to(device)
if world_size > 1:
    ddp_model: torch.nn.Module = DDP(
        model, device_ids=[device_id], find_unused_parameters=True
    )
else:
    ddp_model = model

if rank == 0:
    print(model)

params: List[Dict[str, Union[object, List[object]]]] = model.param_opts(args.lr)
lr_groups: List[float] = [p["lr"] if "lr" in p else args.lr for p in params]
name_groups: List[str] = [cast(str, p["name"]) for p in params]
flat_params: Set[object] = set()
ddp_params: Iterator[Parameter] = ddp_model.parameters()
for group in params:
    for p in cast(List[object], group["params"]):
        flat_params.add(p)
for p in ddp_params:
    assert p in flat_params
optimizer = optim.AdamW(
    params,
    lr=args.lr,
    weight_decay=1e-2,  # 1e-4
)  # increased to reduce exploding gradients
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
scaler: amp.GradScaler = amp.GradScaler()

if args.load:
    state_dict: Dict[str, torch.Tensor] = torch.load(args.load, map_location=device)

    # new save format
    if "optim" in state_dict:
        if not args.skip_load_optim:
            print("loading optim state_dict")
            optim_dict: Dict[str, object] = state_dict["optim"]  # pyre-fixme
            optim_dict = transfer("optim_dict", optim_dict, device=torch.device("cpu"))
            optimizer.load_state_dict(optim_dict)

            # NOTE: this overrides any LR set by schedulers
            assert len(lr_groups) == len(optimizer.param_groups)
            for lr, og in zip(lr_groups, optimizer.param_groups):
                og["lr"] = lr

        state_dict = state_dict["model"]  # pyre-fixme

    # remap state_dict
    state_dict = remap_state_dict(state_dict, model)

    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"failed to load state_dict, err: {e}")


global_step = 0

meaned_losses: Dict[str, Union[float, torch.Tensor]] = {}


def reset_metrics() -> None:
    global loss_count, meaned_losses
    meaned_losses = defaultdict[str, Union[float, torch.Tensor]](lambda: 0.0)
    loss_count = 0


def save(epoch: int) -> None:
    path = os.path.join(args.output, f"model_{epoch}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
        },
        path,
    )
    l = epoch_loss / batch_idx if batch_idx else 0
    print(f"saved to {path}, loss = {l}")


for epoch in range(NUM_EPOCHS):

    batch_idx = 0
    epoch_loss = 0

    reset_metrics()

    if writer:
        writer.add_scalars(
            "lr",
            {
                name: group["lr"]
                for name, group in zip(name_groups, optimizer.param_groups)
            },
            global_step,
        )

    for batch in tqdm(dataloader, desc=f"epoch {epoch}"):
        batch = cast(Optional[Batch], batch)
        if batch is None:
            print("empty batch")
            continue

        batch = batch.to(device)

        log_img, log_text = model.should_log(global_step)

        optimizer.zero_grad(set_to_none=True)

        # with torch.autograd.detect_anomaly():
        losses = ddp_model(batch, global_step, scaler)
        loss: torch.Tensor = cast(torch.Tensor, sum(losses.values()))
        assert not loss.requires_grad

        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            # clip gradients to avoid loss explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            epoch_loss += loss.detach()

            for k, v in losses.items():
                meaned_losses[k] += v
                # compute roll up metrics
                rollupk = "loss/" + k.partition("/")[0]
                if rollupk != k:
                    meaned_losses[rollupk] += v
            meaned_losses["loss"] += loss
            loss_count += 1

            if log_text and writer is not None:
                for k, v in meaned_losses.items():
                    writer.add_scalar(k, v / loss_count, global_step)
                reset_metrics()

            if global_step % args.log_interval == 0:
                for k, v in losses.items():
                    print(f"- {k}: {v.item()}")
                print(f"= {loss.item()}")

            if global_step > 0 and (global_step % 1000) == 0:
                save(epoch)

            batch_idx += 1
            global_step += 1

    epoch_loss_mean = epoch_loss / batch_idx

    if writer is not None:
        writer.add_scalar("epoch_loss", epoch_loss, global_step)
        writer.add_scalar("lr", epoch_loss, global_step)

    print(f"epoch {epoch} loss {epoch_loss_mean}")

    lr_scheduler.step()
    save(epoch)
