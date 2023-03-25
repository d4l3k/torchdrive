import argparse
import binascii
import json
import os
import os.path
from collections import defaultdict
from typing import Callable, cast, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from torchdrive.checkpoint import remap_state_dict
from torchdrive.data import Batch, transfer, TransferCollator
from torchdrive.datasets.rice import MultiCamDataset
from torchdrive.dist import run_ddp_concat
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
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--checkpoint_every", type=int, default=2000)
parser.add_argument("--num_encode_frames", type=int, default=3)
parser.add_argument("--profile", default=False, action="store_true")
parser.add_argument(
    "--grad_sizes", default=False, action="store_true", help="log grad sizes"
)
parser.add_argument(
    "--compile", default=False, action="store_true", help="use torch.compile"
)

# tasks
parser.add_argument("--det", default=False, action="store_true")
parser.add_argument("--ae", default=False, action="store_true")
parser.add_argument("--voxel", default=False, action="store_true")
parser.add_argument("--voxelsem", default=None, type=tuple_str)
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
# pyre-fixme[16]: no attribute set_float32_matmul_precision
torch.set_float32_matmul_precision("high")

BS: int = args.batch_size
NUM_EPOCHS: int = args.epochs

dataset = MultiCamDataset(
    index_file=args.dataset,
    mask_dir=args.masks,
    cameras=args.cameras,
    dynamic=True,
    cam_shape=args.cam_shape,
    # 3 encode frames, 3 decode frames, overlap last frame
    nframes_per_point=args.num_encode_frames + 2,
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
    batch_size=None,
    num_workers=args.num_workers,
    # drop_last=True,
    # collate_fn=nonstrict_collate,
    pin_memory=True,
    sampler=sampler,
)
collator = TransferCollator(dataloader, batch_size=args.batch_size, device=device)

if args.anomaly_detection:
    torch.set_anomaly_enabled(True)

compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m
if args.compile:
    print("using torch.compile")
    # pyre-fixme[16]: no attribute compile
    compile_fn = torch.compile

tasks: Dict[str, BEVTask] = {}
hr_tasks: Dict[str, BEVTask] = {}
if args.path:
    tasks["path"] = PathTask(
        bev_shape=args.bev_shape,
        bev_dim=args.dim,
        dim=args.dim,
        # compile_fn=compile_fn,
    )
if args.det:
    tasks["det"] = DetTask(
        cameras=args.cameras,
        cam_shape=args.cam_shape,
        bev_shape=args.bev_shape,
        dim=args.dim,
        device=device,
        compile_fn=compile_fn,
    )
if args.ae:
    tasks["ae"] = AETask(
        cameras=args.cameras,
        cam_shape=args.cam_shape,
        bev_shape=args.bev_shape,
        dim=args.dim,
    )
if args.voxel:
    hr_tasks["voxel"] = VoxelTask(
        cameras=args.cameras,
        cam_shape=args.cam_shape,
        dim=args.dim,
        hr_dim=args.hr_dim,
        height=16,
        z_offset=0.4,
        device=device,
        semantic=args.voxelsem,
        compile_fn=compile_fn,
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
    compile_fn=compile_fn,
    num_encode_frames=args.num_encode_frames,
)

model = model.to(device)
if False and world_size > 1:
    ddp_model: torch.nn.Module = DistributedDataParallel(
        model,
        device_ids=[device_id],
        find_unused_parameters=False,
        # static_graph=True,
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
# scaler is only needed for fp16 not bf16
# scaler: amp.GradScaler = amp.GradScaler()
scaler: Optional[amp.GradScaler] = None

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
    # state_dict = {k:v for k,v in state_dict.items() if "path" not in k}

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
    if rank != 0:
        return
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


if args.profile and rank == 0:
    prof: Optional[torch.profiler.profile] = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=10, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            os.path.join(args.output, "profile")
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ).__enter__()
else:
    prof = None

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

    for batch in tqdm(collator, desc=f"epoch {epoch}"):
        batch = cast(Optional[Batch], batch)
        if batch is None:
            print("empty batch")
            continue

        batch = batch.to(device)

        log_img, log_text = model.should_log(global_step, BS)

        optimizer.zero_grad(set_to_none=True)

        losses = ddp_model(batch, global_step, scaler)
        loss: torch.Tensor = cast(torch.Tensor, sum(losses.values()))
        assert not loss.requires_grad

        run_ddp_concat(model.parameters())

        if scaler:
            scaler.unscale_(optimizer)
        if log_text and writer and args.grad_sizes:
            with torch.no_grad():
                max_grad = 0
                max_weight = 0
                max_param = "n/a"
                for name, p in model.named_parameters():
                    # find unused parameters
                    if p.requires_grad:
                        assert p.grad is not None, f"missing grad on param {name}"

                    if p.grad is None:
                        continue

                    grad_abs = p.grad.abs().amax()
                    if grad_abs > max_grad:
                        max_grad = grad_abs
                        max_weight = p.abs().amax()
                        max_param = name
                writer.add_scalar("grad/max", max_grad, global_step)
                writer.add_scalar("grad/max_weight", max_weight, global_step)
                writer.add_text("grad/max_name", max_param, global_step)
        if args.grad_clip > 0:
            # clip gradients to avoid loss explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

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

            if log_img:
                for k, v in losses.items():
                    print(f"- {k}: {v.item()}")
                print(f"= {loss.item()}")

            if global_step > 0 and (global_step % (args.checkpoint_every // BS)) == 0:
                save(epoch)

            batch_idx += 1
            global_step += 1

        if prof:
            prof.step()

    epoch_loss_mean = epoch_loss / batch_idx

    if writer is not None:
        writer.add_scalar("epoch_loss", epoch_loss, global_step)
        writer.add_scalar("lr", epoch_loss, global_step)

    print(f"epoch {epoch} loss {epoch_loss_mean}")

    lr_scheduler.step()
    save(epoch)
