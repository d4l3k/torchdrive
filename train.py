import argparse
import binascii
import json
import os
import os.path
import sys
from collections import defaultdict
from typing import Callable, cast, Dict, List, Optional, Set, Union

# set device before loading CUDA/PyTorch
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(LOCAL_RANK))

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torchinfo
import torchmetrics
from torch import nn, optim
from torch.cuda import amp
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from torchdrive.checkpoint import remap_state_dict
from torchdrive.data import Batch, transfer, TransferCollator
from torchdrive.datasets.dataset import Dataset
from torchdrive.debug import assert_not_nan_dict
from torchdrive.dist import run_ddp_concat
from torchdrive.tasks.bev import BEVTaskVan
from torchdrive.train_config import create_parser, TrainConfig
from tqdm import tqdm

plt.rcParams["lines.linewidth"] = 1
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["figure.autolayout"] = True
matplotlib.use("agg")

parser: argparse.ArgumentParser = create_parser()
args: argparse.Namespace = parser.parse_args()

config: TrainConfig = args.config

os.makedirs(args.output, exist_ok=True)

if "RANK" in os.environ:
    dist.init_process_group("nccl")
    WORLD_SIZE: int = dist.get_world_size()
    RANK: int = dist.get_rank()
else:
    WORLD_SIZE = 1
    RANK = 0

# since we set CUDA_VISIBLE_DEVICES there should only be max 1 device
assert torch.cuda.device_count() <= 1
if torch.cuda.is_available():
    device_id = 0
    device = torch.device(device_id)
else:
    device = torch.device("cpu")

torch.set_float32_matmul_precision("high")
# sdpa_kernel(SDPBackend.FLASH_ATTENTION).__enter__()  # force flash attention

BS: int = config.batch_size
NUM_EPOCHS: int = config.epochs

if RANK == 0:
    writer: Optional[SummaryWriter] = SummaryWriter(
        log_dir=os.path.join(args.output, "tb"),
        max_queue=500,
        flush_secs=60,
    )
    writer.add_text("argv", json.dumps(sys.argv, indent=4))
    # pyre-fixme[16]: to_json
    writer.add_text("train_config", config.to_json(indent=4))

    import git

    repo = git.Repo(search_parent_directories=True)
    writer.add_text("git/sha", repo.head.object.hexsha)
    t: str = repo.head.commit.tree
    writer.add_text("git/diff", repo.git.diff(t))
else:
    writer = None


dataset: Dataset = config.create_dataset(smoke=args.smoke)

if RANK == 0:
    # pyre-fixme[6]: len
    print(f"trainset size {len(dataset)}")

if args.anomaly_detection:
    torch.set_anomaly_enabled(True)

if WORLD_SIZE > 1:

    def convert_sync_bn(m: nn.Module) -> nn.Module:
        if isinstance(m, nn.Module) and m.training:
            print(f"converting syncbn: {m.__class__}")
            return nn.SyncBatchNorm.convert_sync_batchnorm(m)
        return m

    compile_fn = convert_sync_bn
else:
    compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m
if args.compile:
    print("using torch.compile")
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = 128
    parent_fn: Callable[[nn.Module], nn.Module] = compile_fn

    def compile_parent(m: nn.Module) -> nn.Module:
        # pyre-ignore[7]: Expected Module
        return torch.compile(parent_fn(m), dynamic=False)

    compile_fn = compile_parent

model: BEVTaskVan = config.create_model(device=device, compile_fn=compile_fn)

if False and WORLD_SIZE > 1:
    ddp_model: torch.nn.Module = DistributedDataParallel(
        model,
        device_ids=[device_id],
        find_unused_parameters=False,
        # static_graph=True,
    )
else:
    ddp_model = model

if RANK == 0:
    print(torchinfo.summary(model))

params: List[Dict[str, Union[object, List[object]]]] = model.param_opts(config.lr)
# pyre-fixme[9]: object
lr_groups: List[float] = [p["lr"] if "lr" in p else config.lr for p in params]
name_groups: List[str] = [cast(str, p["name"]) for p in params]
flat_params: Set[object] = set()
for group in params:
    for p in cast(List[object], group["params"]):
        flat_params.add(p)

for name, p in model.named_parameters():
    assert p in flat_params, f"param_opts doesn't include {name}"

ddp_params: Set[nn.Parameter] = set(ddp_model.parameters())
for name, p in ddp_model.named_parameters():
    assert p in flat_params, name
for p in flat_params:
    assert p in ddp_params

optimizer = optim.AdamW(
    params,
    lr=config.lr,
    weight_decay=1e-2,  # 1e-4
)  # increased to reduce exploding gradients
lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=config.step_size, gamma=0.1
)

global_step: int = 0
CHECKPOINT_PATH: str = os.path.join(args.output, "model.pt")
GLOBAL_STEP_KEY: str = "global_step"
MODEL_KEY: str = "model"
OPTIM_KEY: str = "optim"


def save(epoch: int) -> None:
    if RANK != 0:
        return
    tmp_path = CHECKPOINT_PATH + ".tmp"

    state_dict = model.state_dict()
    # don't save a corrupt checkpoint
    assert_not_nan_dict(state_dict)

    torch.save(
        {
            MODEL_KEY: state_dict,
            OPTIM_KEY: optimizer.state_dict(),
            "epoch": epoch,
            GLOBAL_STEP_KEY: global_step,
        },
        tmp_path,
    )
    # We use a tmp path + rename to ensure we don't end up with a corrupted
    # checkpoint for errors such as running out of disk space.
    os.rename(tmp_path, CHECKPOINT_PATH)
    print(f"saved to {CHECKPOINT_PATH}")


load_path: str = args.load

LOAD_FAULT_TOLERANCE: bool = os.path.exists(CHECKPOINT_PATH)

if LOAD_FAULT_TOLERANCE:
    print(f"loading from fault tolerance checkpoint {CHECKPOINT_PATH}")
    load_path = CHECKPOINT_PATH

if load_path:
    ckpt: Dict[str, torch.Tensor] = torch.load(load_path, weights_only=True)

    if not args.skip_load_optim or LOAD_FAULT_TOLERANCE:
        print("loading optim state_dict")
        optim_dict: Dict[str, object] = ckpt[OPTIM_KEY]  # pyre-fixme
        optim_dict = transfer("optim_dict", optim_dict, device=torch.device("cpu"))
        optimizer.load_state_dict(optim_dict)

        # NOTE: this overrides any LR set by schedulers
        assert len(lr_groups) == len(optimizer.param_groups)
        for lr, og in zip(lr_groups, optimizer.param_groups):
            og["lr"] = lr

        if GLOBAL_STEP_KEY in ckpt:
            # pyre-fixme[9]: int vs Tensor
            global_step = ckpt[GLOBAL_STEP_KEY]

    state_dict: Dict[str, torch.Tensor] = ckpt[MODEL_KEY]  # pyre-fixme

    # remap state_dict
    state_dict = remap_state_dict(state_dict, model)
    # state_dict = {k:v for k,v in state_dict.items() if "path" not in k}

    assert_not_nan_dict(state_dict)

    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"failed to load state_dict, err: {e}")


sampler: DistributedSampler[Dataset] = DistributedSampler(
    dataset,
    num_replicas=WORLD_SIZE,
    rank=RANK,
    shuffle=True,
    drop_last=True,
    seed=binascii.crc32((args.load or args.output).encode("utf-8")) + global_step,
)
dataloader = DataLoader[Batch](
    dataset,
    batch_size=None,
    num_workers=config.num_workers,
    # drop_last=True,
    # collate_fn=nonstrict_collate,
    prefetch_factor=1,
    pin_memory=True,
    sampler=sampler,
)
collator = TransferCollator(dataloader, batch_size=config.batch_size, device=device)


meaned_losses: Dict[str, torchmetrics.aggregation.MeanMetric] = defaultdict(
    lambda: torchmetrics.aggregation.MeanMetric(sync_on_compute=False).to(device)
)


if args.profile:  # and rank == 0:
    prof: Optional[torch.profiler.profile] = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=10, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            os.path.join(args.output, f"profile_step_{global_step}")
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ).__enter__()
else:
    prof = None


@record
def run():
    global global_step

    for epoch in range(NUM_EPOCHS):
        batch_idx = 0

        if writer:
            writer.add_scalars(
                "lr",
                {
                    name: group["lr"]
                    for name, group in zip(name_groups, optimizer.param_groups)
                },
                global_step,
            )

        # only show progress on rank 0
        batch_iter = (
            tqdm(collator, desc=f"epoch {epoch}") if LOCAL_RANK == 0 else collator
        )
        for batch in batch_iter:
            batch = cast(Optional[Batch], batch)
            if batch is None:
                print("empty batch")
                continue

            batch = batch.to(device)

            log_img, log_text = model.should_log(global_step, BS)

            optimizer.zero_grad(set_to_none=True)

            losses = ddp_model(batch, global_step, writer=writer, output=args.output)
            loss: torch.Tensor = cast(torch.Tensor, sum(losses.values()))
            assert not loss.requires_grad

            run_ddp_concat(model.parameters())

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
            if config.grad_clip > 0:
                # clip gradients to avoid loss explosion
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.grad_clip
                )

            optimizer.step()

            with torch.no_grad():
                # pyre-fixme[9]: int
                rollup: Dict[str, torch.Tensor] = defaultdict(lambda: 0)
                for k, v in losses.items():
                    meaned_losses[k].update(v.sum())
                    # compute roll up metrics
                    rollupk = "loss/" + k.partition("/")[0]
                    if rollupk != k:
                        rollup[rollupk] += v.sum()
                for k, v in rollup.items():
                    meaned_losses[k].update(v.sum())
                meaned_losses["loss"].update(loss)

                if log_text and writer is not None:
                    for k, v in meaned_losses.items():
                        writer.add_scalar(k, v.compute(), global_step)
                        v.reset()

                if log_img:
                    for k, v in losses.items():
                        print(f"- {k}: {v.item()}")
                    print(f"= {loss.item()}")

                if (
                    global_step > 0
                    and (global_step % (args.checkpoint_every // BS)) == 0
                ):
                    save(epoch)

                batch_idx += 1
                global_step += 1

            if prof:
                prof.step()

        print(f"epoch {epoch}")

        lr_scheduler.step()
        save(epoch)

    save(epoch + 1)


if __name__ == "__main__":
    run()
