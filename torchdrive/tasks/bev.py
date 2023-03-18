import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, cast, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from torchdrive.amp import autocast
from torchdrive.autograd import autograd_context, autograd_resume
from torchdrive.data import Batch
from torchdrive.losses import losses_backward
from torchdrive.models.bev import BEVMerger, BEVUpsampler, CamBEVEncoder


def _cpu_float(
    v: Union[torch.Tensor, int, float, bool]
) -> Union[torch.Tensor, int, float, bool]:
    if isinstance(v, torch.Tensor):
        return v.detach().float().cpu()
    return v


def _get_orig_mod(m: nn.Module) -> nn.Module:
    if hasattr(m, "_orig_mod"):
        # pyre-fixme[7]: Union[Module, Tensor]
        return m._orig_mod
    return m


@dataclass
class Context:
    log_img: bool
    log_text: bool
    global_step: int
    scaler: Optional[amp.GradScaler]
    writer: Optional[SummaryWriter]
    start_frame: int
    output: str
    weights: torch.Tensor
    cam_feats: Mapping[str, torch.Tensor] = field(default_factory=dict)

    name: str = "<unknown>"

    def backward(self, losses: Dict[str, torch.Tensor]) -> None:
        losses_backward(losses, scaler=self.scaler, weights=self.weights)

    def add_scalars(self, name: str, scalars: Dict[str, torch.Tensor]) -> None:
        if self.writer:
            assert self.log_text
            self.writer.add_scalars(
                f"{self.name}-{name}",
                {k: _cpu_float(v) for k, v in scalars.items()},
                global_step=self.global_step,
            )

    def add_scalar(
        self, name: str, scalar: Union[int, float, bool, torch.Tensor]
    ) -> None:
        if self.writer:
            assert self.log_text
            self.writer.add_scalar(
                f"{self.name}-{name}", _cpu_float(scalar), global_step=self.global_step
            )

    def add_image(self, name: str, img: torch.Tensor) -> None:
        if self.writer:
            assert self.log_img
            self.writer.add_image(
                f"{self.name}-{name}", img, global_step=self.global_step
            )

    def add_figure(self, name: str, figure: object) -> None:
        if self.writer:
            assert self.log_img
            self.writer.add_figure(
                f"{self.name}-{name}", figure, global_step=self.global_step
            )


class BEVTask(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, ctx: Context, batch: Batch, bev: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("must implement Task forward")


class BEVTaskVan(torch.nn.Module):
    def __init__(
        self,
        tasks: Dict[str, BEVTask],
        hr_tasks: Dict[str, BEVTask],
        cam_shape: Tuple[int, int],
        bev_shape: Tuple[int, int],
        cameras: List[str],
        dim: int,
        hr_dim: int,
        writer: Optional[SummaryWriter] = None,
        output: str = "out",
        num_encode_frames: int = 3,
        num_upsamples: int = 4,
        num_backprop_frames: int = 2,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        """
        Args:
            tasks: tasks that consume the coarse BEV grid
            hr_tasks: tasks that consume the fine BEV grid
            num_encode_frames: number of frames to merge for the encoders
            num_backprop_frames: number of frames to run backprop for
        """

        super().__init__()

        self.writer = writer
        self.output = output

        self.cameras = cameras
        self.num_encode_frames = num_encode_frames
        self.num_backprop_frames = num_backprop_frames
        self.frame_encoder: nn.Module = compile_fn(
            CamBEVEncoder(
                cameras,
                cam_shape=cam_shape,
                bev_shape=bev_shape,
                dim=dim,
            )
        )
        self.frame_merger: nn.Module = compile_fn(
            BEVMerger(num_frames=self.num_encode_frames, bev_shape=bev_shape, dim=dim)
        )

        self.cam_shape = cam_shape
        self.tasks = nn.ModuleDict(tasks)

        assert (len(tasks) + len(hr_tasks)) > 0, "no tasks specified"
        self.hr_tasks = nn.ModuleDict(hr_tasks)
        if len(hr_tasks) > 0:
            assert hr_dim is not None, "must specify hr_dim for hr_tasks"
            self.upsample: nn.Module = compile_fn(
                BEVUpsampler(
                    num_upsamples=num_upsamples,
                    bev_shape=bev_shape,
                    dim=dim,
                    output_dim=hr_dim,
                )
            )

    def should_log(self, global_step: int, BS: int) -> Tuple[bool, bool]:
        if self.writer is None:
            return False, False

        log_interval = 1000 // BS
        should_log = (global_step % log_interval) == 0
        log_text_interval = log_interval // 10
        log_text = (global_step % log_text_interval) == 0

        return should_log, log_text

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        frame_encoder = cast(CamBEVEncoder, _get_orig_mod(self.frame_encoder))
        per_cam_params = frame_encoder.per_cam_parameters()
        per_cam_set = set(per_cam_params)
        params = [p for p in self.parameters() if p not in per_cam_set]
        per_cam_lr = lr
        # per_cam_lr /= len (self.cameras)
        return [
            {"name": "default", "params": params, "lr": lr},
            {"name": "per_cam", "params": per_cam_params, "lr": per_cam_lr},
        ]

    def forward(
        self, batch: Batch, global_step: int, scaler: Optional[amp.GradScaler] = None
    ) -> Dict[str, torch.Tensor]:
        BS = len(batch.distances)
        log_text: bool
        log_img, log_text = self.should_log(global_step, BS)

        last_cam_feats = {}

        with autocast():
            bev_frames = []
            first_backprop_frame = max(
                self.num_encode_frames - self.num_backprop_frames, 0
            )
            with torch.no_grad():
                for frame in range(0, first_backprop_frame):
                    cams = {cam: batch.color[cam, frame] for cam in self.cameras}
                    _, bev_frame = self.frame_encoder(cams)
                    bev_frames.append(bev_frame)
            for frame in range(first_backprop_frame, self.num_encode_frames):
                cams = {cam: batch.color[cam, frame] for cam in self.cameras}
                last_cam_feats, bev_frame = self.frame_encoder(
                    cams,
                    # pause the last cam encoder backprop for tasks with image
                    # space losses
                    pause=frame == (self.num_encode_frames - 1),
                )
                bev_frames.append(bev_frame)
            bev = self.frame_merger(bev_frames)

        with autograd_context(bev) as bev:
            losses: Dict[str, torch.Tensor] = {}
            task_times: Dict[str, float] = {}

            ctx: Context = Context(
                log_img=log_img,
                log_text=log_text,
                global_step=global_step,
                scaler=scaler,
                writer=self.writer,
                output=self.output,
                start_frame=self.num_encode_frames - 1,
                weights=batch.weight,
                cam_feats=last_cam_feats,
            )

            def _run_tasks(
                task_type: str, tasks: nn.ModuleDict, task_bev: torch.Tensor
            ) -> None:
                for name, task in tasks.items():
                    with torch.autograd.profiler.record_function(name):
                        ctx.name = name

                        task_start = time.time()
                        with autograd_context(task_bev) as per_task_bev:
                            task_losses = task(ctx, batch, per_task_bev)
                            ctx.backward(task_losses)
                            if log_text and (writer := self.writer) is not None:
                                writer.add_scalars(
                                    f"grad/norm/{task_type}",
                                    {
                                        name: torch.linalg.vector_norm(
                                            per_task_bev.grad
                                        ).float()
                                    },
                                    global_step=global_step,
                                )

                        for k, v in task_losses.items():
                            losses[name + "-" + k] = v

                        task_times[name] = time.time() - task_start

            _run_tasks("bev", self.tasks, bev)

            if len(self.hr_tasks) > 0:
                with autograd_context(bev) as bev:
                    with autocast():
                        hr_bev = self.upsample(bev)
                    with autograd_context(hr_bev) as hr_bev:
                        _run_tasks("hr_bev", self.hr_tasks, hr_bev)

                    if log_text and (writer := self.writer) is not None:
                        writer.add_scalars(
                            "grad/norm/bev",
                            {"hr_bev": torch.linalg.vector_norm(bev.grad).float()},
                            global_step=global_step,
                        )

            if log_text and (writer := self.writer) is not None:
                writer.add_scalars(
                    "task_times",
                    task_times,
                    global_step=global_step,
                )

        # resume autograd on cameras
        autograd_resume(*last_cam_feats.values())

        return losses
