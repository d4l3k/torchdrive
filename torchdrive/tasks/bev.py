from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from torchdrive.amp import autocast
from torchdrive.autograd import autograd_context
from torchdrive.data import Batch
from torchdrive.losses import losses_backward
from torchdrive.models.bev import BEVMerger, BEVUpsampler, CamBEVEncoder


def _cpu_float(
    v: Union[torch.Tensor, int, float, bool]
) -> Union[torch.Tensor, int, float, bool]:
    if isinstance(v, torch.Tensor):
        return v.detach().float().cpu()
    return v


@dataclass
class Context:
    log_img: bool
    log_text: bool
    global_step: int
    scaler: Optional[amp.GradScaler]
    writer: Optional[SummaryWriter]
    start_frame: int
    output: str
    name: str = "<unknown>"

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
        encode_frames: int = 3,
        num_upsamples: int = 4,
    ) -> None:
        """
        Args:
            tasks: tasks that consume the coarse BEV grid
            hr_tasks: tasks that consume the fine BEV grid
        """

        super().__init__()

        self.writer = writer
        self.output = output

        self.cameras = cameras
        self.encode_frames = encode_frames
        self.frame_encoder = CamBEVEncoder(
            cameras, cam_shape=cam_shape, bev_shape=bev_shape, dim=dim
        )
        self.frame_merger = BEVMerger(
            num_frames=self.encode_frames, bev_shape=bev_shape, dim=dim
        )

        self.cam_shape = cam_shape
        self.tasks = nn.ModuleDict(tasks)

        assert (len(tasks) + len(hr_tasks)) > 0, "no tasks specified"
        self.hr_tasks = nn.ModuleDict(hr_tasks)
        if len(hr_tasks) > 0:
            assert hr_dim is not None, "must specify hr_dim for hr_tasks"
            self.upsample: BEVUpsampler = BEVUpsampler(
                num_upsamples=num_upsamples,
                bev_shape=bev_shape,
                dim=dim,
                output_dim=hr_dim,
            )

    def should_log(self, global_step: int) -> Tuple[bool, bool]:
        if self.writer is None:
            return False, False

        log_interval = 500
        should_log = (global_step % log_interval) == 0
        log_text_interval = log_interval // 10
        log_text = (global_step % log_text_interval) == 0

        return should_log, log_text

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        params = list(self.parameters())
        return [
            {"name": "default", "params": params, "lr": lr},
        ]

    def forward(
        self, batch: Batch, global_step: int, scaler: Optional[amp.GradScaler] = None
    ) -> Dict[str, torch.Tensor]:
        log_img, log_text = self.should_log(global_step)
        BS = len(batch.distances)

        with autocast():
            bev_frames = []
            for frame in range(self.encode_frames):
                cams = {cam: batch.color[cam, frame] for cam in self.cameras}
                bev_frames.append(self.frame_encoder(cams))
            bev = self.frame_merger(bev_frames)

        with autograd_context(bev) as bev:
            losses: Dict[str, torch.Tensor] = {}

            ctx: Context = Context(
                log_img=log_img,
                log_text=log_text,
                global_step=global_step,
                scaler=scaler,
                writer=self.writer,
                output=self.output,
                start_frame=self.encode_frames - 1,
            )

            def _run_tasks(tasks: nn.ModuleDict, task_bev: torch.Tensor) -> None:
                for name, task in tasks.items():
                    ctx.name = name

                    task_losses = task(ctx, batch, task_bev)
                    losses_backward(task_losses, scaler)
                    for k, v in task_losses.items():
                        losses[name + "-" + k] = v

            _run_tasks(self.tasks, bev)

            if len(self.hr_tasks) > 0:
                with autocast():
                    hr_bev = self.upsample(bev)
                with autograd_context(hr_bev) as hr_bev:
                    _run_tasks(self.hr_tasks, hr_bev)

        return losses
