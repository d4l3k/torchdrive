from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from torchdrive.autograd import autograd_context
from torchdrive.data import Batch
from torchdrive.models.bev import BEVMerger, CamBEVEncoder


@dataclass
class Context:
    log_img: bool
    log_text: bool
    global_step: int
    scaler: Optional[amp.GradScaler]
    writer: Optional[SummaryWriter]
    start_frame: int


class BEVTask(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, context: Context, batch: Batch, bev: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("must implement Task forward")


class BEVTaskVan(torch.nn.Module):
    def __init__(
        self,
        tasks: Dict[str, BEVTask],
        cam_shape: Tuple[int, int],
        bev_shape: Tuple[int, int],
        cameras: List[str],
        dim: int,
        writer: Optional[SummaryWriter] = None,
        encode_frames: int = 3,
    ) -> None:
        super().__init__()

        self.writer = writer

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
        bev_frames = []
        for frame in range(self.encode_frames):
            cams = {cam: batch.color[cam, frame] for cam in self.cameras}
            bev_frames.append(self.frame_encoder(cams))
        bev = self.frame_merger(bev_frames)

        with autograd_context(bev) as bev:
            losses = {}

            ctx = Context(
                log_img=log_img,
                log_text=log_text,
                global_step=global_step,
                scaler=scaler,
                writer=self.writer,
                start_frame=self.encode_frames - 1,
            )

            for name, task in self.tasks.items():
                task_losses = task(ctx, batch, bev)
                for k, v in task_losses.items():
                    losses[name + "/" + k] = v

        return losses
