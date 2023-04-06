import random
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from torchdrive.amp import autocast
from torchdrive.autograd import autograd_pause, autograd_resume, log_grad_norm
from torchdrive.data import Batch
from torchdrive.models.bev_backbone import BEVBackbone
from torchdrive.tasks.context import Context


def _get_orig_mod(m: nn.Module) -> nn.Module:
    if hasattr(m, "_orig_mod"):
        # pyre-fixme[7]: Union[Module, Tensor]
        return m._orig_mod
    return m


class BEVTask(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, ctx: Context, batch: Batch, bev: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("must implement Task forward")


class BEVTaskVan(torch.nn.Module):
    def __init__(
        self,
        backbone: BEVBackbone,
        cam_encoder: Callable[[], nn.Module],
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
        self.backbone: nn.Module = backbone
        self.camera_encoders = nn.ModuleDict(
            {cam: compile_fn(cam_encoder()) for cam in cameras}
        )

        self.cam_shape = cam_shape
        self.tasks = nn.ModuleDict(tasks)

        assert (len(tasks) + len(hr_tasks)) > 0, "no tasks specified"
        self.hr_tasks = nn.ModuleDict(hr_tasks)
        if len(hr_tasks) > 0:
            assert hr_dim is not None, "must specify hr_dim for hr_tasks"

    def should_log(self, global_step: int, BS: int) -> Tuple[bool, bool]:
        if self.writer is None:
            return False, False

        log_interval = 1000 // BS
        should_log = (global_step % log_interval) == 0
        log_text_interval = log_interval // 10
        log_text = (global_step % log_text_interval) == 0

        return should_log, log_text

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        per_cam_params = list(self.camera_encoders.parameters())
        per_cam_set = set(per_cam_params)
        params = [p for p in self.parameters() if p not in per_cam_set]
        per_cam_lr = lr
        per_cam_lr /= len(self.cameras)
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

        drop_camera = random.choice(self.cameras)
        dropout_cameras = set(self.cameras) - {drop_camera}

        with autocast():
            first_backprop_frame = max(
                self.num_encode_frames - self.num_backprop_frames, 0
            )
            camera_feats = {cam: [] for cam in dropout_cameras}
            for frame in range(0, first_backprop_frame):
                for cam in dropout_cameras:
                    out = self.camera_encoders[cam](batch.color[cam][:, frame]).detach()
                    assert not out.requires_grad
                    camera_feats[cam].append(out)
            for frame in range(first_backprop_frame, self.num_encode_frames):
                pause = frame == (self.num_encode_frames - 1)
                for cam in dropout_cameras:
                    feat = self.camera_encoders[cam](batch.color[cam][:, frame])

                    # pause the last cam encoder backprop for tasks with image
                    # space losses
                    if pause:
                        feat = autograd_pause(feat)
                        last_cam_feats[cam] = feat

                        if log_text:
                            feat = log_grad_norm(
                                feat,
                                self.writer,
                                f"grad/norm/encoder/{cam}",
                                "bev",
                                global_step,
                            )
                    camera_feats[cam].append(feat)

        hr_bev, bev = self.backbone(camera_feats, batch)

        last_cam_feats_resume = last_cam_feats

        if log_text:
            last_cam_feats = {
                cam: log_grad_norm(
                    feat,
                    self.writer,
                    f"grad/norm/encoder/{cam}",
                    "cam_feats",
                    global_step,
                )
                for cam, feat in last_cam_feats.items()
            }

        hr_bev = autograd_pause(hr_bev)
        bev = autograd_pause(bev)

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
                    per_task_bev = task_bev
                    if log_text:
                        per_task_bev = log_grad_norm(
                            per_task_bev,
                            self.writer,
                            f"grad/norm/{task_type}",
                            name,
                            global_step,
                        )
                    task_losses = task(ctx, batch, per_task_bev)
                    ctx.backward(task_losses)

                    for k, v in task_losses.items():
                        losses[name + "-" + k] = v

                    task_times[name] = time.time() - task_start

        if len(self.tasks) > 0:
            _run_tasks("bev", self.tasks, bev)

        if len(self.hr_tasks) > 0:
            _run_tasks("hr_bev", self.hr_tasks, hr_bev)

        if log_text and (writer := self.writer) is not None:
            writer.add_scalars(
                "task_times",
                task_times,
                global_step=global_step,
            )

        # resume grad
        to_resume = []
        if len(self.tasks) > 0:
            to_resume.append(bev)
        if len(self.hr_tasks) > 0:
            to_resume.append(hr_bev)

        assert len(to_resume) > 0, "no bev grids requiring grad"
        autograd_resume(*to_resume)

        # resume autograd on cameras
        autograd_resume(*last_cam_feats_resume.values())

        return losses
