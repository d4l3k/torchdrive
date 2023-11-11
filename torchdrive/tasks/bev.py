import itertools
import random
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torchworld.transforms.img import render_color

from torchdrive.amp import autocast
from torchdrive.autograd import (
    autograd_pause,
    autograd_resume,
    log_grad_norm,
    register_log_grad_norm,
)
from torchdrive.data import Batch
from torchdrive.models.bev_backbone import BEVBackbone
from torchdrive.tasks.context import Context
from torchdrive.transforms.batch import BatchTransform, Identity


def _get_orig_mod(m: nn.Module) -> nn.Module:
    if hasattr(m, "_orig_mod"):
        return m._orig_mod
    return m


class BEVTask(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, ctx: Context, batch: Batch, grids: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("must implement Task forward")

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        return [
            {"name": "default", "params": self.parameters(), "lr": lr},
        ]


class BEVTaskVan(torch.nn.Module):
    def __init__(
        self,
        backbone: BEVBackbone,
        cam_encoder: Callable[[], nn.Module],
        tasks: Dict[str, BEVTask],
        hr_tasks: Dict[str, BEVTask],
        cameras: List[str],
        dim: int,
        hr_dim: int,
        num_encode_frames: int = 3,
        num_backprop_frames: int = 2,
        num_drop_encode_cameras: int = 0,
        transform: BatchTransform = Identity(),
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        """
        Args:
            tasks: tasks that consume the coarse BEV grid
            hr_tasks: tasks that consume the fine BEV grid
            num_encode_frames: number of frames to merge for the encoders
            num_backprop_frames: number of frames to run backprop for
            num_drop_encode_cameras: drop input camera features
        """

        super().__init__()

        self.cameras = cameras
        self.num_encode_frames = num_encode_frames
        self.num_backprop_frames = num_backprop_frames
        self.transform = transform
        self.num_drop_encode_cameras = num_drop_encode_cameras

        self.backbone: nn.Module = backbone
        self.camera_encoders = nn.ModuleDict(
            {cam: compile_fn(cam_encoder()) for cam in cameras}
        )

        self.tasks = nn.ModuleDict(tasks)

        assert (len(tasks) + len(hr_tasks)) > 0, "no tasks specified"
        self.hr_tasks = nn.ModuleDict(hr_tasks)
        if len(hr_tasks) > 0:
            assert hr_dim is not None, "must specify hr_dim for hr_tasks"

    def should_log(self, global_step: int, BS: int) -> Tuple[bool, bool]:
        log_text_interval = 1000 // (BS * 20)
        # log_text_interval = 1
        # It's important to scale the less frequent interval off the more
        # frequent one to avoid divisor issues.
        log_img_interval = log_text_interval * 10
        log_img = (global_step % log_img_interval) == 0
        log_text = (global_step % log_text_interval) == 0

        return log_img, log_text

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        per_cam_params = list(self.camera_encoders.parameters())
        per_cam_lr = lr
        per_cam_lr /= len(self.cameras)
        groups: List[Dict[str, object]] = [
            {"name": "backbone", "params": list(self.backbone.parameters()), "lr": lr},
            {"name": "per_cam", "params": per_cam_params, "lr": per_cam_lr},
        ]
        for name, task in itertools.chain(self.tasks.items(), self.hr_tasks.items()):
            task_groups = task.param_opts(lr)
            for group in task_groups:
                group["name"] = f"{name}/{group['name']}"
                groups.append(group)
        return groups

    def forward(
        self,
        batch: Batch,
        global_step: int,
        scaler: Optional[amp.GradScaler] = None,
        writer: Optional[SummaryWriter] = None,
        output: str = "out",
    ) -> Dict[str, torch.Tensor]:
        BS = len(batch.distances)
        log_text: bool
        log_img, log_text = self.should_log(global_step, BS)

        start_frame = self.num_encode_frames - 1

        batch = self.transform(batch)

        last_cam_feats = {}

        # optionally dropout a camera to prevent overfitting
        drop_cameras = random.choices(self.cameras, k=self.num_drop_encode_cameras)
        dropout_cameras = set(self.cameras) - set(drop_cameras)

        with autocast(), torch.autograd.profiler.record_function("camera_feats"):
            first_backprop_frame = max(
                self.num_encode_frames - self.num_backprop_frames, 0
            )

            # individual frames
            camera_feats = {cam: [] for cam in dropout_cameras}
            last_cam_feats_resume = []

            # these first set of frames don't use backprop so set them to eval
            # mode and don't collect gradient
            with torch.no_grad():
                for cam in dropout_cameras:
                    # run frames in parallel
                    encoder = self.camera_encoders[cam]
                    encoder.eval()
                    inp = batch.color[cam][:, 0:first_backprop_frame]
                    feats = encoder(inp.flatten(0, 1)).unflatten(0, inp.shape[0:2])
                    assert not feats.requires_grad
                    for i in range(first_backprop_frame):
                        camera_feats[cam].append(feats[:, i])

            # frames we want backprop for
            for cam in dropout_cameras:
                # run frames in parallel
                encoder = self.camera_encoders[cam]
                encoder.train()
                inp = batch.color[cam][:, first_backprop_frame : self.num_encode_frames]
                feats = encoder(inp.flatten(0, 1)).unflatten(0, inp.shape[0:2])
                num_frames = feats.size(1)

                # pause the last cam encoder backprop for tasks with image
                # space losses
                feats = autograd_pause(feats)
                last_cam_feats_resume.append(feats)

                for i in range(num_frames):
                    feat = feats[:, i]
                    if i == (num_frames - 1):
                        cam_feat = feat
                        if log_text:
                            cam_feat = log_grad_norm(
                                cam_feat,
                                writer,
                                f"grad/norm/encoder/{cam}",
                                "cam_feats",
                                global_step,
                            )
                        last_cam_feats[cam] = cam_feat

                    if log_text:
                        feat = log_grad_norm(
                            feat,
                            writer,
                            f"grad/norm/encoder/{cam}",
                            "bev",
                            global_step,
                        )
                    camera_feats[cam].append(feat)

        for cam_feats in camera_feats.values():
            assert (
                len(cam_feats) == self.num_encode_frames
            ), f"{len(cam_feats)} {self.num_encode_frames}"

        with torch.autograd.profiler.record_function("backbone"):
            hr_bev, bev_feats, bev_intermediates = self.backbone(camera_feats, batch)

        if log_text and writer:
            for tag, x in bev_intermediates.items():
                register_log_grad_norm(
                    t=x,
                    writer=writer,
                    key="grad/norm/backbone-x4",
                    tag=tag,
                    global_step=global_step,
                )

        if log_img and writer:
            for i, bev in enumerate(bev_feats):
                writer.add_image(
                    f"bev/bev{i}",
                    render_color(bev[0].sum(dim=0)),
                    global_step=global_step,
                )
            writer.add_image(
                "bev/hr_bev",
                render_color(hr_bev[0].sum(dim=(0, 1))),
                global_step=global_step,
            )

        hr_bev = autograd_pause(hr_bev)
        bev_feats = [autograd_pause(feat) for feat in bev_feats]
        if isinstance(bev_feats, torch.Tensor):
            bev_feats = [bev_feats]

        losses: Dict[str, torch.Tensor] = {}
        task_times: Dict[str, float] = {}

        ctx: Context = Context(
            log_img=log_img,
            log_text=log_text,
            global_step=global_step,
            scaler=scaler,
            writer=writer,
            output=output,
            start_frame=start_frame,
            weights=batch.weight,
            cam_feats=last_cam_feats,
        )

        def _run_tasks(
            task_type: str, tasks: nn.ModuleDict, task_bev: List[torch.Tensor]
        ) -> None:
            for name, task in tasks.items():
                with torch.autograd.profiler.record_function(name):
                    ctx.name = name

                    task_start = time.time()
                    per_task_bev = task_bev
                    if log_text:
                        per_task_bev = [
                            log_grad_norm(
                                grid,
                                writer,
                                f"grad/norm/{task_type}/{i}",
                                name,
                                global_step,
                            )
                            for i, grid in enumerate(per_task_bev)
                        ]
                    task_losses = task(ctx, batch, per_task_bev)
                    ctx.backward(task_losses)

                    for k, v in task_losses.items():
                        losses[name + "-" + k] = v

                    task_times[name] = time.time() - task_start

        if len(self.tasks) > 0:
            _run_tasks("bev", self.tasks, bev_feats)

        if len(self.hr_tasks) > 0:
            _run_tasks("hr_bev", self.hr_tasks, [hr_bev])

        if log_text and writer is not None:
            writer.add_scalars(
                "task_times",
                task_times,
                global_step=global_step,
            )

        # resume grad
        to_resume = []
        if len(self.tasks) > 0:
            for feat in bev_feats:
                if feat.grad is not None:
                    to_resume.append(feat)
            assert len(to_resume) > 0
        if len(self.hr_tasks) > 0:
            to_resume.append(hr_bev)

        assert len(to_resume) > 0, "no bev grids requiring grad"
        autograd_resume(*to_resume)

        # resume autograd on cameras
        autograd_resume(*last_cam_feats_resume)

        return losses
