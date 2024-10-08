import itertools
import random
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
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

from torchdrive.tasks.van import Van
from torchdrive.transforms.batch import BatchTransform, Identity

from torchworld.structures.grid import Grid3d
from torchworld.transforms.img import render_color
from torchworld.transforms.mask import random_block_mask


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

    def set_camera_encoders(self, camera_encoders: nn.ModuleDict) -> None:
        pass


class BEVTaskVan(Van, torch.nn.Module):
    def __init__(
        self,
        backbone: BEVBackbone,
        cam_encoder: Callable[[], nn.Module],
        tasks: Dict[str, BEVTask],
        hr_tasks: Dict[str, BEVTask],
        cameras: List[str],
        dim: int,
        cam_dim: int,
        hr_dim: int,
        scale: float,
        grid_shape: Tuple[int, int, int],
        cam_features_mask_ratio: float = 0.0,
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
        self.grid_shape = grid_shape
        self.hr_dim = hr_dim
        self.scale = scale

        for task in tasks.values():
            task.set_camera_encoders(self.camera_encoders)
        for task in hr_tasks.values():
            task.set_camera_encoders(self.camera_encoders)

        self.tasks = nn.ModuleDict(tasks)

        assert (len(tasks) + len(hr_tasks)) > 0, "no tasks specified"
        self.hr_tasks = nn.ModuleDict(hr_tasks)
        if len(hr_tasks) > 0:
            assert hr_dim is not None, "must specify hr_dim for hr_tasks"

        # used for torchscript inference export
        self.infer_batch: Optional[Batch] = None

        # cam masking
        self.cam_features_mask_ratio = cam_features_mask_ratio
        self.cam_mask_value = nn.Embedding(1, cam_dim)

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        per_cam_params = list(self.camera_encoders.parameters())
        per_cam_lr = lr
        per_cam_lr /= len(self.cameras)
        groups: List[Dict[str, object]] = [
            {
                "name": "backbone",
                "params": list(self.backbone.parameters())
                + list(self.cam_mask_value.parameters()),
                "lr": lr,
            },
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
        log_img, log_text = self.should_log(global_step, BS)

        start_frame = self.num_encode_frames - 1

        batch = self.transform(batch)

        last_cam_feats = {}

        # optionally dropout a camera to prevent overfitting
        drop_cameras = random.choices(self.cameras, k=self.num_drop_encode_cameras)
        dropout_cameras = set(self.cameras) - set(drop_cameras)

        with autocast(), torch.autograd.profiler.record_function("camera_feats"):
            # individual frames
            camera_feats = {cam: [] for cam in dropout_cameras}

            # frames we want backprop for
            for cam in dropout_cameras:
                # run frames in parallel
                encoder = self.camera_encoders[cam]
                encoder.train()
                inp = [batch.grid_image(cam, i) for i in range(self.num_encode_frames)]

                # use gradient checkpointing to save memory
                feats = [
                    torch.utils.checkpoint.checkpoint(
                        encoder, frame_inp, use_reentrant=False
                    )
                    for frame_inp in inp
                ]
                num_frames = self.num_encode_frames

                to_mask = random_block_mask(
                    feats[0],
                    block_size=(25, 25),
                    num_blocks=8,
                )
                # TODO: support data normal masks
                # to_mask = torch.bitwise_and(to_mask, F.interpolate(batch.mask[cam], to_mask.shape) > 0.5)

                if log_img and writer:
                    writer.add_image(
                        f"mask/{cam}",
                        render_color(to_mask),
                        global_step=global_step,
                    )
                for i, feat in enumerate(feats):
                    feat[:, :, ~to_mask] = (
                        self.cam_mask_value.weight.unsqueeze(0)
                        .unsqueeze(-1)
                        .to(feat.dtype)
                    )

                camera_feats[cam] = feats

        for cam, cam_feats in camera_feats.items():
            if torch.is_anomaly_check_nan_enabled():
                for feats in cam_feats:
                    assert not torch.isnan(feats).any().item(), cam

        with torch.autograd.profiler.record_function("backbone"):
            feat = camera_feats[cam][start_frame]
            target_grid = Grid3d.from_volume(
                # Grid3d reference frame is [z, y, x]
                data=torch.empty(
                    feat.size(0),
                    self.hr_dim,
                    *self.grid_shape[::-1],
                    device=feat.device,
                    dtype=feat.dtype,
                ),
                voxel_size=1.0 / self.scale,
                time=feat.time,
                volume_translation=(0, 0, -4.0 / self.scale),
            )
            hr_bev, bev_feats, bev_intermediates = self.backbone(
                batch, camera_feats, target_grid
            )

        if torch.is_anomaly_check_nan_enabled():
            assert not torch.isnan(hr_bev).any().item()
            for feat in bev_feats:
                assert not torch.isnan(feat).any().item()
            for key, feat in bev_intermediates.items():
                assert not torch.isnan(feat).any().item(), key

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
            cam_feats=camera_feats,
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

        return losses

    def infer_backbone(
        self, camera_features: Dict[str, List[torch.Tensor]], T: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        infer_backbone takes in the preprocessed camera features and runs the
        backbone and tasks. This requires the camera encoders to be run
        separately.
        """

        batch = self.infer_batch

        # set vehicle path transformation matrix
        batch = replace(batch, cam_T=T)

        assert batch is not None, "must have infer_batch set before running inference"

        hr_bev, bev_feats, _ = self.backbone(camera_features, batch)

        out = {}

        for name, task in self.tasks.items():
            out[name] = task.infer(hr_bev, bev_feats)

        for name, task in self.hr_tasks.items():
            out[name] = task.infer(hr_bev, bev_feats)

        return out
