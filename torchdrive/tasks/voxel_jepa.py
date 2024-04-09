import os.path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F
import torchmetrics
from pytorch3d.renderer import ImplicitRenderer, NDCMultinomialRaysampler
from pytorch3d.structures import Volumes
from pytorch3d.structures.volumes import VolumeLocator
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torchworld.transforms.img import normalize_img, normalize_mask, render_color

from torchdrive.amp import autocast
from torchdrive.autograd import autograd_context
from torchdrive.data import Batch
from torchdrive.losses import multi_scale_projection_loss, smooth_loss, tvl1_loss
from torchdrive.models.regnet import resnet_init
from torchdrive.render.raymarcher import (
    CustomPerspectiveCameras,
    DepthEmissionRaymarcher,
    # DepthEmissionSoftmaxRaymarcher,
)
from torchworld.transforms.grid_sampler import GridSampler
from torchdrive.tasks.bev import BEVTask, Context


def axis_grid(grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a colored box occupancy grid and color grid.

    Returns:
        grid: [BS, 1, X, Y, Z]
        color_grid: [BS, 3, X, Y, Z]
    """
    device: torch.device = grid.device

    w, d, h = grid.shape[2:]

    # preserve grad connection
    grid = grid * 0
    grid[:, :, 0, :, :] = 1
    grid[:, :, -1, :, :] = 1
    grid[:, :, :, 0, :] = 1
    grid[:, :, :, -1, :] = 1
    grid[:, :, :, :, 0] = 1
    grid[:, :, :, :, -1] = 1

    def color(r: float, g: float, b: float, x: int, y: int) -> torch.Tensor:
        return (
            torch.tensor([r, g, b], device=device, dtype=torch.float)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand(3, x, y)
        )

    color_grid = torch.zeros_like(grid).repeat(1, 3, 1, 1, 1)
    color_grid[:, :, 0, :, :] = color(1, 0, 0, d, h)
    color_grid[:, :, -1, :, :] = color(1, 0, 1, d, h)
    color_grid[:, :, :, 0, :] = color(0, 1, 0, w, h)
    color_grid[:, :, :, -1, :] = color(0, 1, 1, w, h)
    color_grid[:, :, :, :, 0] = color(0, 0, 1, w, d)
    color_grid[:, :, :, :, -1] = color(1, 1, 1, w, d)

    return grid, color_grid


class VoxelJEPATask(BEVTask):
    """
    Voxel JEPA task.
    """

    def __init__(
        self,
        cameras: List[str],
        cam_shape: Tuple[int, int],
        cam_feats_shape: Tuple[int, int],
        dim: int,
        hr_dim: int,
        cam_dim: int,
        height: int,
        device: torch.device,
        scale: int = 3,
        z_offset: float = 0.5,
        render_batch_size: int = 1000,
        n_pts_per_ray: int = 216,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda x: x,
        start_offsets: Tuple[int, ...] = (0,),
        offsets: Tuple[int, ...] = (-2, -1, 0, 1, 2),
    ) -> None:
        super().__init__()

        self.cam_shape = cam_shape
        self.cameras = cameras
        self.scale = scale
        self.height = height
        self.render_batch_size = render_batch_size
        self.start_offsets = start_offsets
        self.offsets = offsets
        self.hr_dim = hr_dim
        self.dim = dim

        # TODO support non-centered voxel grids
        self.volume_translation: Tuple[float, float, float] = (
            0,
            0,
            -self.height * z_offset / self.scale,
        )

        # generate voxel grid
        h, w = cam_shape

        voxel_dim = max(hr_dim // height, 1)
        self.voxel_dim = voxel_dim
        self.decoders = nn.ModuleDict(
            {
                str(offset): compile_fn(
                    nn.Conv3d(voxel_dim, voxel_dim + 1, kernel_size=1)
                )
                for offset in offsets
            }
        )
        resnet_init(self.decoders)

        self.max_depth: float = n_pts_per_ray / scale
        self.min_depth = 2#0.5
        self.image_width = w // 8
        self.image_height = h // 8
        self.raysampler = NDCMultinomialRaysampler(
            image_width=self.image_width,
            image_height=self.image_height,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.raymarcher = compile_fn(
            # DepthEmissionSoftmaxRaymarcher()
            DepthEmissionRaymarcher(
                floor=None,
                background=None,
                voxel_size=1 / scale,
            )
        )
        self.renderer = ImplicitRenderer(
            raysampler=self.raysampler,
            raymarcher=self.raymarcher,
        )

        # pyre-fixme[6]: nn.Module
        self.projection_loss: nn.Module = compile_fn(multi_scale_projection_loss)

    def set_camera_encoders(self, camera_encoders: nn.ModuleDict) -> None:
        self.ema_camera_encoders = AveragedModel(
            camera_encoders,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.998),
        )
        # use a list so it's not tracked as a module
        self.camera_encoders = [camera_encoders]

    def forward(
        self, ctx: Context, batch: Batch, grids: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        assert len(grids) == 1
        bev: torch.Tensor = grids[0]

        BS = len(batch.distances)
        frames = batch.distances.shape[1]
        device = bev.device

        bev_shape = bev.shape[2:]

        self.ema_camera_encoders.update_parameters(self.camera_encoders[0])

        losses = {}

        for offset in self.offsets:
            frame = ctx.start_frame + offset

            with autocast():
                embedding = self.decoders[str(offset)](bev)

            # log grad norms on full embedding to make norms comparable
            grid = embedding[:, :1]
            # convert back to float so sigmoid works
            grid = grid.float().sigmoid()
            feat_grid = embedding[:, 1:]

            grid = grid.permute(0, 1, 4, 3, 2)
            feat_grid = feat_grid.permute(0, 1, 4, 3, 2)

            #grid, feat_grid = axis_grid(grid)
            #feat_grid = feat_grid.sum(dim=1, keepdim=True).expand(-1, self.voxel_dim, -1, -1, -1)

            if ctx.log_text:
                ctx.add_scalars(
                    f"grid/{frame}/minmax",
                    {"max": grid.max(), "min": grid.min(), "mean": grid.mean()},
                )
            if ctx.log_text:
                ctx.add_scalars(
                    f"feat_grid/{frame}/minmax",
                    {"max": feat_grid.max(), "min": feat_grid.min(), "mean": feat_grid.mean()},
                )
            if ctx.log_img:
                ctx.add_image(
                    f"grid/{frame}/z",
                    render_color(grid[0, 0].sum(dim=2)),
                )

            # total variation loss to encourage sharp edges
            # tvl1_grid = ctx.log_grad_norm(grid, "grad/norm/grid", "tvl1")
            # losses["tvl1"] = tvl1_loss(tvl1_grid.squeeze(1)) * 0.5

            losses.update(self._losses_frame(ctx, batch, grid, feat_grid, frame))

            # we need to run tvl1 loss early as the child losses don't run
            # backwards on the whole set of losses
            ctx.backward(losses)

        return losses

    def _losses_frame(
        self,
        ctx: Context,
        batch: Batch,
        grid: torch.Tensor,
        feat_grid: Optional[torch.Tensor],
        frame: int,
    ) -> Dict[str, torch.Tensor]:
        """
        compute losses for the provided minibatch starting at the specific frame
        """
        BS = len(batch.distances)
        frames = batch.distances.shape[1]
        frame_time = batch.frame_time - batch.frame_time[:, ctx.start_frame].unsqueeze(
            1
        )
        device = grid.device

        losses = {}

        h, w = self.cam_shape

        dtype = torch.bfloat16 if grid.is_cuda else torch.float32

        volumetric_function = GridSampler(
            densities=grid.permute(0, 1, 4, 2, 3).to(dtype),
            features=(
                feat_grid.permute(0, 1, 4, 2, 3).to(dtype)
                if feat_grid is not None
                else None
            ),
            # we set padding_mode so points outside the grid will inherit from
            # the border so we don't need custom endpoint logic
            padding_mode="border",
        )

        for cam in self.cameras:
            # create world to camera transformation matrix
            # T = batch.world_to_cam(cam, start_frame)
            cameras = batch.camera(cam, frame)
            with torch.autograd.profiler.record_function("render"):
                (
                    voxel_depth,
                    semantic_img,
                    visible_probs,
                    depth_probs,
                ), ray_bundle = self.renderer(
                    cameras=cameras,
                    volumetric_function=volumetric_function,
                    eps=1e-8,
                )
            semantic_img = semantic_img.permute(0, 3, 1, 2)

            with torch.no_grad(), autocast():
                color = batch.color[cam][:, frame]
                target_feats = self.ema_camera_encoders.module[cam](color).detach()
                # normalize target features on the channel dimension
                target_feats = F.layer_norm(
                    target_feats.permute(0, 2, 3, 1),
                    (self.voxel_dim,),
                ).permute(0, 3, 1, 2)

            losses[f"cam_features/{cam}/{frame}"] = F.l1_loss(
                semantic_img, target_feats
            )

            if ctx.log_text:
                ctx.add_scalars(
                    f"{cam}/{frame}/predicted-minmax",
                    {
                        "max": semantic_img.max(),
                        "min": semantic_img.min(),
                        "mean": semantic_img.mean(),
                    },
                )
                ctx.add_scalars(
                    f"{cam}/{frame}/target-minmax",
                    {
                        "max": target_feats.max(),
                        "min": target_feats.min(),
                        "mean": target_feats.mean(),
                    },
                )
                ctx.add_scalars(
                    f"{cam}/{frame}/depth-minmax",
                    {
                        "max": voxel_depth.max(),
                        "min": voxel_depth.min(),
                        "mean": voxel_depth.mean(),
                    },
                )
            if ctx.log_img:
                ctx.add_image(
                    f"{cam}/{frame}/target",
                    render_color(target_feats[0].sum(dim=0)),
                )
                ctx.add_image(
                    f"{cam}/{frame}/predicted",
                    render_color(semantic_img[0].sum(dim=0)),
                )
                ctx.add_image(
                    f"{cam}/{frame}/color",
                    normalize_img(color[0]),
                )
                ctx.add_image(
                    f"{cam}/{frame}/depth",
                    render_color(voxel_depth[0]),
                )

        return losses
