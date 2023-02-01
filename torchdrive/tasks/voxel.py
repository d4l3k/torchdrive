import os.path
from typing import Dict, List, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from pytorch3d.renderer import NDCMultinomialRaysampler, VolumeRenderer
from pytorch3d.structures import Volumes
from torch import nn

from torchdrive.amp import autocast
from torchdrive.autograd import autograd_context
from torchdrive.data import Batch
from torchdrive.losses import losses_backward, projection_loss, tvl1_loss
from torchdrive.models.regnet import resnet_init
from torchdrive.raymarcher import CustomPerspectiveCameras, DepthEmissionRaymarcher
from torchdrive.tasks.bev import BEVTask, Context
from torchdrive.transforms.depth import BackprojectDepth, Project3D
from torchdrive.transforms.img import normalize_img, render_color


class VoxelTask(BEVTask):
    """
    Voxel occupancy grid task. This takes in the high resolution BEV map and
    converts it into a voxel grid and then uses differentiable rendering losses
    to generate depth maps that use a SSIM projection loss to learn the
    occupancy grid.
    """

    def __init__(
        self,
        cameras: List[str],
        cam_shape: Tuple[int, int],
        dim: int,
        height: int,
        scale: int = 3,
    ) -> None:
        super().__init__()

        self.cam_shape = cam_shape
        self.cameras = cameras
        self.scale = scale

        # generate voxel grid
        self.decoder = nn.Conv2d(dim, height, kernel_size=1)
        resnet_init(self.decoder)

        h, w = cam_shape

        self.backproject_depth = BackprojectDepth(h // 2, w // 2)
        self.project_3d = Project3D(h // 2, w // 2)

        raysampler = NDCMultinomialRaysampler(
            image_width=w // 4,
            image_height=h // 4,
            n_pts_per_ray=216,
            min_depth=0.1,
            max_depth=216 / scale,
        )
        raymarcher = DepthEmissionRaymarcher()
        self.renderer = VolumeRenderer(
            raysampler=raysampler,
            raymarcher=raymarcher,
        )

    def forward(
        self, ctx: Context, batch: Batch, bev: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        BS = len(batch.distances)
        frames = batch.distances.shape[1]
        start_frame = ctx.start_frame
        start_T = batch.cam_T[:, ctx.start_frame]
        cam_T = start_T.pinverse().matmul(batch.cam_T)
        device = bev.device

        with autocast():
            grid = self.decoder(bev).unsqueeze(1).sigmoid_()

        grid = grid.permute(0, 1, 4, 3, 2)

        losses = {}
        with autograd_context(grid) as grid:
            if ctx.log_text:
                ctx.add_scalars(
                    "grid/minmax",
                    {"max": grid.max(), "min": grid.min(), "mean": grid.mean()},
                )
            if ctx.log_img:
                ctx.add_image(
                    "grid/x",
                    render_color(grid[0, 0].sum(dim=0).permute(1, 0)),
                )
                ctx.add_image(
                    "grid/y",
                    render_color(grid[0, 0].sum(dim=1).permute(1, 0)),
                )

                gz = render_color(grid[0, 0].sum(dim=2))

                zero_coord = torch.zeros(1, 4, device=device, dtype=torch.float)
                zero_coord[:, -1] = 1
                for frame in range(start_frame, frames):
                    T = cam_T[:, frame]
                    cam_coords = torch.matmul(T, zero_coord.T)
                    coord = cam_coords[0, :3, 0]
                    x, y, z = (coord * self.scale).int()
                    _, d, w = gz.shape
                    if x >= d or y >= w or x < 0 or y < 0:
                        continue
                    gz[:, x, y] = torch.tensor((0, 1, 0))

                ctx.add_image(
                    "grid/z",
                    gz,
                )

                if ctx.output:
                    # save a npz file
                    np.save(
                        os.path.join(ctx.output, f"grid_{ctx.global_step}.npy"),
                        grid[0].cpu().detach().numpy(),
                        allow_pickle=False,
                    )

            volumes = Volumes(
                densities=grid,
                voxel_size=1 / self.scale,
                volume_translation=(0, 0, 0),  # TODO support noncentered voxel grids
            )

            losses = {}

            # total variation loss to encourage sharp edges
            losses["tvl1"] = tvl1_loss(grid.squeeze(1)) * 0.01

            h, w = self.cam_shape

            frame_projects = {
                0: self.cameras,
                2: self.cameras,
                4: self.cameras,
            }

            for frame in range(0, frames, 2):
                for cam in self.cameras:
                    K = batch.K[cam]
                    T = torch.matmul(cam_T[:, frame], batch.T[cam])
                    cameras = CustomPerspectiveCameras(
                        T=T,
                        K=K,
                        image_size=torch.tensor(
                            [[h // 2, w // 2]], device=device, dtype=torch.float
                        ).expand(BS, -1),
                        device=device,
                    )

                    (depth, semantic_img), ray_bundle = self.renderer(
                        cameras=cameras,
                        volumes=volumes,
                        eps=1e-8,
                    )

                    depth = F.interpolate(
                        depth.unsqueeze(1),
                        [h // 2, w // 2],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    depth = depth.unsqueeze(1)
                    # upscale to original size
                    if ctx.log_text:
                        ctx.add_scalars(
                            f"depth/{cam}/{frame}/minmax",
                            {"max": depth.max(), "min": depth.min()},
                        )

                    if ctx.log_img:
                        ctx.add_image(
                            f"depth/{cam}/{frame}",
                            render_color(-depth[0][0]),
                        )
                        ctx.add_image(
                            f"disp/{cam}/{frame}",
                            # pyre-fixme[6]: float / tensor
                            render_color(1 / (depth[0][0] + 1e-7)),
                        )

                    offset = frame + 1
                    T = batch.frame_T[:, offset]
                    primary_color = batch.color[cam, frame]
                    primary_color = F.interpolate(
                        primary_color,
                        [h // 2, w // 2],
                        mode="bilinear",
                        align_corners=False,
                    )
                    primary_mask = batch.mask[cam]
                    primary_mask = F.interpolate(
                        primary_mask,
                        [h // 2, w // 2],
                        mode="bilinear",
                        align_corners=False,
                    )
                    projcolor, projmask = self.project(
                        batch, cam, T, depth, primary_color, primary_mask
                    )
                    projmask *= primary_mask
                    color = batch.color[cam, offset]
                    color = F.interpolate(
                        color, [h // 2, w // 2], mode="bilinear", align_corners=False
                    )

                    MSSIM_SCALES = 3
                    for scale in range(MSSIM_SCALES):
                        if scale > 0:
                            projcolor = F.avg_pool2d(projcolor, 2)
                            projmask = F.avg_pool2d(projmask, 2)
                            color = F.avg_pool2d(color, 2)
                        proj_loss = (
                            projection_loss(projcolor, color, projmask) / MSSIM_SCALES
                        )

                        if ctx.log_img:
                            ctx.add_image(
                                f"{cam}/{offset}/{scale}/color",
                                normalize_img(
                                    torch.cat(
                                        (
                                            color[0],
                                            projcolor[0],
                                        ),
                                        dim=2,
                                    )
                                ),
                            )
                            ctx.add_image(
                                f"{cam}/{offset}/{scale}/color_err",
                                normalize_img((color[0] - projcolor[0]).abs()),
                            )
                            ctx.add_image(
                                f"{cam}/{offset}/{scale}/proj_loss",
                                render_color(proj_loss[0][0]),
                            )
                            ctx.add_image(
                                f"{cam}/{offset}/{scale}/proj_mask",
                                render_color(projmask[0][0]),
                            )
                        losses[f"lossproj/{cam}/o{offset}/s{scale}"] = proj_loss.mean(
                            dim=(1, 2, 3)
                        )

                    losses_backward(losses, ctx.scaler)

        return losses

    def project(
        self,
        batch: Batch,
        cam: str,
        cam_T: torch.Tensor,
        depth: torch.Tensor,
        color: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_K = batch.K[cam].clone()
        # convert to image space
        src_K[:, 0] *= self.backproject_depth.width
        src_K[:, 1] *= self.backproject_depth.height

        target_K = batch.K[cam].clone()
        # convert to image space
        target_K[:, 0] *= self.backproject_depth.width
        target_K[:, 1] *= self.backproject_depth.height
        target_inv_K = target_K.pinverse()

        cam_points = self.backproject_depth(depth, target_inv_K)

        T = batch.T[cam].pinverse().matmul(cam_T).matmul(batch.T[cam])

        pix_coords = self.project_3d(cam_points, src_K, T)

        color = F.grid_sample(
            color,
            pix_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        mask = F.grid_sample(
            mask,
            pix_coords,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        return color, mask
