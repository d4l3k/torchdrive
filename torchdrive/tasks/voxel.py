import os.path
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from pytorch3d.renderer import NDCMultinomialRaysampler, VolumeRenderer
from pytorch3d.structures import Volumes
from torch import nn

from torchdrive.amp import autocast
from torchdrive.autograd import autograd_context
from torchdrive.data import Batch
from torchdrive.losses import projection_loss, tvl1_loss
from torchdrive.models.regnet import resnet_init
from torchdrive.models.semantic import BDD100KSemSeg
from torchdrive.raymarcher import CustomPerspectiveCameras, DepthEmissionRaymarcher
from torchdrive.tasks.bev import BEVTask, Context
from torchdrive.transforms.depth import BackprojectDepth, Project3D
from torchdrive.transforms.img import normalize_img, render_color


def axis_grid(grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a colored box occupancy grid and color grid.

    Returns:
        grid: [BS, 1, X, Y, Z]
        color_grid: [BS, 3, X, Y, Z]
    """
    device: torch.device = grid.device

    w, d, h = grid.shape[2:]

    grid = torch.zeros_like(grid)
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
        device: torch.device,
        scale: int = 3,
        semantic: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.cam_shape = cam_shape
        self.cameras = cameras
        self.scale = scale
        self.height = height
        self.semantic = semantic

        # generate voxel grid
        self.num_elem: int = 1
        if semantic:
            self.classes_elem: int = len(BDD100KSemSeg.INTERESTING)
            self.vel_elem: int = 3
            self.num_elem += self.classes_elem + self.vel_elem
            self.segment: BDD100KSemSeg = BDD100KSemSeg(device=device)

        self.decoder = nn.Conv2d(dim, self.num_elem * height, kernel_size=1)
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
        raymarcher = DepthEmissionRaymarcher(
            floor=0,
        )
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
        cam_T = start_T.unsqueeze(1).pinverse().matmul(batch.cam_T)
        device = bev.device

        bev_shape = bev.shape[2:]

        with autocast():
            embedding = self.decoder(bev).unflatten(1, (self.num_elem, self.height))

        grid = embedding[:, :1].sigmoid_()
        feat_grid = embedding[:, 1:]

        grid = grid.permute(0, 1, 4, 3, 2)
        feat_grid = feat_grid.permute(0, 1, 4, 3, 2)
        # grid, color_grid = axis_grid(grid)

        losses = {}

        grad_tensors = [grid]
        if self.semantic:
            grad_tensors.append(feat_grid)
        with autograd_context(*grad_tensors) as packed:
            if self.semantic:
                grid, feat_grid = packed
            else:
                grid = packed

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

                voxel_center = torch.tensor(
                    (bev_shape[0] / 2, bev_shape[1] / 2, 0),
                    device=device,
                    dtype=torch.float32,
                )

                zero_coord = torch.zeros(1, 4, device=device, dtype=torch.float)
                zero_coord[:, -1] = 1
                for frame in range(0, frames):
                    T = cam_T[:, frame]
                    cam_coords = torch.matmul(T, zero_coord.T)
                    coord = cam_coords[0, :3, 0]
                    x, y, z = (coord * self.scale + voxel_center).int()
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
                        grid[0].detach().float().cpu().numpy(),
                        allow_pickle=False,
                    )

            losses = {}

            # total variation loss to encourage sharp edges
            losses["tvl1"] = tvl1_loss(grid.squeeze(1)) * 0.01 * 40

            h, w = self.cam_shape

            for cam in self.cameras:
                cam_semantic = self.semantic and cam in self.semantic
                volumes = Volumes(
                    densities=grid.permute(0, 1, 4, 3, 2).float(),
                    features=feat_grid.permute(0, 1, 4, 3, 2).float()
                    if cam_semantic
                    else None,
                    voxel_size=1 / self.scale,
                    # TODO support non-centered voxel grids
                    # volume_translation=(-self.height / 2 / self.scale, 0, 0),
                    volume_translation=(0, 0, -self.height / 2 / self.scale),
                )

                for frame in range(0, frames - 1, 2):
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
                        # ctx.add_image(
                        #    f"color/{cam}/{frame}",
                        #    semantic_img[0],
                        # )

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
                    half_color = F.interpolate(
                        color, [h // 2, w // 2], mode="bilinear", align_corners=False
                    )
                    color = half_color

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
                        losses[f"lossproj/{cam}/o{offset}/s{scale}"] = (
                            proj_loss.mean(dim=(1, 2, 3)) * 40
                        )

                    if cam_semantic:
                        losses[f"semantic/{cam}/o{offset}"] = self._semantic_loss(
                            ctx, cam, frame, semantic_img, half_color, primary_mask
                        )

                    ctx.backward(losses)

        return losses

    def _semantic_loss(
        self,
        ctx: Context,
        cam: str,
        frame: int,
        semantic_img: torch.Tensor,
        color: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        _semantic_loss computes the semantic class probability loss.
        """
        semantic_img = semantic_img.permute(0, 3, 1, 2)
        semantic_classes = semantic_img[:, : self.classes_elem]
        semantic_vel = semantic_img[:, self.classes_elem :]

        semantic_target = self.segment(color)
        # select interesting classes and convert to probabilities
        semantic_target = semantic_target[:, BDD100KSemSeg.INTERESTING]
        semantic_target = F.avg_pool2d(semantic_target, 2)
        semantic_target = semantic_target.sigmoid()

        sem_loss = F.binary_cross_entropy_with_logits(
            semantic_classes, semantic_target, reduction="none"
        )
        sem_loss *= F.avg_pool2d(mask, 2)

        if ctx.log_img:
            out_class = torch.argmax(semantic_img[:1], dim=1)
            target_class = torch.argmax(semantic_target[:1], dim=1)
            ctx.add_image(
                f"semantic/{cam}/{frame}/output_target",
                render_color(
                    torch.cat(
                        (
                            out_class[0],
                            target_class[0],
                        ),
                        dim=1,
                    )
                ),
            )
            ctx.add_image(
                f"semantic/{cam}/{frame}/loss",
                render_color(sem_loss[0].mean(dim=0)),
            )

        return sem_loss.mean(dim=(1, 2, 3))

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
