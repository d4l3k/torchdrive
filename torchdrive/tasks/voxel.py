import os.path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

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
from torchdrive.models.depth import DepthDecoder
from torchdrive.models.regnet import resnet_init
from torchdrive.models.semantic import BDD100KSemSeg
from torchdrive.raymarcher import CustomPerspectiveCameras, DepthEmissionRaymarcher
from torchdrive.tasks.bev import BEVTask, Context
from torchdrive.transforms.depth import BackprojectDepth, disp_to_depth, Project3D
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
        hr_dim: int,
        height: int,
        device: torch.device,
        scale: int = 3,
        semantic: Optional[List[str]] = None,
        render_batch_size: int = 2,
        n_pts_per_ray: int = 216,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        super().__init__()

        self.cam_shape = cam_shape
        self.cameras = cameras
        self.scale = scale
        self.height = height
        self.semantic = semantic
        self.render_batch_size = render_batch_size

        # generate voxel grid
        self.num_elem: int = 1
        background: List[float] = []
        if semantic:
            self.classes_elem: int = len(BDD100KSemSeg.INTERESTING)
            background += [-10.0] * self.classes_elem
            self.vel_elem: int = 3
            background += [0.0, 0.0, 0.0]
            self.num_elem += self.classes_elem + self.vel_elem
            self.segment: BDD100KSemSeg = BDD100KSemSeg(
                device=device,  # compile_fn=compile_fn
            )

        h, w = cam_shape

        self.decoder = nn.Conv2d(hr_dim, self.num_elem * height, kernel_size=1)
        resnet_init(self.decoder)

        self.backproject_depth: nn.Module = compile_fn(BackprojectDepth(h // 2, w // 2))
        self.project_3d: nn.Module = compile_fn(Project3D(h // 2, w // 2))

        self.max_depth: float = n_pts_per_ray / scale
        self.min_depth = 0.5
        raysampler = NDCMultinomialRaysampler(
            image_width=w // 4,
            image_height=h // 4,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )
        raymarcher = DepthEmissionRaymarcher(
            floor=0,
            background=torch.tensor(background, device=device)
            if len(background) > 0
            else None,
        )
        self.renderer = VolumeRenderer(
            raysampler=raysampler,
            raymarcher=compile_fn(raymarcher),
        )

        # image space model
        self.depth_decoder: nn.Module = compile_fn(
            DepthDecoder(
                num_upsamples=2,
                cam_shape=(h // 16, w // 16),
                dim=dim,
            )
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

        grad_tensors = [grid]
        if self.semantic:
            grad_tensors.append(feat_grid)
        with autograd_context(*grad_tensors) as packed:
            if self.semantic:
                grid, feat_grid = packed
            else:
                grid = packed
                feat_grid = None

            losses = {}

            # total variation loss to encourage sharp edges
            losses["tvl1"] = tvl1_loss(grid.squeeze(1)) * 0.01 * 400

            mini_batches = batch.split(self.render_batch_size)
            mini_grids = torch.split(grid, self.render_batch_size)
            mini_feats = (
                torch.split(feat_grid, self.render_batch_size)
                if feat_grid is not None
                else [None] * len(mini_grids)
            )
            cam_feats = ctx.cam_feats
            mini_cam_feats = _split_dict(ctx.cam_feats, self.render_batch_size)

            assert len(mini_batches) == len(mini_grids)
            assert len(mini_batches) == len(mini_feats)
            assert len(mini_batches) == len(mini_cam_feats)

            for i, (mini_batch, mini_grid, mini_feat, mini_cam_feat) in enumerate(
                zip(mini_batches, mini_grids, mini_feats, mini_cam_feats)
            ):
                ctx.weights = mini_batch.weight
                ctx.cam_feats = mini_cam_feat
                sub_losses = self._losses(ctx, mini_batch, mini_grid, mini_feat)
                for k, v in sub_losses.items():
                    if k in losses:
                        losses[k] += v
                    else:
                        losses[k] = v

            ctx.weights = batch.weight
            ctx.cam_feats = cam_feats

        return losses

    def _losses(
        self,
        ctx: Context,
        batch: Batch,
        grid: torch.Tensor,
        feat_grid: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        compute losses for the provided minibatch.
        """
        BS = len(batch.distances)
        frames = batch.distances.shape[1]
        start_frame = ctx.start_frame
        start_T = batch.cam_T[:, ctx.start_frame]
        cam_T = start_T.unsqueeze(1).pinverse().matmul(batch.cam_T)
        frame_time = batch.frame_time - batch.frame_time[:, ctx.start_frame].unsqueeze(
            1
        )
        device = grid.device

        losses = {}

        h, w = self.cam_shape

        for cam in self.cameras:
            volumes = Volumes(
                densities=grid.permute(0, 1, 4, 3, 2).float(),
                features=feat_grid.permute(0, 1, 4, 3, 2).float()
                if feat_grid is not None
                else None,
                voxel_size=1 / self.scale,
                # TODO support non-centered voxel grids
                # volume_translation=(-self.height / 2 / self.scale, 0, 0),
                volume_translation=(0, 0, -self.height / 2 / self.scale),
            )

            K = batch.K[cam]
            T = batch.T[cam]
            cameras = CustomPerspectiveCameras(
                T=T,
                K=K,
                image_size=torch.tensor(
                    [[h // 2, w // 2]], device=device, dtype=torch.float
                ).expand(BS, -1),
                device=device,
            )
            with torch.autograd.profiler.record_function("render"):
                (voxel_depth, semantic_img), ray_bundle = self.renderer(
                    cameras=cameras,
                    volumes=volumes,
                    eps=1e-8,
                )
            if semantic_img is not None:
                semantic_img = semantic_img.permute(0, 3, 1, 2)

            frame = ctx.start_frame

            primary_color = batch.color[cam, frame]
            primary_color = F.interpolate(
                primary_color.float(),
                [h // 2, w // 2],
                mode="bilinear",
                align_corners=False,
            )
            primary_mask = batch.mask[cam]
            primary_mask = F.interpolate(
                primary_mask.float(),
                [h // 2, w // 2],
                mode="bilinear",
                align_corners=False,
            )

            per_pixel_weights = torch.ones_like(primary_color)

            if self.semantic:
                semantic_loss, dynamic_mask = self._semantic_loss(
                    ctx, cam, semantic_img, primary_color, primary_mask
                )
                # losses[f"semantic/{cam}"] = semantic_loss * 100

                semantic_vel = semantic_img[:, self.classes_elem :]
                semantic_vel = F.interpolate(
                    semantic_vel.float(),
                    [h // 2, w // 2],
                    mode="bilinear",
                    align_corners=False,
                )
                if ctx.log_img:
                    ctx.add_image(
                        f"{cam}/semantic_vel",
                        normalize_img(semantic_vel[0]),
                    )
                    ctx.add_image(
                        f"{cam}/dynamic_mask",
                        render_color(dynamic_mask[0]),
                    )
                dynamic_mask = dynamic_mask.unsqueeze(1).expand(-1, 3, -1, -1)
                semantic_vel *= dynamic_mask
                semantic_vel *= primary_mask

                # focus more on dynamic objects
                per_pixel_weights += dynamic_mask
                # normalize mean
                per_pixel_weights /= per_pixel_weights.mean()
            else:
                dynamic_mask = torch.zeros_like(primary_color)
                semantic_vel = torch.zeros_like(primary_color)

            self._depth_loss(
                ctx=ctx,
                label="voxel",
                cam=cam,
                depth=voxel_depth,
                semantic_vel=semantic_vel,
                losses=losses,
                h=h,
                w=w,
                batch=batch,
                frame=frame,
                frame_time=frame_time,
                primary_color=primary_color,
                primary_mask=primary_mask,
                per_pixel_weights=per_pixel_weights,
            )
            del voxel_depth
            del semantic_vel
            del semantic_img

            with torch.autograd.profiler.record_function("depth_decoder"), autocast():
                cam_disp, cam_vel = self.depth_decoder(ctx.cam_feats[cam])

            cam_vel = F.interpolate(
                cam_vel.float(),
                [h // 2, w // 2],
                mode="bilinear",
                align_corners=False,
            )
            cam_vel *= dynamic_mask
            cam_vel *= primary_mask
            cam_depth = disp_to_depth(
                cam_disp.float().sigmoid(),
                min_depth=self.min_depth,
                max_depth=self.max_depth,
            )

            self._depth_loss(
                ctx=ctx,
                label="cam",
                cam=cam,
                depth=cam_depth,
                semantic_vel=cam_vel,
                losses=losses,
                h=h,
                w=w,
                batch=batch,
                frame=frame,
                frame_time=frame_time,
                primary_color=primary_color,
                primary_mask=primary_mask,
                per_pixel_weights=per_pixel_weights,
            )

            del cam_vel
            del cam_disp
            del cam_depth

        return losses

    def _depth_loss(
        self,
        ctx: Context,
        label: str,
        cam: str,
        depth: torch.Tensor,
        semantic_vel: torch.Tensor,
        losses: Dict[str, torch.Tensor],
        h: int,
        w: int,
        batch: Batch,
        frame: int,
        frame_time: torch.Tensor,
        primary_color: torch.Tensor,
        primary_mask: torch.Tensor,
        per_pixel_weights: torch.Tensor,
    ) -> None:
        depth = F.interpolate(
            depth.float().unsqueeze(1),
            [h // 2, w // 2],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        depth = depth.unsqueeze(1)
        # upscale to original size
        if ctx.log_text:
            amin, amax = depth.aminmax()
            assert amin >= 0, (amin, amax)
            ctx.add_scalars(
                f"depth{label}/{cam}/minmax",
                {"max": amax, "min": amin},
            )

        if ctx.log_img:
            ctx.add_image(
                f"depth{label}/{cam}",
                render_color(-depth[0][0]),
            )
            ctx.add_image(
                f"disp{label}/{cam}",
                # pyre-fixme[6]: float / tensor
                render_color(1 / (depth[0][0] + 1e-7)),
            )

        for offset in [-1, 1]:
            T = batch.frame_T[:, frame + offset]
            if offset < 0:
                T = T.pinverse()
            time = frame_time[:, frame + offset]

            projcolor, projmask = self.project(
                batch,
                cam,
                T,
                depth,
                primary_color,
                primary_mask,
                semantic_vel * time.reshape(-1, 1, 1, 1),
            )
            projmask *= primary_mask
            color = batch.color[cam, frame + offset]
            half_color = F.interpolate(
                color.float(),
                [h // 2, w // 2],
                mode="bilinear",
                align_corners=False,
            )
            color = half_color
            proj_weights = per_pixel_weights

            MSSIM_SCALES = 3
            for scale in range(MSSIM_SCALES):
                if scale > 0:
                    projcolor = F.avg_pool2d(projcolor, 2)
                    projmask = F.avg_pool2d(projmask, 2)
                    color = F.avg_pool2d(color, 2)
                    proj_weights = F.avg_pool2d(proj_weights, 2)
                proj_loss = (
                    projection_loss(projcolor, color, projmask) / MSSIM_SCALES
                ) * proj_weights

                if ctx.log_img:
                    ctx.add_image(
                        f"{label}/{cam}/{offset}/{scale}/color",
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
                        f"{label}/{cam}/{offset}/{scale}/color_err",
                        normalize_img((color[0] - projcolor[0]).abs()),
                    )
                    ctx.add_image(
                        f"{label}/{cam}/{offset}/{scale}/proj_loss",
                        render_color(proj_loss[0][0]),
                    )
                    ctx.add_image(
                        f"{label}/{cam}/{offset}/{scale}/proj_mask",
                        render_color(projmask[0][0]),
                    )
                losses[f"lossproj-{label}/{cam}/o{offset}/s{scale}"] = (
                    proj_loss.mean(dim=(1, 2, 3)) * 40
                )

        ctx.backward(losses)

    def _semantic_loss(
        self,
        ctx: Context,
        cam: str,
        semantic_img: torch.Tensor,
        color: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        _semantic_loss computes the semantic class probability loss.
        """
        semantic_classes = semantic_img[:, : self.classes_elem].sigmoid()

        segmentation_target = self.segment(color)
        # select interesting classes and convert to probabilities
        semantic_target = segmentation_target[:, BDD100KSemSeg.INTERESTING]
        semantic_target = F.avg_pool2d(semantic_target, 2).sigmoid()

        sem_loss = F.huber_loss(
            semantic_classes.float(), semantic_target.float(), reduction="none"
        )
        sem_loss *= F.avg_pool2d(mask, 2)

        if ctx.log_text:
            pred_min, pred_max = semantic_classes.aminmax()
            targ_min, targ_max = semantic_target.aminmax()
            ctx.add_scalars(
                f"semantic/{cam}/minmax",
                {
                    "pred_min": pred_min,
                    "pred_max": pred_max,
                    "pred_mean": semantic_classes.mean(),
                    "targ_min": targ_min,
                    "targ_max": targ_max,
                    "targ_mean": semantic_target.mean(),
                },
            )

        if ctx.log_img:
            out_class = torch.argmax(semantic_img[:1], dim=1)
            target_class = torch.argmax(semantic_target[:1], dim=1)
            ctx.add_image(
                f"semantic/{cam}/output_target",
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
                f"semantic/{cam}/loss",
                render_color(sem_loss[0].mean(dim=0)),
            )

        dynamic_mask = segmentation_target[:, BDD100KSemSeg.DYNAMIC]
        dynamic_mask = dynamic_mask.sigmoid().amax(dim=1).round()

        return (
            sem_loss.mean(dim=(1, 2, 3)),
            dynamic_mask,
        )

    def project(
        self,
        batch: Batch,
        cam: str,
        cam_T: torch.Tensor,
        depth: torch.Tensor,
        color: torch.Tensor,
        mask: torch.Tensor,
        vel: torch.Tensor,
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

        world_points = self.backproject_depth(depth, target_inv_K, batch.T[cam]).clone()

        # add velocity to points
        world_points[:, :3] += vel.flatten(-2, -1)

        # (world to cam) * camera motion
        T = batch.T[cam].pinverse().matmul(cam_T)
        pix_coords = self.project_3d(world_points, src_K, T)

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


def _split_dict(
    d: Mapping[str, torch.Tensor], size: int
) -> List[Dict[str, torch.Tensor]]:
    out = []
    for k, v in d.items():
        for i, part in enumerate(torch.split(v, size)):
            if len(out) <= i:
                out.append({})
            out[i][k] = part
    return out
