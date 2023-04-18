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
from torchdrive.losses import multi_scale_projection_loss, smooth_loss, tvl1_loss
from torchdrive.models.depth import DepthDecoder
from torchdrive.models.regnet import resnet_init
from torchdrive.models.semantic import BDD100KSemSeg
from torchdrive.raymarcher import CustomPerspectiveCameras, DepthEmissionRaymarcher
from torchdrive.tasks.bev import BEVTask, Context
from torchdrive.transforms.depth import (
    BackprojectDepth,
    depth_to_disp,
    disp_to_depth,
    Project3D,
)
from torchdrive.transforms.img import normalize_img, render_color
from torchdrive.transforms.mat import voxel_to_world


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
        cam_feats_shape: Tuple[int, int],
        dim: int,
        hr_dim: int,
        cam_dim: int,
        height: int,
        device: torch.device,
        scale: int = 3,
        z_offset: float = 0.5,
        semantic: Optional[List[str]] = None,
        render_batch_size: int = 5,
        n_pts_per_ray: int = 216,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda x: x,
        offsets: Tuple[int, ...] = (-2, -1, 1, 2),
        camera_overlap: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Args:
            camera_overlap:
                Dictionary of each camera and which cameras it overlaps with. If
                specified enables stereoscopic losses.
        """
        super().__init__()

        self.cam_shape = cam_shape
        self.cameras = cameras
        self.scale = scale
        self.height = height
        self.semantic = semantic
        self.render_batch_size = render_batch_size
        self.offsets = offsets
        self.camera_overlap = camera_overlap

        # TODO support non-centered voxel grids
        self.volume_translation: Tuple[float, float, float] = (
            0,
            0,
            -self.height * z_offset / self.scale,
        )

        # generate voxel grid
        self.num_elem: int = 1
        background: List[float] = []
        if semantic:
            self.classes_elem: int = len(BDD100KSemSeg.NON_SKY)
            background += [-100.0] * self.classes_elem
            self.vel_elem: int = 3
            background += [0.0, 0.0, 0.0]
            self.num_elem += self.classes_elem + self.vel_elem
            self.segment: BDD100KSemSeg = BDD100KSemSeg(
                device=device, compile_fn=compile_fn
            )

        h, w = cam_shape

        self.decoder: nn.Module = compile_fn(
            nn.Conv2d(hr_dim, self.num_elem * height, kernel_size=1)
        )
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
            floor=None,
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
                cam_shape=cam_feats_shape,
                dim=cam_dim,
            )
        )
        # pyre-fixme[6]: nn.Module
        self.projection_loss: nn.Module = compile_fn(multi_scale_projection_loss)

    def forward(
        self, ctx: Context, batch: Batch, bev: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        BS = len(batch.distances)
        frames = batch.distances.shape[1]
        start_frame = ctx.start_frame
        device = bev.device

        bev_shape = bev.shape[2:]

        with autocast():
            embedding = self.decoder(bev).unflatten(1, (self.num_elem, self.height))
        # convert back to float so sigmoid works
        embedding = embedding.float()

        grid = embedding[:, :1].sigmoid()
        feat_grid = embedding[:, 1:]

        grid = grid.permute(0, 1, 4, 3, 2)
        feat_grid = feat_grid.permute(0, 1, 4, 3, 2)
        # grid, _ = axis_grid(grid)

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

            vtw = voxel_to_world(
                center=(-bev_shape[0] // 2, -bev_shape[1] // 2, 0),
                scale=self.scale,
                device=device,
            )

            zero_coord = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float)
            for frame in range(0, frames):
                # create car to voxel transform
                T = batch.world_to_car(frame)
                T = T.matmul(vtw)
                T = T.pinverse()

                cam_coords = T.matmul(zero_coord.T).squeeze(-1)
                cam_coords /= cam_coords[:, 3:].clamp(min=1e-8)
                coord = cam_coords[0, :3]
                x, y, z = coord.int()
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
            losses["tvl1"] = tvl1_loss(grid.squeeze(1))

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
        frame_time = batch.frame_time - batch.frame_time[:, ctx.start_frame].unsqueeze(
            1
        )
        device = grid.device

        losses = {}

        h, w = self.cam_shape
        frame = ctx.start_frame

        primary_colors = {}
        primary_masks = {}
        for cam in self.cameras:
            primary_color = batch.color[cam][:, frame]
            primary_color = F.interpolate(
                primary_color.float(),
                [h // 2, w // 2],
                mode="bilinear",
                align_corners=False,
            )
            primary_colors[cam] = primary_color
            primary_mask = batch.mask[cam]
            primary_mask = F.interpolate(
                primary_mask.float(),
                [h // 2, w // 2],
                mode="bilinear",
                align_corners=False,
            )
            primary_masks[cam] = primary_mask

        semantic_targets = {}
        dynamic_masks = {}
        if self.semantic:
            with torch.autograd.profiler.record_function("segment"):
                for cam in self.cameras:
                    semantic_target = self.segment(primary_colors[cam]).sigmoid()
                    semantic_targets[cam] = semantic_target

                    dynamic_mask = semantic_target[:, BDD100KSemSeg.DYNAMIC]
                    dynamic_mask = dynamic_mask.amax(dim=1).round()
                    dynamic_masks[cam] = dynamic_mask

        for cam in self.cameras:
            volumes = Volumes(
                densities=grid.permute(0, 1, 4, 3, 2),
                features=feat_grid.permute(0, 1, 4, 3, 2).float()
                if feat_grid is not None
                else None,
                voxel_size=1 / self.scale,
                volume_translation=self.volume_translation,
            )

            K = batch.K[cam]
            # create camera to world transformation matrix
            T = batch.cam_to_world(cam, start_frame)
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
            semantic_img = semantic_img.permute(0, 3, 1, 2)

            to_pause = [voxel_depth]
            if self.semantic:
                to_pause.append(semantic_img)

            # pause backwards pass
            with autograd_context(*to_pause) as paused:
                if self.semantic:
                    voxel_depth, semantic_img = paused
                else:
                    voxel_depth = paused

                primary_color = primary_colors[cam]
                primary_mask = primary_masks[cam]
                per_pixel_weights = torch.ones_like(primary_color).mean(
                    dim=1, keepdim=True
                )

                if self.semantic:
                    dynamic_mask = dynamic_masks[cam]
                    semantic_loss = self._semantic_loss(
                        ctx=ctx,
                        cam=cam,
                        semantic_img=semantic_img,
                        semantic_target=semantic_targets[cam],
                        color=primary_color,
                        mask=primary_mask,
                    )

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
                    dynamic_mask = dynamic_mask.unsqueeze(1)
                    semantic_vel *= dynamic_mask
                    semantic_vel *= primary_mask

                    # focus more on dynamic objects
                    per_pixel_weights += dynamic_mask
                    # normalize mean
                    per_pixel_weights /= per_pixel_weights.mean()

                    semantic_loss = semantic_loss * F.avg_pool2d(per_pixel_weights, 2)
                    if ctx.log_img:
                        ctx.add_image(
                            f"semantic/{cam}/loss",
                            render_color(semantic_loss[0].mean(dim=0)),
                        )
                    losses[f"semantic/{cam}"] = semantic_loss.mean(dim=(1, 2, 3)) * 100
                else:
                    dynamic_mask = torch.zeros_like(primary_color)
                    semantic_vel = torch.zeros_like(primary_color)

                voxel_disp = depth_to_disp(
                    voxel_depth,
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                )

                self._sfm_loss(
                    ctx=ctx,
                    label="voxel",
                    cam=cam,
                    disp=voxel_disp,
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
                    per_pixel_weights=per_pixel_weights,  # * 1e-1,
                )

                camera_overlap = self.camera_overlap
                if camera_overlap:
                    self._stereoscopic_loss(
                        ctx=ctx,
                        batch=batch,
                        primary_cam=cam,
                        overlap_cams=camera_overlap[cam],
                        cam_features=semantic_targets,
                        cam_masks=primary_masks,
                        per_pixel_weights=per_pixel_weights,
                        primary_depth=voxel_depth,
                        losses=losses,
                        h=h,
                        w=w,
                    )
            del voxel_depth
            del semantic_vel
            del semantic_img

            if cam in ctx.cam_feats:
                with torch.autograd.profiler.record_function(
                    "depth_decoder"
                ), autocast():
                    cam_disp, cam_vel = self.depth_decoder(ctx.cam_feats[cam])

                cam_vel = F.interpolate(
                    cam_vel.float(),
                    [h // 2, w // 2],
                    mode="bilinear",
                    align_corners=False,
                )
                cam_vel *= dynamic_mask
                cam_vel *= primary_mask
                cam_disp = cam_disp.float().sigmoid()
                cam_depth = disp_to_depth(
                    cam_disp,
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                )

                self._sfm_loss(
                    ctx=ctx,
                    label="cam",
                    cam=cam,
                    disp=cam_disp,
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
                    per_pixel_weights=per_pixel_weights * 0.5,
                )

                del cam_vel
                del cam_depth
                del cam_disp

        return losses

    def _sfm_loss(
        self,
        ctx: Context,
        label: str,
        cam: str,
        disp: torch.Tensor,
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
        """
        Computes the structure from motion loss for a single camera across
        multiple times.
        """
        depth = F.interpolate(
            depth.float().unsqueeze(1),
            [h // 2, w // 2],
            mode="bilinear",
            align_corners=False,
        )
        disp = F.interpolate(
            disp.float().unsqueeze(1),
            [h // 2, w // 2],
            mode="bilinear",
            align_corners=False,
        )
        # upscale to original size
        if ctx.log_text:
            amin, amax = depth.aminmax()
            assert amin >= 0, (amin, amax)
            ctx.add_scalars(
                f"depth{label}/{cam}/minmax",
                {"max": amax, "min": amin},
            )

        for typ, t in [("disp", disp), ("vel", semantic_vel)]:
            losssmooth = smooth_loss(t, primary_color) * per_pixel_weights
            losses[f"losssmooth-{label}-{typ}/{cam}"] = (
                losssmooth.mean(dim=(1, 2, 3)) * 20
            )
            if ctx.log_img:
                ctx.add_image(
                    f"losssmooth-{label}-{typ}/{cam}-pixel",
                    render_color(losssmooth[0].mean(dim=0)),
                )

        if ctx.log_img:
            ctx.add_image(
                f"depth{label}/{cam}",
                render_color(-depth[0][0]),
            )
            out_disp = disp[0, 0] * primary_mask[0, 0]
            ctx.add_image(
                f"disp{label}/{cam}",
                render_color(out_disp),
            )

        for offset in self.offsets:
            target_frame = frame + offset
            assert target_frame >= 0, (frame, offset)
            backproject_T = batch.cam_to_world(cam, frame)
            project_T = batch.world_to_cam(cam, target_frame)
            time = frame_time[:, target_frame]

            projcolor, projmask = self.project(
                batch=batch,
                primary_cam=cam,
                target_cam=cam,
                backproject_T=backproject_T,
                project_T=project_T,
                depth=depth,
                color=primary_color,
                mask=primary_mask,
                vel=semantic_vel * time.reshape(-1, 1, 1, 1),
            )
            projmask *= primary_mask
            color = batch.color[cam][:, target_frame]
            half_color = F.interpolate(
                color.float(),
                [h // 2, w // 2],
                mode="bilinear",
                align_corners=False,
            )
            color = half_color
            proj_weights = per_pixel_weights

            proj_loss = self.projection_loss(projcolor, color, scales=3, mask=projmask)
            # identity_proj_loss = self.projection_loss(
            #    color, primary_color, scales=3, mask=projmask
            # )
            # identity_proj_loss += torch.full_like(identity_proj_loss, 0.00001)

            # automask
            min_proj_loss = proj_loss
            # min_proj_loss = torch.minimum(proj_loss, identity_proj_loss)

            min_proj_loss = min_proj_loss * proj_weights
            losses[f"lossproj-{label}/{cam}/o{offset}"] = (
                min_proj_loss.mean(dim=(1, 2, 3)) * 40
            )

            if ctx.log_img:
                ctx.add_image(
                    f"{label}/{cam}/{offset}/color",
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
                    f"{label}/{cam}/{offset}/min_proj_loss",
                    render_color(min_proj_loss[0, 0]),
                )
                # ctx.add_image(
                #    f"{label}/{cam}/{offset}/automask",
                #    render_color(proj_loss[0, 0] < identity_proj_loss[0, 0]),
                # )

        ctx.backward(losses)

    def _stereoscopic_loss(
        self,
        ctx: Context,
        batch: Batch,
        primary_cam: str,
        cam_features: Dict[str, torch.Tensor],
        cam_masks: Dict[str, torch.Tensor],
        overlap_cams: List[str],
        per_pixel_weights: torch.Tensor,
        primary_depth: torch.Tensor,
        losses: Dict[str, torch.Tensor],
        h: int,
        w: int,
    ) -> None:
        """
        Computes the stereoscopic projection loss between the target cam and the
        overlapping cameras.
        """
        frame = ctx.start_frame
        primary_mask = cam_masks[primary_cam]
        primary_features = cam_features[primary_cam]

        primary_depth = F.interpolate(
            primary_depth.float().unsqueeze(1),
            [h // 2, w // 2],
            mode="bilinear",
            align_corners=False,
        )

        for target_cam in overlap_cams:
            backproject_T = batch.cam_to_world(primary_cam, frame)
            project_T = batch.world_to_cam(target_cam, frame)
            target_mask = cam_masks[target_cam]
            target_features = cam_features[target_cam]

            print(primary_features.shape, primary_depth.shape)

            proj_features, proj_mask = self.project(
                batch=batch,
                primary_cam=primary_cam,
                target_cam=target_cam,
                backproject_T=backproject_T,
                project_T=project_T,
                depth=primary_depth,
                color=primary_features,
                mask=primary_mask,
            )
            proj_mask *= target_mask

            proj_loss = self.projection_loss(
                proj_features, target_features, scales=3, mask=proj_mask
            )
            proj_loss = proj_loss * per_pixel_weights
            losses[f"lossstereoscopic/{primary_cam}/{target_cam}"] = (
                proj_loss.mean(dim=(1, 2, 3)) * 40
            )

            if ctx.log_img:
                ctx.add_image(
                    f"stereoscopic/{primary_cam}/{target_cam}/feats",
                    normalize_img(
                        torch.cat(
                            (
                                target_features[0],
                                proj_features[0],
                            ),
                            dim=2,
                        )
                    ),
                )
                ctx.add_image(
                    f"stereoscopic/{primary_cam}/{target_cam}/proj_loss",
                    render_color(proj_loss[0, 0]),
                )

        ctx.backward(losses)

    def _semantic_loss(
        self,
        ctx: Context,
        cam: str,
        semantic_img: torch.Tensor,
        semantic_target: torch.Tensor,
        color: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        _semantic_loss computes the semantic class probability loss.
        """
        semantic_classes = semantic_img[:, : self.classes_elem].sigmoid()

        # select interesting classes and convert to probabilities
        semantic_target = semantic_target[:, BDD100KSemSeg.NON_SKY]
        semantic_target = F.avg_pool2d(semantic_target, 2)

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

        return sem_loss

    def project(
        self,
        batch: Batch,
        primary_cam: str,
        target_cam: str,
        backproject_T: torch.Tensor,
        project_T: torch.Tensor,
        depth: torch.Tensor,
        color: torch.Tensor,
        mask: torch.Tensor,
        vel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_K = batch.K[target_cam].clone()
        # convert to image space
        src_K[:, 0] *= self.backproject_depth.width
        src_K[:, 1] *= self.backproject_depth.height

        target_K = batch.K[primary_cam].clone()
        # convert to image space
        target_K[:, 0] *= self.backproject_depth.width
        target_K[:, 1] *= self.backproject_depth.height
        target_inv_K = target_K.pinverse()

        world_points = self.backproject_depth(
            depth, target_inv_K, backproject_T
        ).clone()

        if vel is not None:
            # add velocity to points
            world_points[:, :3] += vel.flatten(-2, -1)

        # (world to cam) * camera motion
        pix_coords = self.project_3d(world_points, src_K, project_T)

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
