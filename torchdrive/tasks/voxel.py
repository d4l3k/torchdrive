import os.path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F
import torchmetrics
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
from torchdrive.transforms.img import normalize_img, normalize_mask, render_color
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
            self.semantic_confusion_matrix = torchmetrics.ConfusionMatrix(
                task="multiclass",
                num_classes=self.classes_elem,
                normalize="true",
            )
            self.semantic_accuracy = torchmetrics.Accuracy(
                task="multiclass",
                num_classes=self.classes_elem,
                average="macro",
            )
        else:
            self.classes_elem = 0
        self.register_buffer(
            "semantic_bin_counts",
            torch.zeros(len(BDD100KSemSeg.LABELS), dtype=torch.int64),
            persistent=False,
        )
        # compute inverse frequency to weight the pixel labels according to rarity
        semantic_weights = 1 / torch.tensor(
            BDD100KSemSeg.CLASS_FREQUENCY, dtype=torch.float
        )
        semantic_weights /= semantic_weights.mean()
        self.register_buffer(
            "semantic_weights",
            semantic_weights,
            persistent=False,
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
                num_upsamples=1,
                cam_shape=cam_feats_shape,
                dim=cam_dim,
                num_classes=self.classes_elem,
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

        # log grad norms on full embedding to make norms comparable
        grid = ctx.log_grad_norm(embedding, "grad/norm/grid_embedding", "grid")[
            :, :1
        ].sigmoid()
        feat_grid = ctx.log_grad_norm(
            embedding, "grad/norm/grid_embedding", "feat_grid"
        )[:, 1:]

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
            tvl1_grid = ctx.log_grad_norm(grid, "grad/norm/grid", "tvl1")
            losses["tvl1"] = tvl1_loss(tvl1_grid.squeeze(1)) * 0.25

            mini_batches = batch.split(self.render_batch_size)
            mini_grids = ctx.log_grad_norm(grid, "grad/norm/grid", "mini_grids")
            mini_grids = torch.split(mini_grids, self.render_batch_size)
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

        primary_colors: Dict[str, torch.Tensor] = {}
        primary_masks: Dict[str, torch.Tensor] = {}
        cam_pix_weights: Dict[str, torch.Tensor] = {}
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

            cam_pix_weights[cam] = torch.ones_like(primary_color).mean(
                dim=1, keepdim=True
            )

        semantic_targets: Dict[str, torch.Tensor] = {}
        dynamic_masks: Dict[str, torch.Tensor] = {}
        if self.semantic:
            with torch.autograd.profiler.record_function("segment"):
                for cam in self.cameras:
                    semantic_target = self.segment(primary_colors[cam]).sigmoid()
                    semantic_targets[cam] = semantic_target

                    semantic_classes = semantic_target.argmax(dim=1, keepdim=True)
                    class_bincount = torch.bincount(
                        semantic_classes.flatten(), minlength=semantic_target.size(1)
                    )
                    self.semantic_bin_counts += class_bincount

                    dynamic_mask = semantic_target[:, BDD100KSemSeg.DYNAMIC]
                    dynamic_mask = dynamic_mask.amax(dim=1).round().unsqueeze(1)
                    dynamic_masks[cam] = dynamic_mask

                    # Compute per pixel weights based on the inverse frequency
                    # of the classes to counter weight imbalance and normalize
                    # mean to 1.
                    per_pixel_weights = torch.index_select(
                        self.semantic_weights, dim=0, index=semantic_classes.flatten()
                    ).view_as(semantic_classes)
                    per_pixel_weights /= per_pixel_weights.mean()
                    cam_pix_weights[cam] = per_pixel_weights

                    if ctx.log_img:
                        ctx.add_image(
                            f"semantic/per_pixel_weights/{cam}",
                            render_color(per_pixel_weights[0, 0]),
                        )

            if ctx.log_img:
                # log distribution of semantic classes
                bin_frequency = (
                    self.semantic_bin_counts / self.semantic_bin_counts.sum()
                )
                # print("bin_frequency", bin_frequency)
                self.semantic_bin_counts.zero_()
                ctx.add_scalars(
                    "semantic/label_distribution",
                    {
                        label: bin_frequency[i]
                        for i, label in BDD100KSemSeg.LABELS.items()
                    },
                )

        else:
            for cam in self.cameras:
                dynamic_masks[cam] = torch.zeros(BS, 1, h // 2, w // 2, device=device)

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
                per_pixel_weights = cam_pix_weights[cam]

                dynamic_mask = dynamic_masks[cam]
                if self.semantic:
                    semantic_vel = ctx.log_grad_norm(
                        semantic_img, "grad/semantic_img", "semantic_vel"
                    )[:, self.classes_elem :]
                    semantic_vel = F.interpolate(
                        semantic_vel.float(),
                        [h // 2, w // 2],
                        mode="bilinear",
                        align_corners=False,
                    )
                    semantic_classes = ctx.log_grad_norm(
                        semantic_img, "grad/semantic_img", "semantic_classes"
                    )[:, : self.classes_elem].sigmoid()
                    semantic_classes = F.interpolate(
                        semantic_classes.float(),
                        [h // 2, w // 2],
                        mode="bilinear",
                        align_corners=False,
                    )

                    if ctx.log_text:
                        ctx.add_scalars(
                            f"semantic_vel/{cam}/abs",
                            {
                                "max": semantic_vel.abs().amax(),
                                "mean": semantic_vel.abs().mean(),
                            },
                        )
                    if ctx.log_img:
                        ctx.add_image(
                            f"semantic_vel/{cam}",
                            normalize_img(semantic_vel[0]),
                        )
                        ctx.add_image(
                            f"{cam}/dynamic_mask",
                            render_color(dynamic_mask[0, 0]),
                        )
                    semantic_vel *= dynamic_mask
                    semantic_vel *= primary_mask

                    semantic_loss = self._semantic_loss(
                        ctx=ctx,
                        label="voxel",
                        cam=cam,
                        semantic_classes=semantic_classes,
                        semantic_target=semantic_targets[cam],
                        mask=primary_mask,
                        per_pixel_weights=per_pixel_weights,
                    )
                    losses[f"semantic-voxel/{cam}"] = semantic_loss.mean(dim=(1, 2, 3))

                    # update and plot confusion matrix
                    preds = semantic_classes.argmax(dim=1).flatten()
                    target = (
                        semantic_targets[cam][:, BDD100KSemSeg.NON_SKY]
                        .argmax(dim=1)
                        .flatten()
                    )
                    self.semantic_confusion_matrix.update(
                        preds=preds,
                        target=target,
                    )
                    self.semantic_accuracy.update(
                        preds=preds,
                        target=target,
                    )
                else:
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
                        label="voxel",
                        primary_cam=cam,
                        overlap_cams=camera_overlap[cam],
                        # cam_features=semantic_targets,
                        cam_features=primary_colors,
                        cam_masks=primary_masks,
                        cam_pix_weights=cam_pix_weights,
                        primary_depth=voxel_depth,
                        losses=losses,
                        h=h,
                        w=w,
                    )

            if self.semantic:
                if ctx.log_img:
                    fig, ax = self.semantic_confusion_matrix.plot(
                        labels=[BDD100KSemSeg.LABELS[i] for i in BDD100KSemSeg.NON_SKY],
                        add_text=False,
                    )
                    ctx.add_figure("semantic/confusion_matrix", fig)
                    self.semantic_confusion_matrix.reset()
                if ctx.log_text:
                    ctx.add_scalar(
                        "semantic/accuracy", self.semantic_accuracy.compute()
                    )
                    self.semantic_accuracy.reset()

            del voxel_depth
            del semantic_vel
            del semantic_img

            if cam in ctx.cam_feats:
                with torch.autograd.profiler.record_function(
                    "depth_decoder"
                ), autocast():
                    cam_disp, cam_vel, cam_sem = self.depth_decoder(ctx.cam_feats[cam])

                cam_vel = F.interpolate(
                    cam_vel.float(),
                    [h // 2, w // 2],
                    mode="bilinear",
                    align_corners=False,
                )
                if self.semantic:
                    cam_sem = F.interpolate(
                        cam_sem.float(),
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

                with autograd_context(cam_disp, cam_depth, cam_vel, cam_sem) as (
                    cam_disp,
                    cam_depth,
                    cam_vel,
                    cam_sem,
                ):
                    if self.semantic:
                        semantic_loss = self._semantic_loss(
                            ctx=ctx,
                            label="cam",
                            cam=cam,
                            semantic_classes=cam_sem.float().sigmoid(),
                            semantic_target=semantic_targets[cam],
                            mask=primary_mask,
                            per_pixel_weights=per_pixel_weights * 0.1,
                        )
                        losses[f"semantic-cam/{cam}"] = semantic_loss.mean(
                            dim=(1, 2, 3)
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
                        per_pixel_weights=per_pixel_weights * 0.1,
                    )

                    camera_overlap = self.camera_overlap
                    if camera_overlap:
                        self._stereoscopic_loss(
                            ctx=ctx,
                            batch=batch,
                            label="cam",
                            primary_cam=cam,
                            overlap_cams=camera_overlap[cam],
                            # cam_features=semantic_targets,
                            cam_features=primary_colors,
                            cam_masks=primary_masks,
                            cam_pix_weights=cam_pix_weights,
                            loss_scale=0.1,
                            primary_depth=cam_depth,
                            losses=losses,
                            h=h,
                            w=w,
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

        for typ, t in [("disp", disp)]:  # , ("vel", semantic_vel)]:
            if not t.requires_grad:
                continue
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
            src_frame = frame + offset
            assert src_frame >= 0, (frame, offset)
            target_cam_to_world = batch.cam_to_world(cam, frame)
            world_to_src_cam = batch.world_to_cam(cam, src_frame)
            time = frame_time[:, src_frame]

            if ctx.log_text and offset != 0:
                time_max = time.abs().amax()
                assert (
                    time_max > 0 and time_max < 60
                ), f"frame_time is bad {offset} {time}"
                ctx.add_scalar(f"frame_time_max/{offset}", time_max)

            src_color = batch.color[cam][:, src_frame]
            src_color = F.interpolate(
                src_color.float(),
                [h // 2, w // 2],
                mode="bilinear",
                align_corners=False,
            )

            # project from offset frame (src) to start frame (target)
            projcolor, projmask = self._project(
                batch=batch,
                src_cam=cam,
                target_cam=cam,
                target_cam_to_world=target_cam_to_world,
                world_to_src_cam=world_to_src_cam,
                target_depth=depth,
                src_color=src_color,
                src_mask=primary_mask,
                target_vel=semantic_vel * time.reshape(-1, 1, 1, 1),
            )
            projmask *= primary_mask
            proj_weights = per_pixel_weights

            proj_loss = self.projection_loss(
                projcolor, primary_color, scales=3, mask=projmask
            )
            # identity_proj_loss = self.projection_loss(
            #    color, primary_color, scales=3, mask=projmask
            # )
            # identity_proj_loss += torch.full_like(identity_proj_loss, 0.00001)

            # automask
            min_proj_loss = proj_loss
            # min_proj_loss = torch.minimum(proj_loss, identity_proj_loss)

            min_proj_loss = min_proj_loss * proj_weights
            losses[f"lossproj-{label}/{cam}/o{offset}"] = (
                min_proj_loss.mean(dim=(1, 2, 3)) * 40 * 6
            )

            if ctx.log_img:
                ctx.add_image(
                    f"{label}/{cam}/{offset}/color",
                    normalize_img(
                        torch.cat(
                            (
                                src_color[0],
                                projcolor[0],
                                primary_color[0],
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
        label: str,
        primary_cam: str,
        cam_features: Dict[str, torch.Tensor],
        cam_masks: Dict[str, torch.Tensor],
        overlap_cams: List[str],
        cam_pix_weights: Dict[str, torch.Tensor],
        primary_depth: torch.Tensor,
        losses: Dict[str, torch.Tensor],
        h: int,
        w: int,
        loss_scale: float = 1,
    ) -> None:
        """
        Computes the stereoscopic projection loss between the target cam and the
        overlapping cameras.
        """
        frame = ctx.start_frame
        primary_mask = cam_masks[primary_cam]
        primary_features = cam_features[primary_cam].float()

        primary_depth = F.interpolate(
            primary_depth.float().unsqueeze(1),
            [h // 2, w // 2],
            mode="bilinear",
            align_corners=False,
        )

        for src_cam in overlap_cams:
            target_cam_to_world = batch.cam_to_world(primary_cam, frame)
            world_to_src_cam = batch.world_to_cam(src_cam, frame)
            src_mask = cam_masks[src_cam]
            src_features = cam_features[src_cam].float()

            proj_features, proj_mask = self._project(
                batch=batch,
                src_cam=src_cam,
                target_cam=primary_cam,
                target_cam_to_world=target_cam_to_world,
                world_to_src_cam=world_to_src_cam,
                target_depth=primary_depth,
                src_color=src_features,
                src_mask=src_mask,
            )
            proj_mask *= primary_mask

            proj_features = normalize_mask(proj_features, proj_mask)
            target_features = normalize_mask(primary_features, proj_mask)

            proj_loss = self.projection_loss(
                proj_features, target_features, scales=3, mask=proj_mask
            )
            # proj_loss = proj_loss * cam_pix_weights[target_cam]
            losses[f"lossstereoscopic-{label}/{primary_cam}/{src_cam}"] = (
                proj_loss.mean(dim=(1, 2, 3)) * 40 * loss_scale
            )

            if ctx.log_img:
                ctx.add_image(
                    f"stereoscopic-{label}/{primary_cam}/{src_cam}/feats",
                    normalize_img(
                        torch.cat(
                            (
                                target_features[0] * primary_mask[0],
                                proj_features[0] * proj_mask[0],
                            ),
                            dim=2,
                        ),
                    ),
                )
                ctx.add_image(
                    f"stereoscopic-{label}/{primary_cam}/{src_cam}/proj_loss",
                    render_color(proj_loss[0, 0]),
                )

        ctx.backward(losses)

    def _semantic_loss(
        self,
        ctx: Context,
        label: str,
        cam: str,
        semantic_classes: torch.Tensor,
        semantic_target: torch.Tensor,
        mask: torch.Tensor,
        per_pixel_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        _semantic_loss computes the semantic class probability loss.
        """
        # select interesting classes and convert to probabilities
        semantic_target = semantic_target[:, BDD100KSemSeg.NON_SKY]

        assert semantic_classes.shape == semantic_target.shape, (
            semantic_classes.shape,
            semantic_target.shape,
        )

        sem_loss = self.projection_loss(
            semantic_classes.float(),
            semantic_target.float(),
            scales=3,
            mask=mask,
        )
        sem_loss = sem_loss * per_pixel_weights * 100 * 1000 / 16

        if ctx.log_text:
            pred_min, pred_max = semantic_classes.aminmax()
            targ_min, targ_max = semantic_target.aminmax()
            ctx.add_scalars(
                f"semantic-{label}/{cam}/minmax",
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
            out_class = torch.argmax(semantic_classes[:1], dim=1)
            target_class = torch.argmax(semantic_target[:1], dim=1)
            ctx.add_image(
                f"semantic-{label}/{cam}/output_target",
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
                f"semantic-{label}/{cam}/loss",
                render_color(sem_loss[0].mean(dim=0)),
            )

        return sem_loss

    def _project(
        self,
        batch: Batch,
        src_cam: str,
        target_cam: str,
        target_cam_to_world: torch.Tensor,
        world_to_src_cam: torch.Tensor,
        target_depth: torch.Tensor,
        src_color: torch.Tensor,
        src_mask: torch.Tensor,
        target_vel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_K = batch.K[src_cam].clone()
        # convert to image space
        src_K[:, 0] *= self.backproject_depth.width
        src_K[:, 1] *= self.backproject_depth.height

        target_K = batch.K[target_cam].clone()
        # convert to image space
        target_K[:, 0] *= self.backproject_depth.width
        target_K[:, 1] *= self.backproject_depth.height
        target_inv_K = target_K.pinverse()

        world_points = self.backproject_depth(
            target_depth, target_inv_K, target_cam_to_world
        ).clone()

        if target_vel is not None:
            # add velocity to points
            world_points[:, :3] += target_vel.flatten(-2, -1)

        # (world to cam) * camera motion
        pix_coords = self.project_3d(world_points, src_K, world_to_src_cam)

        color = F.grid_sample(
            src_color,
            pix_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        mask = F.grid_sample(
            src_mask,
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
