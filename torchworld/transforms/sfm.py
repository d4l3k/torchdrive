from typing import Optional

import torch
import torch.nn.functional as F

from torchworld.structures.grid import GridImage


def project(
    dst: GridImage,
    src: GridImage,
    depth: GridImage,
    vel: Optional[GridImage] = None,
    eps: float = 1e-8,
) -> GridImage:
    """
    project projects the dst target into src using depth and velocities
    corresponding to src.
    """
    if dst.data.shape[1:] != src.data.shape[1:]:
        raise TypeError("dst and src must have same non-batch shape")
    if depth.data.shape[2:] != src.data.shape[2:]:
        raise TypeError("src and depth must have same height/width dimensions")
    if depth.data.size(1) != 1:
        raise TypeError("depth must have have 1 channel")
    if vel is not None:
        if vel.data.shape[2:] != src.data.shape[2:]:
            raise TypeError("src and depth must have same height/width dimensions")
        if vel.data.size(1) != 3:
            raise TypeError("vel must have 3 channels")
        if src.camera is not vel.camera:
            raise TypeError("src camera must be same as vel")
    if src.camera is not depth.camera:
        raise TypeError("src camera must be same as depth")

    device = src.data.device
    BS = len(src.data)

    channels = torch.meshgrid(
        *(
            torch.arange(-1, 1 - eps, 2 / dim, device=device)
            for dim in src.grid_shape()
        ),
        indexing="ij",
    )
    src_points = torch.stack(channels, dim=-1)
    src_points = src_points.expand(BS, -1, -1, -1)
    src_points = torch.cat((src_points, depth.data.permute(0, 2, 3, 1)), dim=-1)
    src_points = src_points.flatten(1, 2)  # [bs, y*x, 3]

    world_points = src.camera.unproject_points(src_points)  # [bs, y*x, 3]
    world_points = world_points.unflatten(1, src.grid_shape())  # [bs, y, x, 3]

    if vel is not None:
        delta = dst.time - src.time
        diff = vel.data.permute(0, 2, 3, 1) * delta.reshape(-1, 1, 1, 1)
        world_points += diff

    world_points = world_points.flatten(1, 2)  # [bs, y*x, 3]
    dst_points = dst.camera.transform_points(world_points, eps=eps)
    dst_points = dst_points.unflatten(1, src.grid_shape())  # [bs, y, x, 3]
    dst_points = dst_points[..., :2]  # [bs, y, x, 2]

    color = F.grid_sample(
        dst.data,
        dst_points,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    if dst.mask is not None:
        mask = F.grid_sample(
            dst.mask,
            dst_points,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
    else:
        mask = None

    return src.replace(
        data=color,
        mask=mask,
    )
