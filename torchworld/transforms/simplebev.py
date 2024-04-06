from typing import Tuple

import torch
import torch.nn.functional as F

from torchworld.structures.grid import Grid3d, GridImage


def lift_image_to_3d(
    src: GridImage,
    dst: Grid3d,
    eps: float = 1e-7,
) -> Tuple[Grid3d, Grid3d]:
    """
    Lift the features from a camera to a Voxel volume.

    Implements Simple BEV lifting operation.

    See: https://arxiv.org/pdf/2206.07959.pdf

    Arguments
    ---------
    src: Source features and camera.
    dst: Destination 3D grid.
    eps: A small value to avoid NaNs.

    Returns
    -------
    features: grid with features
    mask: grid of the mask where the camera could see
    """
    if dst.numel() != 0:
        raise TypeError(f"dst should be batch size zero {dst.shape}")

    device = src.device
    BS = len(src)
    grid_shape = dst.grid_shape()

    # calculate the x/y/z coordinates for each voxel in the grid
    channels = torch.meshgrid(
        *(torch.arange(-1, 1 - eps, 2 / dim, device=device) for dim in grid_shape),
        indexing="ij",
    )
    grid_points = torch.stack(channels[::-1], dim=-1)
    grid_points = grid_points.flatten(0, -2).unsqueeze(0)

    world_to_ndc_transform = src.camera.world_to_ndc_transform()
    local_to_ndc_transform = dst.local_to_world.compose(world_to_ndc_transform)

    # grid_points = grid_points[:, :, ::-1]

    image_points = local_to_ndc_transform.transform_points(grid_points, eps=eps)

    with torch.no_grad():
        # hide samples behind camera
        x = image_points[..., 0]
        y = image_points[..., 1]
        z = image_points[..., 2]

        valid_z = z > 0
        valid_x = torch.logical_and(torch.greater(x, -1), torch.less(x, 1))
        valid_y = torch.logical_and(torch.greater(y, -1), torch.less(y, 1))
        valid = torch.logical_and(torch.logical_and(valid_x, valid_y), valid_z)
        valid = valid.unflatten(1, grid_shape).unsqueeze(1)

    # drop z axis
    image_points = image_points[..., :2]
    # grid_sample needs a 2d input so we add a dummy dimension
    image_points = image_points.unsqueeze(1)

    # make batch size match
    if len(image_points) == 1:
        image_points = image_points.expand(BS, -1, -1, -1)
        valid = valid.expand(BS, -1, -1, -1, -1)

    # grid_sample doesn't support bfloat16 so cast to float
    values = F.grid_sample(src.float(), image_points, align_corners=False)
    values = values.to(src.dtype)

    # restore to grid shape
    values = values.squeeze(2).unflatten(-1, grid_shape)
    values *= valid

    # convert to dst type
    return (
        Grid3d(
            data=values._data,
            local_to_world=dst.local_to_world,
            time=dst.time,
        ),
        Grid3d(
            data=valid,
            local_to_world=dst.local_to_world,
            time=dst.time,
        ),
    )


def merge_grids(
    grids: Tuple[Grid3d, ...],
    masks: Tuple[Grid3d, ...],
) -> Tuple[Grid3d, Grid3d]:
    """
    Merge multiple grids into a single grid.

    Arguments
    ---------
    grids: Tuple of grids to merge.
    masks: Tuple of masks to merge.

    Returns
    -------
    merged_grid: Merged grid
    merged_mask: Merged mask
    """

    merged_grid = torch.stack(grids).sum(dim=0)
    merged_mask = torch.stack(masks).sum(dim=0)
    clamped_mask = merged_mask.clamp(min=0.0, max=1.0)
    scale = (1.0 / merged_mask.clamp(min=1.0)) * clamped_mask
    merged_grid *= scale
    return merged_grid, clamped_mask
