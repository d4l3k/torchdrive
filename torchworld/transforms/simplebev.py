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
    if dst.data.numel() != 0:
        raise TypeError(f"dst should be batch size zero {dst.shape}")

    device = src.device
    BS = len(src)
    grid_shape = dst.grid_shape()

    # calculate the x/y/z coordinates for each voxel in the grid
    channels = torch.meshgrid(
        *(torch.arange(-1, 1 - eps, 2 / dim, device=device) for dim in grid_shape),
        indexing="ij",
    )
    grid_points = torch.stack(channels, dim=-1)
    grid_points = grid_points.flatten(0, -2).unsqueeze(0)
    T = dst.local_to_world
    T = T.compose(src.camera.get_full_projection_transform())
    assert src.camera.in_ndc(), "TODO support non-ndc cameras"

    image_points = T.transform_points(grid_points, eps=eps)

    # hide samples behind camera
    z = image_points[..., 2]

    valid = z > 0
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
    values = F.grid_sample(src.data.float(), image_points, align_corners=False)
    values = values.to(src.data.dtype)

    # restore to grid shape
    values = values.squeeze(2).unflatten(-1, grid_shape)
    values *= valid

    return (
        dst.replace(data=values, time=src.time),
        dst.replace(data=valid, time=src.time),
    )
