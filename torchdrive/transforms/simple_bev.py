from typing import Tuple

import torch
import torch.nn.functional as F


def lift_cam_to_voxel(
    features: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Lift the features from a camera to a Voxel volume.

    Implements Simple BEV lifting operation.

    See: https://arxiv.org/pdf/2206.07959.pdf

    Arguments:
        features: [batch_size, channels, height, width]
            camera space features
        K: [batch_size, 4, 4]
            camera intrinsics
        T: [batch_size, 4, 4]
            camera extrinsics (world to camera)
            the grid starts at 0, 0 so will need to adjust this to position the
            camera in the grid
        grid_shape:
            Tuple size of the grid. Size should be backed into T
        eps:
            A small value to avoid NaNs.
    """
    device = features.device
    dtype = features.dtype
    BS = features.size(0)

    # calculate the x/y/z coordinates for each voxel in the grid
    channels = torch.meshgrid(*(torch.arange(dim, device=device) for dim in grid_shape))
    ones = torch.ones(*grid_shape, device=device)
    points = torch.stack(channels + (ones,), dim=-1)
    points = points.flatten(0, -2).T.unsqueeze(0)
    P = torch.matmul(K, T)[:, :3, :]
    cam_points = torch.matmul(P, points)
    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + eps)

    # hide samples behind camera
    x = cam_points[:, 0, :]
    y = cam_points[:, 1, :]
    z = cam_points[:, 2, :]

    valid = z > 0
    valid = valid.reshape(BS, *grid_shape)

    # grid_sample needs a 2d input so we add a dummy dimension
    pix_coords = pix_coords.permute(0, 2, 1).unsqueeze(1)
    pix_coords = (pix_coords - 0.5) * 2
    # features = features.permute(0, 1, 3, 2)
    values = F.grid_sample(features, pix_coords, align_corners=False)

    # restore to grid shape
    values = values.squeeze(2).unflatten(-1, grid_shape)
    values *= valid.unsqueeze(1)

    return values
