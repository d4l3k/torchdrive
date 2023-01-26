from typing import Tuple

import torch

MIN_CAR_SIZE: float = 1 / 3
MAX_CAR_SIZE = 20
WORLD_D = 300
WORLD_W = 200
WORLD_H = 15
MAX_VEL = 70  # m/s


def decode_bboxes3d(
    bboxes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    converts the bboxes array into 3d world (xyz, vel, size)
    input dim: (BS, num_queries, 9)
    output dim: (3, BS, num_queries, 3)
    """
    xyz = bboxes[..., 0:3].clone()
    xyz[..., 0] *= WORLD_D
    xyz[..., 0] -= WORLD_D / 2
    xyz[..., 1] *= WORLD_W
    xyz[..., 1] -= WORLD_W / 2
    xyz[..., 2] *= WORLD_H
    xyz[..., 2] -= WORLD_H / 2

    vel = bboxes[..., 6:9].clone()
    vel *= MAX_VEL
    vel -= MAX_VEL / 2

    sizes = bboxes[..., 3:6].clone()
    sizes *= MAX_CAR_SIZE - MIN_CAR_SIZE
    sizes += MIN_CAR_SIZE

    return xyz, vel, sizes


def bboxes3d_to_points(bboxes: torch.Tensor, time: float = 0.0) -> torch.Tensor:
    """
    converts the bboxes array into 3d world coordinates
    input dim: (BS, num_queries, 9)
    output dim: (BS, num_queries, 8, 3)
    """

    xyz, vel, sizes = decode_bboxes3d(bboxes)

    xyz += vel * time

    # each corner point
    patterns = [
        (0.5, 0.5, 0.5),
        (0.5, 0.5, -0.5),
        (0.5, -0.5, 0.5),
        (0.5, -0.5, -0.5),
        (-0.5, 0.5, 0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, -0.5, -0.5),
    ]
    points = []
    for sx, sy, sz in patterns:
        signed_sizes = sizes.clone()
        signed_sizes[..., 0] *= sx
        signed_sizes[..., 1] *= sy
        signed_sizes[..., 2] *= sz
        point = xyz + signed_sizes
        points.append(point)
    return torch.stack(points, dim=2)


def points_to_bboxes2d(
    points: torch.Tensor, K: torch.Tensor, ex: torch.Tensor, w: int, h: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    points_to_bboxes2d projects the 3d bounding boxes into image space
    coordinates.

    Args:
        points: (BS, num_queries, 8, 3)
        K:  camera intrinsics normalized (world to camera) [BS, 4, 4]
        ex: camera extrinsics (camera pos to world) [BS, 4, 4]
        w: image width
        h: image height

    Returns:
        pix_points: (BS, num_queries, 8, 2)
        bboxes: (BS, num_queries, 4)
    """

    BS = len(K)
    num_queries = points.shape[1]
    device = K.device

    K = K.clone()
    K[:, 0] *= w
    K[:, 1] *= h
    K[:, 2, 2] = 1

    # convert to list of points
    points = points.reshape(-1, 3)
    ones = torch.ones(*points.shape[:-1], 1, device=device)
    points = torch.cat([points, ones], dim=-1).unsqueeze(2)

    inv_ex = ex.pinverse()
    # inv_ex: convert to image space
    # K: convert to local space
    P = torch.matmul(K, inv_ex)

    # repeat for each query*points combo
    P = P.repeat_interleave(num_queries * 8, dim=0)

    points = torch.matmul(P, points)

    # box mask
    invalid_mask = (points[:, 2, 0] < 0).reshape(BS, num_queries, 8).any(dim=2)

    # TODO: discard boxes where points is negative?
    pix_coords = points[:, (0, 1), 0] / (points[:, 2:3, 0] + 1e-8)

    pix_coords = pix_coords.reshape(BS, num_queries, 8, 2)

    xmin = pix_coords[..., 0].amin(dim=-1)
    xmax = pix_coords[..., 0].amax(dim=-1)
    ymin = pix_coords[..., 1].amin(dim=-1)
    ymax = pix_coords[..., 1].amax(dim=-1)

    bbox = torch.stack((xmin, ymin, xmax, ymax), dim=-1)

    return pix_coords, bbox, invalid_mask
