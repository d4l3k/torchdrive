import math
from typing import Tuple

import torch
from pytorch3d.transforms import euler_angles_to_matrix, Transform3d


def transformation_from_parameters(
    axisangle: torch.Tensor, translation: torch.Tensor, invert: bool = False
) -> torch.Tensor:
    """Convert the network's (axisangle, translation) output into a 4x4 matrix"""
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def rot_from_axisangle(vec: torch.Tensor) -> torch.Tensor:
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4), device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def get_translation_matrix(translation_vector: torch.Tensor) -> torch.Tensor:
    """Convert a translation vector into a 4x4 transformation matrix"""
    T = torch.zeros(translation_vector.shape[0], 4, 4, device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def random_z_rotation(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Returns a transformation matrix that will randomly rotate around the z
    plane.

    Args:
        batch_size: the batch size
        device: device to place the transformation matrix onto
    Returns:
        transformation matrix [batch_size, 4, 4]
    """

    angles = torch.zeros((batch_size, 3), device=device)
    angles[:, 2] = torch.rand(batch_size, device=device) * (math.pi * 2)
    rotation_matrix = euler_angles_to_matrix(
        angles,
        "XYZ",
    )
    return (
        Transform3d(device=device).rotate(rotation_matrix).get_matrix().permute(0, 2, 1)
    )


def voxel_to_world(
    center: Tuple[int, int, int], scale: float, device: torch.device
) -> torch.Tensor:
    voxel_to_world = (
        Transform3d(device=device)
        .translate(*center)
        .scale(1 / scale)
        .get_matrix()
        .permute(0, 2, 1)
    )
    return voxel_to_world
