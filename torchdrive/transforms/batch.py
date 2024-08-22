from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Tuple

import torch

from torchdrive.data import Batch
from torchdrive.transforms.mat import random_translation, random_z_rotation
from torchvision.transforms import v2


class BatchTransform(ABC):
    """
    BatchTransform is a protocol for batch transforms.
    """

    @abstractmethod
    def __call__(self, batch: Batch) -> Batch:
        ...


class Identity(BatchTransform):
    """
    Returns the original batch.
    """

    def __call__(self, batch: Batch) -> Batch:
        return batch


class Compose(BatchTransform):
    """
    Compose multiple BatchTransforms together.
    """

    def __init__(self, *transforms: BatchTransform) -> None:
        self.transforms: Tuple[BatchTransform] = transforms

    def __call__(self, batch: Batch) -> Batch:
        for transform in self.transforms:
            batch = transform(batch)
        return batch


class NormalizeCarPosition(BatchTransform):
    """
    Normalize car position makes the start_frame's car transform the identity
    matrix and all other frame positions relative to that.
    """

    def __init__(self, start_frame: int) -> None:
        self.start_frame = start_frame

    def __call__(self, batch: Batch) -> Batch:
        start_T = batch.cam_T[:, self.start_frame]
        inv_start_T = start_T.unsqueeze(1).inverse()
        cam_T = batch.cam_T.matmul(inv_start_T)
        long_cam_T, long_cam_T_mask, long_cam_T_lengths = batch.long_cam_T

        long_cam_T = long_cam_T.matmul(inv_start_T)

        return replace(
            batch,
            cam_T=cam_T,
            long_cam_T=(long_cam_T, long_cam_T_mask, long_cam_T_lengths),
        )


class CenterCar(BatchTransform):
    """
    Centers the car XYZ position relative to the target frame. Preserves
    rotations.
    """

    def __init__(self, start_frame: int) -> None:
        self.start_frame = start_frame

    def __call__(self, batch: Batch) -> Batch:
        # calculate the xyz for the target frame
        BS = batch.batch_size()
        start_T = batch.cam_T[:, self.start_frame].inverse()
        point = torch.tensor((0, 0, 0, 1.0))
        coord = start_T.matmul(point)
        coord = coord / coord[:, 3:4]  # normalize w

        # create inverse position transform
        transform = torch.eye(4).repeat(BS, 1, 1)
        transform[:, :3, 3] = -coord[:, :3]
        transform = transform.unsqueeze(1)  # [BS, 1, 4, 4]
        transform = transform.inverse()

        # apply transformation matrix to car position matrices
        cam_T = batch.cam_T.matmul(transform)
        long_cam_T, long_cam_T_mask, long_cam_T_lengths = batch.long_cam_T
        long_cam_T = long_cam_T.matmul(transform)

        return replace(
            batch,
            cam_T=cam_T,
            long_cam_T=(long_cam_T, long_cam_T_mask, long_cam_T_lengths),
        )


class RandomRotation(BatchTransform):
    """
    RandomRotation applies a random z rotation around the origin to the car
    position transform.
    """

    def __call__(self, batch: Batch) -> Batch:
        rot = random_z_rotation(batch.batch_size(), batch.cam_T.device).unsqueeze(1)
        cam_T = batch.cam_T.matmul(rot)
        long_cam_T, long_cam_T_mask, long_cam_T_lengths = batch.long_cam_T
        long_cam_T = long_cam_T.matmul(rot)
        return replace(
            batch,
            cam_T=cam_T,
            long_cam_T=(long_cam_T, long_cam_T_mask, long_cam_T_lengths),
        )


class RandomTranslation(BatchTransform):
    """
    RandomRotation applies a translation around the origin to the car
    position transform.
    """

    def __init__(self, distances: Tuple[float, float, float]) -> None:
        self.distances = distances

    def __call__(self, batch: Batch) -> Batch:
        rot = random_translation(
            batch_size=batch.batch_size(),
            distances=self.distances,
            device=batch.cam_T.device,
        ).unsqueeze(1)
        cam_T = batch.cam_T.matmul(rot)
        long_cam_T, long_cam_T_mask, long_cam_T_lengths = batch.long_cam_T
        long_cam_T = long_cam_T.matmul(rot)
        return replace(
            batch,
            cam_T=cam_T,
            long_cam_T=(long_cam_T, long_cam_T_mask, long_cam_T_lengths),
        )


class ImageTransform(BatchTransform):
    """
    ImageTransform applies a random z rotation around the origin to the car
    position transform.
    """

    def __init__(self, *transform: v2.Transform) -> None:
        self.transform = v2.Compose(transform)

    def __call__(self, batch: Batch) -> Batch:
        new_color = {}
        for key, color in batch.color.items():
            new_color[key] = self.transform(color)
        return replace(batch, color=new_color)
