import unittest
from dataclasses import replace

import torch

from torchdrive.data import Batch, dummy_batch
from torchdrive.transforms.batch import (
    BatchTransform,
    CenterCar,
    Compose,
    Identity,
    NormalizeCarPosition,
    RandomRotation,
    RandomTranslation,
)


class IncrementWeight(BatchTransform):
    def __call__(self, batch: Batch) -> Batch:
        return replace(batch, weight=batch.weight + 1)


class TestBatchTransforms(unittest.TestCase):
    def test_compose(self) -> None:
        transform = Compose(
            IncrementWeight(),
            IncrementWeight(),
        )
        batch = dummy_batch()
        out = transform(batch)
        torch.testing.assert_close(out.weight, batch.weight + 2)

    def test_identity(self) -> None:
        transform = Identity()
        batch = dummy_batch()
        out = transform(batch)
        self.assertIs(batch, out)

    def test_normalize_car_position(self) -> None:
        transform = NormalizeCarPosition(start_frame=1)
        batch = dummy_batch()
        out = transform(batch)
        torch.testing.assert_close(out.cam_T[:, 1], torch.eye(4).expand(2, -1, -1))

    def test_random_rotation(self) -> None:
        transform = Compose(
            NormalizeCarPosition(start_frame=1),
            RandomRotation(),
        )
        batch = dummy_batch()
        out = transform(batch)

        # origin shouldn't change
        zero = torch.tensor((0, 0, 0.1, 1.0)).expand(2, -1).unsqueeze(-1)
        torch.testing.assert_close(out.cam_T[:, 1].matmul(zero), zero)

    def test_random_translation(self) -> None:
        transform = Compose(
            NormalizeCarPosition(start_frame=1),
            RandomTranslation((0, 0, 0)),
        )
        batch = dummy_batch()
        out = transform(batch)

        # origin shouldn't change
        zero = torch.tensor((0, 0, 0, 1.0)).expand(2, -1).unsqueeze(-1)
        torch.testing.assert_close(out.cam_T[:, 1].matmul(zero), zero)

    def test_center_car(self) -> None:
        transform = CenterCar(start_frame=1)
        batch = dummy_batch()
        out = transform(batch)

        # center should be zero point
        zero = torch.tensor((0, 0, 0, 1.0)).expand(2, -1).unsqueeze(-1)
        new_pos = out.cam_T[:, 1].matmul(zero)
        new_pos /= new_pos[:, 3:4]  # normalize w
        torch.testing.assert_close(new_pos, zero)
