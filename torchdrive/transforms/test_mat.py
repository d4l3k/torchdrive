import unittest

import torch

from torchdrive.transforms.mat import (
    random_translation,
    random_z_rotation,
    transformation_from_parameters,
    voxel_to_world,
)


class TestMat(unittest.TestCase):
    def test_transformation_from_parameters(self) -> None:
        out = transformation_from_parameters(torch.rand(2, 1, 3), torch.rand(2, 1, 3))
        self.assertEqual(out.shape, (2, 4, 4))

    def test_random_z_rotation(self) -> None:
        out = random_z_rotation(batch_size=2, device=torch.device("cpu"))
        self.assertEqual(out.shape, (2, 4, 4))

    def test_random_translation(self) -> None:
        out = random_translation(
            batch_size=2, distances=(1, 2, 0), device=torch.device("cpu")
        )
        self.assertEqual(out.shape, (2, 4, 4))

    def test_voxel_to_world(self) -> None:
        out = voxel_to_world((-128, -128, 0), 3, torch.device("cpu"))
        self.assertEqual(out.shape, (1, 4, 4))
