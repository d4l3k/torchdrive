import unittest

import torch

from torchdrive.transforms.simple_bev import lift_cam_to_voxel


class TestSimpleBEV(unittest.TestCase):
    def test_lift_cam_to_voxel(self) -> None:
        features = torch.rand(2, 5, 3, 4)
        K = torch.rand(2, 4, 4)
        T = torch.rand(2, 4, 4)
        grid_shape = (2, 3, 4)
        grid, valid = lift_cam_to_voxel(
            features=features, K=K, T=T, grid_shape=grid_shape
        )
        self.assertEqual(grid.shape, (2, 5, 2, 3, 4))
        self.assertEqual(valid.shape, (2, 1, 2, 3, 4))
