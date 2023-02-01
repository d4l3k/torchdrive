import unittest

import torch
import torch.nn.functional as F

from torchdrive.transforms.depth import BackprojectDepth, Project3D


class TestDepth(unittest.TestCase):
    def test_project(self) -> None:
        backproject_depth = BackprojectDepth(
            height=4,
            width=6,
        )
        project_3d = Project3D(
            height=4,
            width=6,
        )
        target_inv_K = torch.rand(2, 4, 4)
        src_K = torch.rand(2, 4, 4)
        T = torch.rand(2, 4, 4)
        depth = torch.rand(2, 4, 6)
        color = torch.rand(2, 3, 4, 6)

        cam_points = backproject_depth(depth, target_inv_K)
        pix_coords = project_3d(cam_points, src_K, T)

        color = F.grid_sample(
            color,
            pix_coords,
            mode="nearest",
            padding_mode="border",
            align_corners=False,
        )
        self.assertEqual(color.shape, (2, 3, 4, 6))
