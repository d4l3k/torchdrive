import unittest

import torch
import torch.nn.functional as F

from torchdrive.transforms.depth import (
    BackprojectDepth,
    depth_to_disp,
    disp_to_depth,
    Project3D,
)


class TestDepth(unittest.TestCase):
    def test_project(self) -> None:
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)

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
        T1 = torch.rand(2, 4, 4)
        T2 = torch.rand(2, 4, 4)
        depth = torch.rand(2, 4, 6, requires_grad=True)
        color = torch.rand(2, 3, 4, 6)

        cam_points = backproject_depth(depth, target_inv_K, T2)
        torch.testing.assert_close(cam_points[:, 3].mean(), torch.tensor(1.0))

        cam_points += 0
        pix_coords = project_3d(cam_points, src_K, T1)

        out = pix_coords.sum()
        print(out.item())
        torch.testing.assert_close(out, torch.tensor(-12.4088287))

        color = F.grid_sample(
            color,
            pix_coords,
            mode="nearest",
            padding_mode="border",
            align_corners=False,
        )
        self.assertEqual(color.shape, (2, 3, 4, 6))
        color.sum().backward()

    def test_disp_to_depth(self) -> None:
        out = disp_to_depth(torch.tensor((0.0, 100000)))
        expected = torch.tensor((100.0, 0.0))
        print(out, expected)
        torch.testing.assert_close(out, expected)

    def test_disp_conversions(self) -> None:
        expected_disp = torch.rand(10)
        depth = disp_to_depth(expected_disp)
        torch.testing.assert_close(depth_to_disp(depth), expected_disp)
