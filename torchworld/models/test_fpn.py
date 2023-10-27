import unittest

import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import Transform3d

from torchworld.models.fpn import Resnet18FPN3d, Resnet18FPNImage

from torchworld.structures.grid import Grid3d, GridImage


class TestFPN(unittest.TestCase):
    def test_resnet18_fpn_3d(self) -> None:
        grid = Grid3d(
            data=torch.rand(2, 3, 8, 16, 24),
            local_to_world=Transform3d(),
            time=torch.rand(2),
        )
        m = Resnet18FPN3d(in_channels=3)
        out = m(grid)
        for grid in out:
            self.assertIsInstance(grid, Grid3d)

    def test_resnet18_fpn_image(self) -> None:
        grid = GridImage(
            data=torch.rand(2, 3, 8, 16),
            camera=PerspectiveCameras(),
            time=torch.rand(2),
        )
        m = Resnet18FPNImage(in_channels=3)
        out = m(grid)
        self.assertEqual(len(out), 4)
        for grid in out:
            self.assertIsInstance(grid, GridImage)
