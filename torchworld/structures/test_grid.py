import unittest

import torch
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer.cameras import PerspectiveCameras

from torchworld.structures.grid import Grid3d, GridImage

class TestGrid(unittest.TestCase):
    def test_grid_3d(self) -> None:
        grid = Grid3d(
            data=torch.rand(2, 3, 4, 5, 6),
            transform=Transform3d(),
            time=torch.rand(2),
        )

    def test_grid_image(self) -> None:
        grid = GridImage(
            data=torch.rand(2, 3, 4, 5),
            camera=PerspectiveCameras(),
            time=torch.rand(2),
        )

