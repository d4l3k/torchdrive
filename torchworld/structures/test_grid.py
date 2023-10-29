import unittest

import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import Transform3d

from torchworld.structures.grid import Grid3d, GridImage


class TestGrid(unittest.TestCase):
    def test_grid_3d(self) -> None:
        grid = Grid3d(
            data=torch.rand(2, 3, 4, 5, 6),
            local_to_world=Transform3d(),
            time=torch.rand(2),
        )

        grid = grid.to("cpu")
        grid = grid.cpu()
        grid = grid.replace()

        self.assertEqual(len(grid), 2)
        self.assertEqual(grid.device, grid.data.device)
        self.assertEqual(grid.dtype, torch.float)
        self.assertEqual(grid.grid_shape(), (4, 5, 6))

    def test_grid_image(self) -> None:
        grid = GridImage(
            data=torch.rand(2, 3, 4, 5),
            camera=PerspectiveCameras(),
            time=torch.rand(2),
        )

        grid = grid.to("cpu")
        grid = grid.cpu()
        grid = grid.replace()

        self.assertEqual(grid.grid_shape(), (4, 5))

    def test_grid_3d_from_volume(self) -> None:
        grid = Grid3d.from_volume(
            data=torch.rand(2, 3, 4, 5, 6),
            voxel_size=1.0,
            volume_translation=(1.0, -1.0, 0.1),
            time=1.0,
        )
        self.assertIsInstance(grid, Grid3d)
