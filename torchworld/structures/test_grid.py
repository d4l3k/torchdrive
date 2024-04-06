import unittest

import torch
from torch import nn

from torchworld.structures.cameras import PerspectiveCameras
from torchworld.structures.grid import Grid3d, GridImage
from torchworld.transforms.transform3d import Transform3d


class TestGrid(unittest.TestCase):
    def test_grid3d(self) -> None:
        grid = Grid3d(
            data=torch.rand(2, 3, 4, 5, 6),
            local_to_world=Transform3d(),
            time=torch.rand(2),
        )

        grid = grid.to("cpu")
        grid = grid.cpu()

        self.assertEqual(len(grid), 2)
        self.assertEqual(grid.device, grid.data.device)
        self.assertEqual(grid.dtype, torch.float)
        self.assertEqual(grid.grid_shape(), (4, 5, 6))

    def test_grid3d_dispatch(self) -> None:
        grid = Grid3d(
            data=torch.rand(2, 3, 4, 5, 6),
            local_to_world=Transform3d(),
            time=torch.rand(2),
        )

        scaled_grid = grid * 2

        self.assertIsInstance(scaled_grid, Grid3d)
        torch.testing.assert_close(grid._data * 2, scaled_grid._data)
        self.assertIs(grid.local_to_world, scaled_grid.local_to_world)
        self.assertIs(grid.time, scaled_grid.time)

    def test_grid3d_repr(self) -> None:
        grid = Grid3d(
            data=torch.rand(2, 3, 4, 5, 6),
            local_to_world=Transform3d(),
            time=torch.rand(2),
        )
        str(grid)
        repr(grid)

    def test_grid3d_numpy(self) -> None:
        grid = Grid3d(
            data=torch.rand(2, 3, 4, 5, 6),
            local_to_world=Transform3d(),
            time=torch.rand(2),
        )
        grid.numpy()


    def test_grid_image(self) -> None:
        grid = GridImage(
            data=torch.rand(2, 3, 4, 5),
            camera=PerspectiveCameras(),
            time=torch.rand(2),
        )

        grid = grid.to("cpu")
        grid = grid.cpu()

        self.assertEqual(grid.grid_shape(), (4, 5))

    def test_grid_image_repr(self) -> None:
        grid = GridImage(
            data=torch.rand(2, 3, 4, 5),
            camera=PerspectiveCameras(),
            time=torch.rand(2),
        )

        str(grid)
        repr(grid)

    def test_grid_image_numpy(self) -> None:
        grid = GridImage(
            data=torch.rand(2, 3, 4, 5),
            camera=PerspectiveCameras(),
            time=torch.rand(2),
        )
        grid.numpy()

    def test_grid_image_permute(self) -> None:
        grid = GridImage(
            data=torch.rand(2, 3, 4, 5),
            camera=PerspectiveCameras(),
            time=torch.rand(2),
            mask=torch.rand(2, 3, 4, 5),
        )

        grid = grid.permute(3, 2, 1, 0)

        self.assertEqual(grid.shape, (5, 4, 3, 2))
        self.assertEqual(grid.mask.shape, (5, 4, 3, 2))

    def test_grid_image_index(self) -> None:
        grid = GridImage(
            data=torch.rand(2, 3, 4, 5),
            camera=PerspectiveCameras(),
            time=torch.rand(2),
            mask=torch.rand(2, 3, 4, 5),
        )

        grid = grid[1:2, 2:, 2:, :3]

        self.assertEqual(grid.shape, (1, 1, 2, 3))
        self.assertEqual(grid.mask.shape, (1, 1, 2, 3))
        self.assertEqual(grid.time.shape, (1,))

    def test_grid_image_dispatch(self) -> None:
        grid = GridImage(
            data=torch.rand(2, 3, 4, 5),
            camera=PerspectiveCameras(),
            time=torch.rand(2),
            mask=torch.rand(2, 1, 4, 5),
        )

        m = nn.Conv2d(3, 6, 1)
        out = m(grid)
        self.assertIsInstance(out, GridImage)
        self.assertIs(out.camera, grid.camera)
        self.assertIs(out.mask, grid.mask)
        self.assertIs(out.time, grid.time)

    def test_grid_3d_from_volume(self) -> None:
        grid = Grid3d.from_volume(
            data=torch.rand(2, 3, 4, 5, 6),
            voxel_size=1.0,
            volume_translation=(1.0, -1.0, 0.1),
            time=1.0,
        )
        self.assertIsInstance(grid, Grid3d)
