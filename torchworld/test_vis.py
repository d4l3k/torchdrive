import unittest

import pythreejs

import torch

from torchworld import vis

from torchworld.structures.cameras import PerspectiveCameras
from torchworld.structures.grid import Grid3d, GridImage
from torchworld.structures.lidar import Lidar
from torchworld.transforms.transform3d import Transform3d


class TestVis(unittest.TestCase):
    def test_camera(self) -> None:
        out = vis.camera(PerspectiveCameras())
        self.assertIsInstance(out, pythreejs.AxesHelper)

    def test_grid_image(self) -> None:
        img = GridImage(
            data=torch.rand(1, 3, 10, 20),
            camera=PerspectiveCameras(),
            time=torch.rand(1),
        )
        out = vis.grid_image(img)
        self.assertIsInstance(out, pythreejs.Group)

    def test_grid_3d_occupancy(self) -> None:
        grid = Grid3d.from_volume(
            data=torch.zeros(1, 1, 3, 4, 5),
            voxel_size=1 / 3,
        )
        out = vis.grid_3d_occupancy(grid)
        self.assertIsInstance(out, pythreejs.Group)

    def test_path(self) -> None:
        positions = Transform3d(matrix=torch.rand(5, 4, 4))
        out = vis.path(positions)
        self.assertIsInstance(out, pythreejs.Group)

    def test_lidar(self) -> None:
        lidar = Lidar(data=torch.rand(1, 4, 10))
        out = vis.lidar(lidar)
        self.assertIsInstance(out, pythreejs.Points)
