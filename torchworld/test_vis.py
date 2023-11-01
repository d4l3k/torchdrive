import unittest

import pythreejs

import torch

from torchworld.structures.cameras import PerspectiveCameras
from torchworld.structures.grid import Grid3d, GridImage
from torchworld.vis import add_camera, add_grid_3d_occupancy, add_grid_image


class TestVis(unittest.TestCase):
    def test_camera(self) -> None:
        scene = pythreejs.Scene()
        out = add_camera(scene, PerspectiveCameras())
        self.assertIsInstance(out, pythreejs.AxesHelper)

    def test_grid_image(self) -> None:
        scene = pythreejs.Scene()
        img = GridImage(
            data=torch.rand(1, 3, 10, 20),
            camera=PerspectiveCameras(),
            time=torch.rand(1),
        )
        out = add_grid_image(scene, img)
        self.assertIsInstance(out, pythreejs.Mesh)

    def test_grid_3d_occupancy(self) -> None:
        scene = pythreejs.Scene()
        grid = Grid3d.from_volume(
            data=torch.zeros(1, 1, 3, 4, 5),
            voxel_size=1 / 3,
        )
        out = add_grid_3d_occupancy(scene, grid)
        self.assertIsInstance(out, pythreejs.Mesh)
