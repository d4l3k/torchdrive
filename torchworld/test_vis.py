import unittest

import pythreejs

import torch

from torchworld.structures.cameras import PerspectiveCameras
from torchworld.structures.grid import GridImage
from torchworld.vis import add_camera, add_grid_image


class TestVis(unittest.TestCase):
    def test_camera(self) -> None:
        scene = pythreejs.Scene()
        out = add_camera(scene, PerspectiveCameras())
        self.assertIsInstance(out, pythreejs.AxesHelper)

    def test_grid_iamge(self) -> None:
        scene = pythreejs.Scene()
        img = GridImage(
            data=torch.rand(1, 3, 10, 20),
            camera=PerspectiveCameras(),
            time=torch.rand(1),
        )
        out = add_grid_image(scene, img)
        self.assertIsInstance(out, pythreejs.AxesHelper)
