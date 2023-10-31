import unittest

import pythreejs

from torchworld.structures.cameras import PerspectiveCameras

from torchworld.vis import add_camera


class TestVis(unittest.TestCase):
    def test_camera(self) -> None:
        scene = pythreejs.Scene()
        out = add_camera(scene, PerspectiveCameras())
        self.assertIsInstance(out, pythreejs.AxesHelper)
