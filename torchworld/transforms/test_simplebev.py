import unittest

import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import Transform3d

from torchworld.structures.grid import Grid3d, GridImage
from torchworld.transforms.simplebev import lift_image_to_3d


class TestSimpleBEV(unittest.TestCase):
    def test_lift_image_to_3d(self) -> None:
        device = torch.device("cpu")
        dtype = torch.half
        dst = Grid3d(
            data=torch.rand(0, 3, 4, 5, 6, device=device, dtype=dtype),
            local_to_world=Transform3d(device=device),
            time=torch.rand(2, device=device),
        )
        src = GridImage(
            data=torch.rand(2, 3, 4, 5, device=device, dtype=dtype),
            camera=PerspectiveCameras(device=device),
            time=torch.rand(2, device=device),
        )

        out, mask = lift_image_to_3d(src, dst)
        self.assertEqual(out.data.shape, (2, 3, 4, 5, 6))
        self.assertEqual(mask.data.shape, (2, 1, 4, 5, 6))


