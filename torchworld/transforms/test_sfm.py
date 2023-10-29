import unittest

import torch

from torchworld.structures.cameras import PerspectiveCameras
from torchworld.structures.grid import GridImage
from torchworld.transforms.sfm import project


class TestSFM(unittest.TestCase):
    def test_project(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        camera = PerspectiveCameras(device=device)
        mask = torch.ones(2, 1, 4, 6, device=device, dtype=dtype)
        src = GridImage(
            data=torch.ones(2, 3, 4, 6, device=device, dtype=dtype),
            camera=camera,
            time=torch.rand(2, device=device),
            mask=mask,
        )
        depth = GridImage(
            data=torch.ones(2, 1, 4, 6, device=device, dtype=dtype),
            camera=camera,
            time=torch.rand(2, device=device),
            mask=mask,
        )
        vel = GridImage(
            data=torch.zeros(2, 3, 4, 6, device=device, dtype=dtype),
            camera=camera,
            time=torch.rand(2, device=device),
            mask=mask,
        )
        compiled_project = torch.compile(project, fullgraph=True, backend="eager")
        proj = compiled_project(dst=src, src=src, depth=depth, vel=vel)
        torch.testing.assert_allclose(proj.data, src.data)
        torch.testing.assert_allclose(proj.mask.data, mask.data)
