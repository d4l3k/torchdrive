import unittest

import torch
from torch import nn
from torch.export import export

from torchworld.structures.cameras import PerspectiveCameras
from torchworld.structures.grid import GridImage
from torchworld.transforms.sfm import project


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.device = torch.device("cpu")
        self.dtype: torch.dtype = torch.float32

        self.camera = PerspectiveCameras(device=self.device)
        self.mask: torch.Tensor = torch.ones(
            2, 1, 4, 6, device=self.device, dtype=self.dtype
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        src = GridImage(
            data=data,
            camera=self.camera,
            time=torch.rand(2, device=self.device),
            mask=self.mask,
        )
        depth = GridImage(
            data=torch.ones(2, 1, 4, 6, device=self.device, dtype=self.dtype),
            camera=self.camera,
            time=torch.rand(2, device=self.device),
            mask=self.mask,
        )
        vel = GridImage(
            data=torch.zeros(2, 3, 4, 6, device=self.device, dtype=self.dtype),
            camera=self.camera,
            time=torch.rand(2, device=self.device),
            mask=self.mask,
        )
        return project(dst=src, src=src, depth=depth, vel=vel).data


class TestSFM(unittest.TestCase):
    def test_export(self) -> None:
        data = torch.ones(2, 3, 4, 6)
        model = MyModel()
        model(data)
        exported = export(model, args=(data,))
        self.assertIsNotNone(exported)
        print(exported)
        self.fail()

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
