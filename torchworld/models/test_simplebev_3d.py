import unittest

import torch

from torchdrive.data import dummy_batch

from torchworld.models.simplebev_3d import SimpleBEV3DBackbone
from torchworld.structures.grid import Grid3d, GridImage
from torchworld.transforms.transform3d import Transform3d


class TestSimpleBEV3D(unittest.TestCase):
    def test_simplebev_3d(self) -> None:
        device = torch.device("cpu")
        dtype = torch.half
        num_frames = 3

        backbone = SimpleBEV3DBackbone(
            dim=3,
            hr_dim=3,
            cam_dim=3,
            num_frames=num_frames,
            grid_shape=(-1, -1, 8),
        ).to(device)

        target_grid = Grid3d(
            data=torch.empty(0, 3, 8, 8, 8, device=device, dtype=dtype),
            local_to_world=Transform3d(device=device),
            time=torch.rand(2, device=device),
        )

        batch = dummy_batch()
        x0, (x1, x2, x3, x4), _ = backbone(
            batch=batch,
            camera_features={
                cam: [batch.grid_image(cam, i) for i in range(num_frames)]
                for cam in batch.camera_names()
            },
            target_grid=target_grid,
        )
        self.assertEqual(x0.shape, (2, 3, 8, 8, 8))
        self.assertEqual(x1.shape, (2, 3, 8, 8))
        self.assertEqual(x2.shape, (2, 3, 4, 4))
        self.assertEqual(x3.shape, (2, 3, 2, 2))
        self.assertEqual(x4.shape, (2, 3, 1, 1))
