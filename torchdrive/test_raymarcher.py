import unittest

import torch
from pytorch3d.renderer.implicit.utils import RayBundle

from torchdrive.raymarcher import CustomPerspectiveCameras, DepthEmissionRaymarcher


class TestRaymarcher(unittest.TestCase):
    def test_depth_emission(self) -> None:
        BS = 2
        X = 3
        Y = 3
        PTS_PER_RAY = 4
        FEATS = 5
        raymarcher = DepthEmissionRaymarcher(
            background=torch.tensor([1] + [0] * (FEATS - 1), dtype=torch.float32)
        )
        depth, features = raymarcher(
            rays_densities=torch.rand(BS, X, Y, PTS_PER_RAY, 1),
            rays_features=torch.rand(BS, X, Y, PTS_PER_RAY, FEATS),
            ray_bundle=RayBundle(
                origins=torch.rand(BS, X, Y, 3),
                directions=torch.rand(BS, X, Y, 3),
                lengths=torch.rand(BS, X, Y, PTS_PER_RAY),
                xys=torch.rand(BS, X, Y, 2),
            ),
        )
        self.assertEqual(depth.shape, (BS, X, Y))
        self.assertEqual(features.shape, (BS, X, Y, FEATS))

    def test_cameras(self) -> None:
        cameras = CustomPerspectiveCameras(
            T=torch.rand(2, 4, 4),
            K=torch.rand(2, 4, 4),
            image_size=torch.tensor([[4, 6]], dtype=torch.float).expand(2, -1),
            device=torch.device("cpu"),
        )
        self.assertIsNotNone(cameras.get_world_to_view_transform())
