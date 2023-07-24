import unittest
from typing import Optional

from parameterized import parameterized

import torch
from pytorch3d.renderer.implicit.utils import RayBundle

from torchdrive.raymarcher import CustomPerspectiveCameras, DepthEmissionRaymarcher


class TestRaymarcher(unittest.TestCase):
    # pyre-fixme[16]: no attribute expand
    @parameterized.expand(
        [
            (0.1,),
            (None,),
        ]
    )
    def test_depth_emission(self, floor: Optional[float]) -> None:
        BS = 2
        X = 3
        Y = 3
        PTS_PER_RAY = 4
        FEATS = 5
        raymarcher = DepthEmissionRaymarcher(
            background=torch.tensor([1] + [0] * (FEATS - 1), dtype=torch.float32),
            floor=floor,
        )
        ray_densities = torch.rand(BS, X, Y, PTS_PER_RAY, 1, requires_grad=True)
        ray_features = torch.rand(BS, X, Y, PTS_PER_RAY, FEATS, requires_grad=True)
        depth, features = raymarcher(
            rays_densities=ray_densities.clone(),
            rays_features=ray_features.clone(),
            ray_bundle=RayBundle(
                origins=torch.rand(BS, X, Y, 3),
                directions=torch.rand(BS, X, Y, 3),
                lengths=torch.rand(BS, X, Y, PTS_PER_RAY),
                xys=torch.rand(BS, X, Y, 2),
            ),
        )
        self.assertEqual(depth.shape, (BS, X, Y))
        self.assertEqual(features.shape, (BS, X, Y, FEATS))
        (depth.mean() + features.mean()).backward()

    def test_cameras(self) -> None:
        cameras = CustomPerspectiveCameras(
            T=torch.rand(2, 4, 4),
            K=torch.rand(2, 4, 4),
            image_size=torch.tensor([[4, 6]], dtype=torch.float).expand(2, -1),
            device=torch.device("cpu"),
        )
        self.assertIsNotNone(cameras.get_world_to_view_transform())
