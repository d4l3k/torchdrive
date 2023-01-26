import unittest

import torch
from pytorch3d.renderer.implicit.utils import RayBundle

from torchdrive.raymarcher import DepthEmissionRaymarcher


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
