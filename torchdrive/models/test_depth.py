import unittest

import torch

from torchdrive.models.depth import DepthDecoder


class TestDepth(unittest.TestCase):
    def test_bev_upsampler(self) -> None:
        m = DepthDecoder(num_upsamples=2, cam_shape=(4, 4), dim=5, final_dim=3)
        depth, vel = m(torch.rand(2, 5, 4, 4))
        self.assertEqual(depth.shape, (2, 16, 16))
        self.assertEqual(vel.shape, (2, 3, 16, 16))
