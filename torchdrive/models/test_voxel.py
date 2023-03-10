import unittest

import torch

from torchdrive.models.voxel import ResUpsample3d


class TestVoxel(unittest.TestCase):
    def test_res_upsample3d(self) -> None:
        m = ResUpsample3d(num_upsamples=2, dim=16, min_dim=7, depth=1)
        out = m(torch.rand(2, 16, 1, 2, 3))
        self.assertEqual(out.shape, (2, 7, 4, 8, 12))
