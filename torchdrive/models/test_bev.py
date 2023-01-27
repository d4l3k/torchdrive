import unittest

import torch

from torchdrive.models.bev import CamBEVTransformer
from torchdrive.testing import skipIfNoCUDA


class TestBEVTransformer(unittest.TestCase):
    def test_cpu(self) -> None:
        m = CamBEVTransformer(
            bev_shape=(4, 4),
            cam_shape=(4, 6),
            dim=10,
            num_cameras=3,
            num_heads=2,
        )
        cam_feats = torch.rand(2, 3 * 10, 4, 6)
        out = m(cam_feats)
        self.assertEqual(out.shape, (2, 10, 4, 4))

    @skipIfNoCUDA()
    def test_cuda(self) -> None:
        m = (
            CamBEVTransformer(
                bev_shape=(4, 4),
                cam_shape=(4, 6),
                dim=16,
                num_cameras=3,
                num_heads=2,
            )
            .cuda()
            .bfloat16()
        )
        cam_feats = torch.rand(2, 3 * 16, 4, 6).cuda().bfloat16()
        out = m(cam_feats)
        self.assertEqual(out.shape, (2, 16, 4, 4))
