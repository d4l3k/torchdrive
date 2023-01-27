import unittest

import torch

from torchdrive.models.bev import CamBEVTransformer, CamBEVTransformerFlashAttn


class TestBEVTransformer(unittest.TestCase):
    def test_cam_bev_transformer(self) -> None:
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

    def test_cam_bev_transformer_flash_attn(self) -> None:
        m = CamBEVTransformerFlashAttn(
            bev_shape=(4, 4),
            cam_shape=(4, 6),
            dim=16,
            num_cameras=3,
            num_heads=2,
        ).bfloat16()
        cam_feats = torch.rand(2, 3 * 16, 4, 6).bfloat16()

        if torch.cuda.is_available():
            m = m.cuda()
            cam_feats = cam_feats.cuda()

            out = m(cam_feats)
            self.assertEqual(out.shape, (2, 16, 4, 4))
