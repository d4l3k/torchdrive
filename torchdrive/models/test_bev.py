import unittest

import torch
from torch.testing import assert_close

from torchdrive.models.bev import (
    FlashAttnCamBEVTransformer,
    NaiveCamBEVTransformer,
    XformersCamBEVTransformer,
)
from torchdrive.testing import manual_seed, skipIfNoCUDA, xformers_available


class TestBEVTransformer(unittest.TestCase):
    def test_cam_bev_naive(self) -> None:
        m = NaiveCamBEVTransformer(
            bev_shape=(4, 4),
            cam_shape=(4, 6),
            dim=10,
            num_cameras=3,
            num_heads=2,
        )
        cam_feats = torch.rand(2, 3 * 10, 4, 6)
        out = m(cam_feats)
        self.assertEqual(out.shape, (2, 10, 4, 4))

    def test_cam_bev_flash_attn(self) -> None:
        m = FlashAttnCamBEVTransformer(
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

    def test_cam_bev_xformers(self) -> None:
        m = XformersCamBEVTransformer(
            bev_shape=(4, 4),
            cam_shape=(4, 6),
            dim=16,
            num_cameras=3,
            num_heads=2,
        )
        cam_feats = torch.rand(2, 3 * 16, 4, 6)

        if xformers_available():
            out = m(cam_feats)
            self.assertEqual(out.shape, (2, 16, 4, 4))

    @skipIfNoCUDA()
    def test_compat(self) -> None:
        impls = [
            # NaiveCamBEVTransformer, # stock PT attention has projection layers
            FlashAttnCamBEVTransformer,
            XformersCamBEVTransformer,
        ]

        cam_feats = torch.rand(2, 3 * 16, 4, 6).cuda().bfloat16()

        out = []

        for i, impl in enumerate(impls):
            manual_seed(0)

            m = (
                impl(
                    bev_shape=(4, 4),
                    cam_shape=(4, 6),
                    dim=16,
                    num_cameras=3,
                    num_heads=2,
                )
                .cuda()
                .bfloat16()
            )
            out.append(m(cam_feats))

        for i in range(len(out) - 1):
            print(i)
            assert_close(out[i], out[i + 1], atol=6e-3, rtol=0.04)
