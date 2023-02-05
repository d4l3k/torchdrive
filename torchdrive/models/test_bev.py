import unittest

import torch

from torchdrive.models.bev import (
    BEVMerger,
    BEVUpsampler,
    CamBEVEncoder,
    GridTransformer,
)
from torchdrive.testing import skipIfNoCUDA


class TestBEVTransformer(unittest.TestCase):
    def test_cpu(self) -> None:
        m = GridTransformer(
            output_shape=(4, 4),
            input_shape=(4, 6),
            dim=8,
            num_inputs=3,
            num_heads=2,
        )
        cam_feats = [torch.rand(2, 8, 4, 6)] * 3
        out = m(cam_feats)
        self.assertEqual(out.shape, (2, 8, 4, 4))

    @skipIfNoCUDA()
    def test_cuda(self) -> None:
        m = (
            GridTransformer(
                output_shape=(4, 4),
                input_shape=(4, 6),
                dim=16,
                num_inputs=3,
                num_heads=2,
            )
            .cuda()
            .bfloat16()
        )
        cam_feats = [torch.rand(2, 16, 4, 6).cuda().bfloat16()] * 3
        out = m(cam_feats)
        self.assertEqual(out.shape, (2, 16, 4, 4))

    def test_cam_bev_encoder(self) -> None:
        m = CamBEVEncoder(
            cameras=["left", "right"], bev_shape=(4, 4), cam_shape=(48, 64), dim=8
        )
        img = torch.rand(2, 3, 48, 64)
        camera_frames = {
            "left": img,
            "right": img,
        }
        out = m(camera_frames)
        self.assertEqual(out.shape, (2, 8, 4, 4))

    def test_bev_merger(self) -> None:
        m = BEVMerger(num_frames=3, bev_shape=(4, 4), dim=5)
        out = m([torch.rand(2, 5, 4, 4)] * 3)
        self.assertEqual(out.shape, (2, 5, 4, 4))

    def test_bev_upsampler(self) -> None:
        m = BEVUpsampler(num_upsamples=2, bev_shape=(4, 4), dim=5, output_dim=3)
        out = m(torch.rand(2, 5, 4, 4))
        self.assertEqual(out.shape, (2, 3, 16, 16))
