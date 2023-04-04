import unittest

import torch

from torchdrive.models.bev import (
    BEVMerger,
    BEVUpsampler,
    CamBEVEncoder,
    GridTransformer,
    RiceBackbone,
)
from torchdrive.testing import skipIfNoCUDA


class TestBEVTransformer(unittest.TestCase):
    def test_cpu(self) -> None:
        m = GridTransformer(
            output_shape=(4, 4),
            input_shape=(4, 6),
            input_dim=8,
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
                input_dim=16,
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
        device = torch.device("cpu")
        m = CamBEVEncoder(
            cameras=["left", "right"],
            bev_shape=(4, 4),
            cam_shape=(48, 64),
            dim=8,
            # compile_fn=torch.compile,
        ).to(device)
        img = torch.rand(2, 3, 48, 64, device=device)
        camera_frames = {
            "left": img,
            "right": img,
        }
        cam_feats, out = m(camera_frames)
        cam_feats, out = m(camera_frames, pause=True)
        self.assertEqual(out.shape, (2, 8, 4, 4))
        self.assertCountEqual(cam_feats.keys(), ("left", "right"))

    def test_bev_merger(self) -> None:
        m = BEVMerger(num_frames=3, bev_shape=(4, 4), dim=5)
        out = m([torch.rand(2, 5, 4, 4)] * 3)
        self.assertEqual(out.shape, (2, 5, 4, 4))

    def test_bev_upsampler(self) -> None:
        m = BEVUpsampler(num_upsamples=2, bev_shape=(4, 4), dim=5, output_dim=3)
        out = m(torch.rand(2, 5, 4, 4))
        self.assertEqual(out.shape, (2, 3, 16, 16))

    def test_rice_backbone(self) -> None:
        cameras = ["left", "right"]
        num_frames = 2
        m = RiceBackbone(
            cam_dim=15,
            dim=16,
            bev_shape=(4, 4),
            input_shape=(4, 6),
            hr_dim=4,
            num_upsamples=1,
            num_frames=num_frames,
            cameras=cameras,
        )
        x, x4 = m(
            {cam: [torch.rand(2, 15, 4, 6)] * num_frames for cam in cameras}, None
        )
        self.assertEqual(x.shape, (2, 4, 8, 8))
        self.assertEqual(x4.shape, (2, 16, 4, 4))
