import unittest

import torch
from torchvision import models

from torchdrive.data import dummy_batch
from torchdrive.models.simple_bev import (
    FPN,
    RegNetEncoder,
    ResNetEncoder,
    Segnet,
    segnet_rgb,
    SegnetBackbone,
)


class TestSimpleBEV(unittest.TestCase):
    def test_segnet(self) -> None:
        X = 8
        Y = 16
        Z = 24
        BS = 2
        S = 3
        H = 12
        W = 16
        latent_dim = 7
        m = Segnet(
            grid_shape=(X, Y, Z),
            latent_dim=latent_dim,
        )

        raw_feat, feat, segmentation, instance_center, instance_offset = m(
            rgb_camXs=torch.rand(BS, S, 3, H, W),
            pix_T_cams=torch.rand(BS, S, 4, 4),
            cam0_T_camXs=torch.rand(BS, S, 4, 4),
        )
        self.assertEqual(raw_feat.shape, (BS, latent_dim, X, Y))
        self.assertEqual(feat.shape, (BS, latent_dim, X, Y))
        self.assertEqual(segmentation.shape, (BS, 1, X, Y))
        self.assertEqual(instance_center.shape, (BS, 1, X, Y))
        self.assertEqual(instance_offset.shape, (BS, 2, X, Y))

    def test_segnet_pretrained(self) -> None:
        m = segnet_rgb(grid_shape=(16, 16, 8), pretrained=True)
        self.assertIsInstance(m, Segnet)

    def test_segnet_batch(self) -> None:
        batch = dummy_batch()
        X = 8
        Y = 16
        Z = 24
        latent_dim = 7
        m = Segnet(
            grid_shape=(X, Y, Z),
            latent_dim=latent_dim,
        )
        raw_feat, *_ = m.forward_batch(batch, frame=0)
        self.assertEqual(raw_feat.shape, (batch.batch_size(), latent_dim, X, Y))

    def test_fpn(self) -> None:
        m = FPN(3)
        x, x4 = m(torch.rand(2, 3, 8, 16))
        self.assertEqual(x.shape, (2, 3, 8, 16))
        self.assertEqual(x4.shape, (2, 256, 1, 2))

    def test_segnet_backbone(self) -> None:
        batch = dummy_batch()
        X = 8
        Y = 16
        Z = 24
        latent_dim = 7
        num_frames = 2
        m = SegnetBackbone(
            grid_shape=(X, Y, Z),
            dim=latent_dim,
            num_frames=num_frames,
        )
        camera_features = {
            camera: [torch.rand(batch.batch_size(), latent_dim, 48, 64)] * num_frames
            for camera in batch.cameras()
        }
        x, x4 = m(camera_features, batch)
        self.assertEqual(x.shape, (batch.batch_size(), latent_dim, X, Y))
        self.assertEqual(x4.shape, (batch.batch_size(), 256, X // 8, Y // 8))

    def test_resnet_50(self) -> None:
        m = ResNetEncoder(8, models.resnet50())
        out = m(torch.rand(2, 3, 48, 64))
        self.assertEqual(out.shape, (2, 8, 6, 8))

    def test_resnet_101(self) -> None:
        m = ResNetEncoder(8, models.resnet101())
        out = m(torch.rand(2, 3, 48, 64))
        self.assertEqual(out.shape, (2, 8, 6, 8))

    def test_regnet_x_400mf(self) -> None:
        m = RegNetEncoder(8, models.regnet_x_400mf())
        out = m(torch.rand(2, 3, 48, 64))
        self.assertEqual(out.shape, (2, 8, 6, 8))

    def test_regnet_x_800mf(self) -> None:
        m = RegNetEncoder(8, models.regnet_x_800mf())
        out = m(torch.rand(2, 3, 48, 64))
        self.assertEqual(out.shape, (2, 8, 6, 8))
