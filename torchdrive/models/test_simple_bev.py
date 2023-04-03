import unittest

import torch

from torchdrive.data import dummy_batch

from torchdrive.models.simple_bev import Segnet, segnet_rgb


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
