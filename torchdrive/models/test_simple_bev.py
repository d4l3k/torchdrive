
import unittest

import torch

from torchdrive.models.simple_bev import Segnet


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
            X=X,
            Y=Y,
            Z=Z,
            latent_dim=latent_dim,
        )

        raw_feat, feat, segmentation, instance_center, instance_offset = m(
            rgb_camXs=torch.rand(BS, S, 3, H, W),
            pix_T_cams=torch.rand(BS, S, 4, 4),
            cam0_T_camXs=torch.rand(BS, S, 4, 4),
        )
        self.assertEqual(raw_feat.shape, (BS, latent_dim, Z, X))
        self.assertEqual(feat.shape, (BS, latent_dim, Z, X))
        self.assertEqual(segmentation.shape, (BS, 1, Z, X))
        self.assertEqual(instance_center.shape, (BS, 1, Z, X))
        self.assertEqual(instance_offset.shape, (BS, 2, Z, X))
