import unittest

import torch

from torchdrive.models.det import BDD100KDet, ConvMLP, DetBEVDecoder


class TestDet(unittest.TestCase):
    def test_bdd100kdet(self) -> None:
        device = torch.device("cpu")
        m = BDD100KDet(device, half=False)
        out = m(torch.rand(1, 3, 48, 64))
        self.assertIsNotNone(out)

    def test_conv_mlp(self) -> None:
        a = ConvMLP(3, 4, 5)
        out = a(torch.rand(2, 3, 6))
        self.assertEqual(out.shape, (2, 5, 6))

    def test_det_bev_decoder(self) -> None:
        m = DetBEVDecoder(
            bev_shape=(4, 4),
            dim=16,
            num_queries=10,
            num_heads=2,
        )
        classes, bboxes = m(torch.rand(2, 16, 4, 4))
        self.assertEqual(classes.shape, (2, 10, 11))
        self.assertEqual(bboxes.shape, (2, 10, 9))
