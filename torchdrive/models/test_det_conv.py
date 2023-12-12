import unittest

import torch

from torchdrive.models.det_conv import DetConvDecoder


class TestDetConv(unittest.TestCase):
    def test_det_conv_decoder(self) -> None:
        m = DetConvDecoder(
            dim=16,
            bev_shape=(4, 5),
        )
        levels = [
            torch.rand(2, 16, 32, 40),
            torch.rand(2, 16, 16, 20),
            torch.rand(2, 16, 8, 10),
            torch.rand(2, 16, 4, 5),
        ]
        classes, bboxes = m(levels)
        self.assertEqual(classes.shape, (2, 80, 11))
        self.assertEqual(bboxes.shape, (2, 80, 9))

        self.assertIsNotNone(m.decoder_params())
