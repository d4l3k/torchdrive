import unittest

import torch

from torchdrive.models.det_deform import DetDeformableTransformerDecoder


class TestDetDeform(unittest.TestCase):
    def test_det_bev_deform_transformer_decoder(self) -> None:
        m = DetDeformableTransformerDecoder(
            dim=16,
            num_queries=10,
            num_heads=2,
            dim_feedforward=32,
            num_levels=4,
            bev_shape=(4, 5),
        )
        levels = [
            torch.rand(2, 16, 32, 40),
            torch.rand(2, 16, 16, 20),
            torch.rand(2, 16, 8, 10),
            torch.rand(2, 16, 4, 5),
        ]
        classes, bboxes = m(levels)
        self.assertEqual(classes.shape, (2, 10, 11))
        self.assertEqual(bboxes.shape, (2, 10, 9))
