import unittest

import torch
from torchworld.models.vit import MaskViT


class TestViT(unittest.TestCase):
    def test_vit_mask(self):
        m = MaskViT(
            attention_dropout=0.1,
            cam_shape=(48, 64),
            dim=16,
            weights=None,
        )
        x = torch.rand(1, 3, 48, 64)
        mask = torch.ones(3, 4, dtype=torch.bool)

        features, out = m(x, mask)
        self.assertEqual(features.shape, (1, 768, 3, 4))
        self.assertEqual(out.shape, (1, 3 * 4, 16))

        m.freeze_pretrained_weights()
        needs_grad = [param for param in m.parameters() if param.requires_grad]
        # positional embedding + linear weight/bias
        self.assertEqual(len(needs_grad), 3)
