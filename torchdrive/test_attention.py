import unittest

import torch

from torchdrive.attention import attention


class TestAttention(unittest.TestCase):
    def test_attention(self) -> None:
        q = torch.rand(2, 8, 16)
        kv = torch.rand(2, 8, 32)

        out = attention(q, kv, dim=16, num_heads=1, dropout_p=0.1)
        self.assertEqual(out.shape, (2, 8, 16))
