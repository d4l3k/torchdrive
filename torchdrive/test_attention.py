import unittest

import torch
from torch.testing import assert_close

from torchdrive.attention import (
    attention,
    flash_attention,
    naive_attention,
    xformers_attention,
)
from torchdrive.testing import manual_seed, skipIfNoCUDA


class TestAttention(unittest.TestCase):
    def test_attention(self) -> None:
        q = torch.rand(2, 8, 16)
        kv = torch.rand(2, 8, 32)

        out = attention(q, kv, dim=16, num_heads=1)
        self.assertEqual(out.shape, (2, 8, 16))

    @skipIfNoCUDA()
    def test_compat(self) -> None:
        manual_seed(0)
        q = torch.rand(2, 8, 16).cuda().bfloat16()
        kv = torch.rand(2, 8, 32).cuda().bfloat16()

        funcs = [xformers_attention, naive_attention, attention, flash_attention]
        outputs = []
        for attn in funcs:
            out = attn(q, kv, dim=16, num_heads=1)
            self.assertEqual(out.shape, (2, 8, 16), attn)
            outputs.append(out)

        for i in range(len(outputs) - 1):
            print(i)
            assert_close(outputs[i], outputs[i + 1])
