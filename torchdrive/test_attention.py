import unittest
from typing import List

import torch
from torch.testing import assert_close

from torchdrive.attention import (
    attention,
    AttentionType,
    flash_attention,
    HAS_FLASH_ATTN,
    naive_attention,
    xformers_attention,
)
from torchdrive.testing import manual_seed, skipIfNoCUDA


def _attn_funcs() -> List[AttentionType]:
    funcs: List[AttentionType] = [naive_attention, attention, xformers_attention]
    if HAS_FLASH_ATTN:
        funcs.append(flash_attention)
    return funcs


class TestAttention(unittest.TestCase):
    def test_attention(self) -> None:
        q = torch.rand(2, 8, 16)
        kv = torch.rand(2, 8, 32)

        out = attention(q, kv, dim=16, num_heads=1, dropout_p=0.1)
        self.assertEqual(out.shape, (2, 8, 16))

    def _test_compat(self, causal: bool) -> None:
        manual_seed(0)
        q = torch.rand(2, 8, 16).cuda().bfloat16()
        kv = torch.rand(2, 8, 32).cuda().bfloat16()

        funcs = _attn_funcs()

        outputs = []
        for attn in funcs:
            out = attn(q, kv, dim=16, num_heads=1, causal=causal)
            self.assertEqual(out.shape, (2, 8, 16), attn)
            outputs.append(out)

        for i in range(len(outputs) - 1):
            print(i, funcs[i], funcs[i + 1])
            assert_close(outputs[i], outputs[i + 1])

    @skipIfNoCUDA()
    def test_compat(self) -> None:
        self._test_compat(causal=False)

    @skipIfNoCUDA()
    def test_compat_causal(self) -> None:
        self._test_compat(causal=True)
