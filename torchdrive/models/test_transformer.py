import unittest

import torch

from torchdrive.debug import assert_not_nan

from torchdrive.models.transformer import StockTransformerDecoder, TransformerDecoder


class TransformerTest(unittest.TestCase):
    def test_decoder(self) -> None:
        BS = 2
        dim = 8
        seq_len = 5
        seq2_len = 7
        sequence = torch.rand(BS, seq_len, dim)
        bev = torch.rand(BS, seq2_len, dim)

        for impl in (TransformerDecoder, StockTransformerDecoder):
            with self.subTest(impl=impl):
                m = impl(dim=dim, layers=2, num_heads=2)
                out = m(sequence, bev)
                self.assertEqual(out.shape, (BS, seq_len, dim))
                assert_not_nan(out)
