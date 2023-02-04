import unittest

import torch

from torchdrive.debug import assert_not_nan

from torchdrive.models.transformer import TransformerDecoder


class TransformerTest(unittest.TestCase):
    def test_decoder(self) -> None:
        BS = 2
        dim = 6
        seq_len = 5
        seq2_len = 7
        m = TransformerDecoder(dim=dim, layers=2, num_heads=2)
        sequence = torch.rand(BS, seq_len, dim)
        bev = torch.rand(BS, seq2_len, dim)
        out = m(sequence, bev)
        self.assertEqual(out.shape, (BS, seq_len, dim))
        assert_not_nan(out)
