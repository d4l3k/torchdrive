import unittest

import torch

from torchdrive.debug import assert_not_nan

from torchworld.positional_encoding import (
    apply_sin_cos_enc1d,
    apply_sin_cos_enc2d,
    sequence_encoding,
    sin_cos_enc,
    sin_cos_enc2d,
    LearnedPositionalEncodingSeq,
    LearnedPositionalEncoding1d,
    LearnedPositionalEncoding2d,
)


class TestPositionalEncoding(unittest.TestCase):
    def test_sequence_encoding(self) -> None:
        x = torch.rand(2, 5, 6)
        y = sequence_encoding(x)
        self.assertEqual(x.shape, y.shape)
        assert_not_nan(y)

    def test_sin_cos_enc(self) -> None:
        device = torch.device("cpu")
        out = sin_cos_enc(seq_len=5, dim=6, device=device)
        self.assertEqual(out.shape, (5, 6))

    def test_apply_sin_cos_enc1d(self) -> None:
        out = apply_sin_cos_enc1d(torch.rand(2, 6, 4))
        self.assertEqual(out.shape, (2, 6, 4))

    def test_sin_cos_enc2d(self) -> None:
        device = torch.device("cpu")
        out = sin_cos_enc2d(h=4, w=5, dim=8, device=device)
        self.assertEqual(out.shape, (8, 4, 5))

    def test_apply_sin_cos_enc2d(self) -> None:
        out = apply_sin_cos_enc2d(torch.rand(2, 8, 4, 5))
        self.assertEqual(out.shape, (2, 8, 4, 5))

    def test_learned_positonal_encoding_seq(self) -> None:
        m = LearnedPositionalEncodingSeq(10, 5)
        out = m(torch.rand(2, 8, 5))
        self.assertEqual(out.shape, (2, 8, 5))

    def test_learned_positonal_encoding_1d(self) -> None:
        m = LearnedPositionalEncoding1d(10, 5)
        out = m(torch.rand(2, 5, 8))
        self.assertEqual(out.shape, (2, 5, 8))

    def test_learned_positonal_encoding_2d(self) -> None:
        m = LearnedPositionalEncoding2d((2, 3), 5)
        out = m(torch.rand(2, 5, 2, 3))
        self.assertEqual(out.shape, (2, 5, 2, 3))

