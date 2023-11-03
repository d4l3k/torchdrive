import unittest

import torch

from torchdrive.debug import assert_not_nan

from torchdrive.positional_encoding import (
    apply_sin_cos_enc1d,
    apply_sin_cos_enc2d,
    positional_encoding,
    sequence_encoding,
    sin_cos_enc,
    sin_cos_enc2d,
)


class TestPositionalEncoding(unittest.TestCase):
    def test_positional_encoding(self) -> None:
        out = positional_encoding(4, 5)
        self.assertEqual(out.shape, (1, 6, 4, 5))
        assert_not_nan(out)

    def test_positional_encoding_device(self) -> None:
        device = torch.device("cpu")
        out = positional_encoding(4, 5, device=device)
        self.assertEqual(out.shape, (1, 6, 4, 5))
        assert_not_nan(out)

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
