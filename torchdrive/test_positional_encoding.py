import unittest

import torch

from torchdrive.debug import assert_not_nan

from torchdrive.positional_encoding import positional_encoding, sequence_encoding


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
