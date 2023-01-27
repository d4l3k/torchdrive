import unittest

import torch

from torchdrive.positional_encoding import positional_encoding


class TestPositionalEncoding(unittest.TestCase):
    def test_positional_encoding(self) -> None:
        out = positional_encoding(4, 5)
        self.assertEqual(out.shape, (1, 6, 4, 5))

    def test_positional_encoding_device(self) -> None:
        device = torch.device("cpu")
        out = positional_encoding(4, 5, device=device)
        self.assertEqual(out.shape, (1, 6, 4, 5))
