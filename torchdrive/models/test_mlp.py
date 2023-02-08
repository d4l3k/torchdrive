import unittest

import torch

from torchdrive.models.mlp import ConvMLP


class TestMLP(unittest.TestCase):
    def test_conv_mlp(self) -> None:
        a = ConvMLP(3, 4, 5)
        out = a(torch.rand(2, 3, 6))
        self.assertEqual(out.shape, (2, 5, 6))
