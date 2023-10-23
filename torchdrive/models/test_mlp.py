import unittest

import torch

from torchdrive.models.mlp import MLP


class TestMLP(unittest.TestCase):
    def test_conv_mlp(self) -> None:
        a = MLP(3, 4, 5, num_layers=3)
        out = a(torch.rand(2, 3, 6))
        self.assertEqual(out.shape, (2, 5, 6))
