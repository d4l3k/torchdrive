import unittest

import torch

from torchworld.models.resnet_3d import resnet3d18


class TestResNet3d(unittest.TestCase):
    def test_resnet3d18(self) -> None:
        m = resnet3d18()
        out = m(torch.rand(2, 3, 4, 5, 6))
        self.assertEqual(out.shape, (2, 1000))
