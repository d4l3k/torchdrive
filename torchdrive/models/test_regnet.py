import unittest

import torch
from torchvision import models

from torchdrive.models.regnet import ConvPEBlock, RegNetEncoder


class TestRegNet(unittest.TestCase):
    def test_regnet(self) -> None:
        m = RegNetEncoder((48, 64), 10, trunk=models.regnet_y_400mf)
        out = m(torch.rand(1, 3, 48, 64))
        self.assertEqual(out.shape, (1, 10, 3, 4))

    def test_conv_pe_block(self) -> None:
        m = ConvPEBlock(4, 5, (4, 4))
        out = m(torch.rand(2, 4, 4, 4))
        self.assertEqual(out.shape, (2, 5, 4, 4))
