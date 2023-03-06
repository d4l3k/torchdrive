import unittest

import torch
from parameterized import parameterized
from torchvision import models

from torchdrive.models.regnet import ConvPEBlock, RegNetEncoder, UpsamplePEBlock


class TestRegNet(unittest.TestCase):
    # pyre-fixme[16]: no attribute expand
    @parameterized.expand(
        [
            (True,),
            (False,),
        ]
    )
    def test_regnet(self, use_f4: bool) -> None:
        m = RegNetEncoder((64, 96), 10, trunk=models.regnet_y_400mf, use_f4=use_f4)
        out = m(torch.rand(1, 3, 64, 96))
        self.assertEqual(out.shape, (1, 10, 4, 6))
        out.mean().backward()

    def test_conv_pe_block(self) -> None:
        m = ConvPEBlock(4, 5, (4, 4))
        out = m(torch.rand(2, 4, 4, 4))
        self.assertEqual(out.shape, (2, 5, 4, 4))

    def test_upsample_pe_block(self) -> None:
        m = UpsamplePEBlock(4, 5, (4, 4))
        out = m(torch.rand(2, 4, 4, 4))
        self.assertEqual(out.shape, (2, 5, 8, 8))
