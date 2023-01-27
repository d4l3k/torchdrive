import unittest

import torch
from torchvision import models

from torchdrive.models.regnet import RegNetEncoder


class TestRegNet(unittest.TestCase):
    def test_regnet(self) -> None:
        m = RegNetEncoder(64, 48, 10, trunk=models.regnet_y_400mf)
        out = m(torch.rand(1, 3, 48, 64))
        self.assertEqual(out.shape, (1, 10, 3, 4))
