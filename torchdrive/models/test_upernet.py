import unittest

import torch

from torchdrive.models.upernet import upernet_resnet

class TestUperNet(unittest.TestCase):
    def test_resnet(self) -> None:
        m = upernet_resnet(2, backbone='resnet18', pretrained=False)
        out = m(torch.rand(2, 3, 48, 64))
        self.assertEqual(out.shape, (2, 2, 48, 64))
