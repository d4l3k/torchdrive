import unittest

import torch

from torchdrive.models.det import BDD100KDet


class TestDet(unittest.TestCase):
    def test_bdd100kdet(self) -> None:
        device = torch.device("cpu")
        m = BDD100KDet(device, half=False)
        out = m(torch.rand(1, 3, 48, 64))
        self.assertIsNotNone(out)
