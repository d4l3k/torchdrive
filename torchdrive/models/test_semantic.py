import unittest

import torch

from torchdrive.models.semantic import BDD100KSemSeg


class TestSemantic(unittest.TestCase):
    def test_bdd100ksemseg(self) -> None:
        device = torch.device("cpu")
        m = BDD100KSemSeg(device)
        out = m(torch.rand(1, 3, 48, 64, device=device))
        self.assertIsNotNone(out)
