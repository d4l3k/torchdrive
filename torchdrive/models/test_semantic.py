import unittest

import torch

from torchdrive.models.semantic import BDD100KSemSeg

try:
    import mmcv
except ModuleNotFoundError:
    mmcv = False


class TestSemantic(unittest.TestCase):
    @unittest.skipIf(not mmcv, "must have mmcv")
    def test_bdd100ksemseg(self) -> None:
        device = torch.device("cpu")
        mmlab = BDD100KSemSeg(device, mmlab=True)
        ts = BDD100KSemSeg(device, mmlab=False)
        inp = torch.rand(1, 3, 48, 64, device=device)
        out_mmlab = mmlab(inp)
        out_ts = ts(inp)

        torch.testing.assert_allclose(out_mmlab, out_ts)
