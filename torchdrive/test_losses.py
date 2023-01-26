import unittest

import torch

from torchdrive.losses import SSIM, smooth_loss, tvl1_loss


class TestLosses(unittest.TestCase):
    def test_ssim(self) -> None:
        ssim = SSIM()
        out = ssim(torch.rand(2, 3, 9, 16), torch.rand(2, 3, 9, 16))
        self.assertEqual(out.shape, (2, 3, 9, 16))

    def test_smooth(self) -> None:
        out = smooth_loss(torch.rand(2, 3, 9, 16), torch.rand(2, 3, 9, 16))
        self.assertEqual(out.shape, tuple())

    def test_tvl1(self) -> None:
        out = tvl1_loss(torch.rand(2, 3, 4, 5))
        self.assertEqual(out.shape, (2,))
