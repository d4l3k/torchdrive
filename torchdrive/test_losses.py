import unittest

import torch

from torchdrive.losses import losses_backward, smooth_loss, SSIM, tvl1_loss


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

    def test_losses_backward(self) -> None:
        t = torch.rand(4)
        t.requires_grad = True
        losses = {
            "a": t * 2,
            "b": t * 4,
        }
        losses_backward(losses, weights=torch.rand(4))

        for name, loss in losses.items():
            self.assertFalse(loss.requires_grad, name)

        self.assertIsNotNone(t.grad)
