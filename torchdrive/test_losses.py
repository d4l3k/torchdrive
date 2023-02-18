import unittest

import torch
from torch.testing import assert_close

from torchdrive.losses import (
    losses_backward,
    projection_loss,
    smooth_loss,
    SSIM,
    ssim_loss,
    tvl1_loss,
)


class TestLosses(unittest.TestCase):
    def test_ssim(self) -> None:
        ssim = SSIM()
        x = torch.rand(2, 3, 9, 16)
        y = torch.rand(2, 3, 9, 16)
        out = ssim(x, y)
        self.assertEqual(out.shape, (2, 3, 9, 16))

        out2 = ssim_loss(x, y)
        assert_close(out, out2)

    def test_projection_loss(self) -> None:
        x = torch.rand(2, 3, 9, 16)
        y = torch.rand(2, 3, 9, 16)
        mask = torch.rand(2, 1, 9, 16)
        out = projection_loss(x, y, mask)
        self.assertEqual(out.shape, (2, 1, 9, 16))

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

    def test_losses_backwards_value(self) -> None:
        t = torch.arange(4, dtype=torch.float)
        t.requires_grad = True
        weights = torch.arange(4, dtype=torch.float)
        losses = {
            "t": t,
        }
        losses_backward(losses,weights=weights)
        torch.testing.assert_close(losses["t"], (t*weights).sum())

        t = torch.arange(1, dtype=torch.float).mean()
        t.requires_grad = True
        weights = torch.arange(4, dtype=torch.float)
        losses = {
            "t": t,
        }
        losses_backward(losses,weights=weights)
        torch.testing.assert_close(losses["t"], (t*weights).sum())

        losses = {
            "t": t,
        }
        losses_backward(losses)
        torch.testing.assert_close(losses["t"], t.mean())

        t = torch.ones(1).mean()
        t.requires_grad=True
        losses = {
            "t": t*2,
        }
        losses_backward(losses)
        torch.testing.assert_close(losses["t"], torch.tensor(2.0))
