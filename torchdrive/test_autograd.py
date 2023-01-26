import unittest

import torch

from torchdrive.autograd import autograd_context, autograd_pause, autograd_resume


class TestAutograd(unittest.TestCase):
    def test_pause(self) -> None:
        t = torch.zeros(1, 2)
        t.requires_grad = True

        t_paused = autograd_pause(t)
        t_paused.sum().backward()
        self.assertIsNotNone(t_paused.grad)
        self.assertIsNone(t.grad)

        autograd_resume(t_paused)
        self.assertIsNotNone(t.grad)

    def test_nograd_resume(self) -> None:
        t = torch.zeros(1, 2)
        t.requires_grad = True

        t_paused = autograd_pause(t)
        with self.assertRaisesRegex(AssertionError, "missing grad"):
            autograd_resume(t_paused)

    def test_context(self) -> None:
        a = torch.rand(1, 2)
        a.requires_grad = True

        b = torch.rand(1, 2)
        b.requires_grad = True

        with autograd_context(a, b) as (a_paused, b_paused):
            loss = (a_paused + b_paused).sum()
            loss.backward()
            self.assertIsNone(a.grad)
            self.assertIsNone(b.grad)
            self.assertIsNotNone(a_paused.grad)
            self.assertIsNotNone(b_paused.grad)

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)

    def test_context_single(self) -> None:
        a = torch.rand(1, 2)
        a.requires_grad = True

        with autograd_context(a) as a_paused:
            a_paused.sum().backward()
            self.assertIsNotNone(a_paused.grad)
            self.assertIsNone(a.grad)

        self.assertIsNotNone(a.grad)
