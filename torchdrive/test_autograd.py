import unittest
from unittest.mock import MagicMock

import torch

from torchdrive.autograd import (
    autograd_context,
    autograd_optional,
    autograd_pause,
    autograd_resume,
    log_grad_norm,
)


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

    def test_resume_twice(self) -> None:
        t = torch.zeros(1, 2)
        t.requires_grad = True

        t_paused = autograd_pause(t)
        t_paused.sum().backward()
        autograd_resume(t_paused)

        with self.assertRaisesRegex(RuntimeError, "tensor has already been resumed"):
            t_paused.sum().backward()

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

    def test_context_nested(self) -> None:
        a = torch.rand(1, 2)
        a.requires_grad = True
        with autograd_context(a) as a1:
            with autograd_context(a1) as a2:
                a2.mean().backward()
        self.assertIsNotNone(a.grad)

    def test_optional(self) -> None:
        a = torch.rand(1, 2)
        a.requires_grad = True

        with autograd_optional(a) as a_paused, autograd_optional(None) as none:
            a_paused.sum().backward()
            self.assertIsNotNone(a_paused.grad)
            self.assertIsNone(a.grad)

        self.assertIsNotNone(a.grad)

    def test_log_grad_norm(self) -> None:
        a = torch.rand(1, 2)
        a.requires_grad = True

        writer = MagicMock()
        b = log_grad_norm(a, writer, key="key", tag="tag", global_step=10)
        b.mean().backward()
        self.assertEquals(writer.add_scalars.call_count, 1)
        self.assertIsNotNone(a.grad)
