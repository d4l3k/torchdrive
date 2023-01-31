import unittest
from typing import Dict
from unittest.mock import call, MagicMock

import torch

from torchdrive.data import Batch, dummy_batch

from torchdrive.tasks.bev import BEVTask, BEVTaskVan, Context


class DummyBEVTask(BEVTask):
    def forward(
        self, ctx: Context, batch: Batch, bev: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        bev.mean().backward()
        ctx.add_scalar("test", 1.5)
        return {
            "foo": torch.tensor(5.0),
        }


class TestBEV(unittest.TestCase):
    def test_bev_task_van(self) -> None:
        m = BEVTaskVan(
            tasks={"dummy": DummyBEVTask()},
            cam_shape=(48, 64),
            bev_shape=(4, 4),
            cameras=["left", "right"],
            dim=5,
            encode_frames=2,
            writer=MagicMock(),
        )
        losses = m(dummy_batch(), global_step=500)
        self.assertIn("dummy-foo", losses)
        writer = m.writer
        self.assertIsNotNone(writer)
        self.assertEqual(writer.add_scalar.call_count, 1)
        self.assertEqual(
            writer.add_scalar.mock_calls, [call("dummy-test", 1.5, global_step=500)]
        )
