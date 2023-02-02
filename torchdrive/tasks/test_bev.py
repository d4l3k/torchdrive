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
        ctx.add_scalar("test", bev.shape[-1])
        return {
            "foo": torch.tensor(5.0),
        }


class TestBEV(unittest.TestCase):
    def test_bev_task_van(self) -> None:
        m = BEVTaskVan(
            tasks={"dummy": DummyBEVTask()},
            hr_tasks={"hr_dummy": DummyBEVTask()},
            cam_shape=(48, 64),
            bev_shape=(4, 4),
            cameras=["left", "right"],
            dim=5,
            hr_dim=1,
            num_upsamples=1,
            num_encode_frames=2,
            num_backprop_frames=1,
            writer=MagicMock(),
        )
        losses = m(dummy_batch(), global_step=500)
        self.assertCountEqual(losses.keys(), ["dummy-foo", "hr_dummy-foo"])
        writer = m.writer
        self.assertIsNotNone(writer)
        self.assertEqual(writer.add_scalar.call_count, 2)
        self.assertEqual(
            writer.add_scalar.mock_calls,
            [
                call("dummy-test", 4, global_step=500),
                call("hr_dummy-test", 8, global_step=500),
            ],
        )
