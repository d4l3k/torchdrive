import unittest
from typing import Dict

import torch

from torchdrive.data import Batch, dummy_batch

from torchdrive.tasks.bev import BEVTask, BEVTaskVan, Context


class DummyBEVTask(BEVTask):
    def forward(
        self, ctx: Context, batch: Batch, bev: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        bev.mean().backward()
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
        )
        losses = m(dummy_batch(), global_step=5)
        self.assertIn("dummy/foo", losses)
