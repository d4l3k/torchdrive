import unittest
from unittest.mock import MagicMock

import torch

from torchdrive.data import dummy_batch

from torchdrive.tasks.bev import Context
from torchdrive.tasks.path import PathTask


class TestPath(unittest.TestCase):
    def test_path_task(self) -> None:
        m = PathTask(
            bev_shape=(4, 4),
            bev_dim=5,
            dim=6,
            num_heads=2,
            num_layers=1,
        )
        batch = dummy_batch()
        ctx = Context(
            log_img=True,
            log_text=True,
            global_step=0,
            writer=MagicMock(),
            start_frame=1,
            scaler=None,
            name="det",
            output="/invalid",
            weights=batch.weight,
        )
        bev = torch.rand(2, 5, 4, 4)
        losses = m(ctx, batch, bev)
        self.assertCountEqual(losses.keys(), ["position"])
        self.assertEqual(losses["position"].shape, (2,))
