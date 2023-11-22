import unittest
from unittest.mock import MagicMock

import torch

from torchdrive.data import dummy_batch

from torchdrive.tasks.bev import Context
from torchdrive.tasks.det import DetTask


class TestDet(unittest.TestCase):
    def test_det_task(self) -> None:
        m = DetTask(
            cameras=["left", "right"],
            cam_shape=(4, 6),
            bev_shape=(4, 5),
            dim=16,
            device=torch.device("cpu"),
            num_queries=10,
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
        grids = [
            torch.rand(2, 16, 32, 40),
            torch.rand(2, 16, 16, 20),
            torch.rand(2, 16, 8, 10),
            torch.rand(2, 16, 4, 5),
        ]
        losses = m(ctx, batch, grids)
        self.assertIn("unmatched", losses)

        m.param_opts(lr=1e-4)
