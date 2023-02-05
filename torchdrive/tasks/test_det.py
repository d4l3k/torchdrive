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
            bev_shape=(4, 4),
            dim=8,
            device=torch.device("cpu"),
        )
        batch = dummy_batch()
        ctx = Context(
            log_img=False,
            log_text=True,
            global_step=0,
            writer=MagicMock(),
            start_frame=1,
            scaler=None,
            name="det",
            output="/invalid",
            weights=batch.weight,
        )
        bev = torch.rand(2, 8, 4, 4)
        losses = m(ctx, batch, bev)
        self.assertIn("unmatched", losses)
