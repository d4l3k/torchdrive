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
            dim=5,
            device=torch.device("cpu"),
        )
        ctx = Context(
            log_img=False,
            log_text=True,
            global_step=0,
            writer=MagicMock(),
            start_frame=1,
            scaler=None,
            name="det",
            output="/invalid",
        )
        bev = torch.rand(2, 5, 4, 4)
        batch = dummy_batch()
        losses = m(ctx, batch, bev)
        self.assertIn("unmatched", losses)
