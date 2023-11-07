import unittest
from unittest.mock import MagicMock

import torch

from torchdrive.data import dummy_batch
from torchdrive.tasks.ae import AETask

from torchdrive.tasks.bev import Context


class TestAE(unittest.TestCase):
    def test_ae_task(self) -> None:
        m = AETask(
            cameras=["left", "right"],
            cam_shape=(320, 240),
            bev_shape=(4, 4),
            dim=8,
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
        bev = torch.rand(2, 8, 4, 4)
        losses = m(ctx, batch, [bev])
        self.assertCountEqual(losses.keys(), ["ssim/left", "ssim/right"])
