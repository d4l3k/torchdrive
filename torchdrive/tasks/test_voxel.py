import unittest
from unittest.mock import MagicMock

import torch

from torchdrive.data import dummy_batch

from torchdrive.tasks.bev import Context
from torchdrive.tasks.voxel import VoxelTask


class TestVoxel(unittest.TestCase):
    def test_voxel_task(self) -> None:
        m = VoxelTask(
            cameras=["left", "right"],
            cam_shape=(320, 240),
            dim=5,
            height=12,
        )
        ctx = Context(
            log_img=True,
            log_text=True,
            global_step=0,
            writer=MagicMock(),
            start_frame=1,
            scaler=None,
            name="det",
            output="",
        )
        bev = torch.rand(2, 5, 4, 4)
        batch = dummy_batch()
        losses = m(ctx, batch, bev)
        self.assertCountEqual(
            losses.keys(),
            [
                "tvl1",
                "lossproj/right/o1/s0",
                "lossproj/right/o1/s1",
                "lossproj/right/o1/s2",
                "lossproj/left/o1/s0",
                "lossproj/left/o1/s1",
                "lossproj/left/o1/s2",
            ],
        )
