import unittest
from unittest.mock import MagicMock

import torch

from torchdrive.data import dummy_batch

from torchdrive.tasks.bev import Context
from torchdrive.tasks.voxel import axis_grid, VoxelTask


class TestVoxel(unittest.TestCase):
    def test_voxel_task(self) -> None:
        device = torch.device("cpu")
        m = VoxelTask(
            cameras=["left", "right"],
            cam_shape=(320, 240),
            dim=5,
            height=12,
            device=device,
            render_batch_size=1,
            n_pts_per_ray=10,
        ).to(device)
        batch = dummy_batch().to(device)
        ctx = Context(
            log_img=True,
            log_text=True,
            global_step=0,
            writer=MagicMock(),
            start_frame=1,
            scaler=None,
            name="det",
            output="",
            weights=batch.weight,
        )
        bev = torch.rand(2, 5, 4, 4, device=device)
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
                "lossproj/right/o-1/s0",
                "lossproj/right/o-1/s1",
                "lossproj/right/o-1/s2",
                "lossproj/left/o-1/s0",
                "lossproj/left/o-1/s1",
                "lossproj/left/o-1/s2",
            ],
        )

    def test_semantic_voxel_task(self) -> None:
        device = torch.device("cpu")
        m = VoxelTask(
            cameras=["left", "right"],
            cam_shape=(320, 240),
            dim=5,
            height=12,
            device=device,
            semantic=["left"],
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
            output="",
            weights=batch.weight,
        )
        bev = torch.rand(2, 5, 4, 4)
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
                "semantic/left",
                "semantic/right",
                "lossproj/right/o-1/s0",
                "lossproj/right/o-1/s1",
                "lossproj/right/o-1/s2",
                "lossproj/left/o-1/s0",
                "lossproj/left/o-1/s1",
                "lossproj/left/o-1/s2",
            ],
        )

    def test_axis_grid(self) -> None:
        grid, color = axis_grid(torch.rand(2, 1, 3, 4, 5))
        self.assertEqual(grid.shape, (2, 1, 3, 4, 5))
        self.assertEqual(color.shape, (2, 3, 3, 4, 5))
