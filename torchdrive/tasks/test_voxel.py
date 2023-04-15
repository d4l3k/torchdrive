import unittest
from unittest.mock import MagicMock

import torch

from torchdrive.data import dummy_batch

from torchdrive.tasks.bev import Context
from torchdrive.tasks.voxel import axis_grid, VoxelTask

SHARED_LOSSES = [
    "tvl1",
    "lossproj-voxel/right/o1",
    "lossproj-voxel/left/o1",
    "lossproj-voxel/right/o0",
    "lossproj-voxel/left/o0",
    "lossproj-voxel/right/o-1",
    "lossproj-voxel/left/o-1",
    "lossproj-cam/right/o1",
    "lossproj-cam/left/o1",
    "lossproj-cam/right/o0",
    "lossproj-cam/left/o0",
    "lossproj-cam/right/o-1",
    "lossproj-cam/left/o-1",
    "losssmooth/voxel/left/vel",
    "losssmooth/voxel/left/color",
    "losssmooth/voxel/right/vel",
    "losssmooth/voxel/right/color",
    "losssmooth/cam/left/vel",
    "losssmooth/cam/left/color",
    "losssmooth/cam/right/vel",
    "losssmooth/cam/right/color",
]


class TestVoxel(unittest.TestCase):
    def test_voxel_task(self) -> None:
        device = torch.device("cpu")
        cameras = ["left", "right"]
        m = VoxelTask(
            cameras=cameras,
            cam_shape=(320, 240),
            cam_feats_shape=(320 // 16, 240 // 16),
            dim=4,
            hr_dim=5,
            cam_dim=6,
            height=12,
            device=device,
            render_batch_size=1,
            n_pts_per_ray=10,
            offsets=(-1, 0, 1),
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
            cam_feats={cam: torch.rand(2, 6, 320 // 16, 240 // 16) for cam in cameras},
        )
        bev = torch.rand(2, 5, 4, 4, device=device)
        losses = m(ctx, batch, bev)
        self.assertCountEqual(losses.keys(), SHARED_LOSSES)

    def test_semantic_voxel_task(self) -> None:
        device = torch.device("cpu")
        cameras = ["left", "right"]
        m = VoxelTask(
            cameras=cameras,
            cam_shape=(320, 240),
            cam_feats_shape=(320 // 16, 240 // 16),
            dim=4,
            cam_dim=4,
            hr_dim=5,
            height=12,
            device=device,
            semantic=["left"],
            offsets=(-1, 0, 1),
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
            cam_feats={cam: torch.rand(2, 4, 320 // 16, 240 // 16) for cam in cameras},
        )
        bev = torch.rand(2, 5, 4, 4)
        losses = m(ctx, batch, bev)
        self.assertCountEqual(
            losses.keys(),
            [
                "semantic/left",
                "semantic/right",
            ]
            + SHARED_LOSSES,
        )

    def test_axis_grid(self) -> None:
        grid, color = axis_grid(torch.rand(2, 1, 3, 4, 5))
        self.assertEqual(grid.shape, (2, 1, 3, 4, 5))
        self.assertEqual(color.shape, (2, 3, 3, 4, 5))
