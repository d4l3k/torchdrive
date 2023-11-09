import unittest
from typing import Dict, List
from unittest.mock import call, MagicMock

import torch
from torchvision import models

from torchdrive.data import Batch, dummy_batch
from torchdrive.models.bev import RiceBackbone
from torchdrive.models.regnet import RegNetEncoder
from torchdrive.tasks.bev import BEVTask, BEVTaskVan, Context
from torchdrive.transforms.batch import Compose, NormalizeCarPosition, RandomRotation


class DummyBEVTask(BEVTask):
    def forward(
        self, ctx: Context, batch: Batch, grids: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for grid in grids:
            grid.mean().backward()

        ctx.add_scalar("test", 1234)

        for cam_feat in ctx.cam_feats.values():
            cam_feat.mean().backward()

        # check that start position is at zero
        car_to_world = batch.car_to_world(ctx.start_frame)
        zero = (
            car_to_world.new_tensor([0.0, 0.0, 0.0, 1.0])
            .expand(car_to_world.size(0), -1)
            .unsqueeze(-1)
        )
        long_cam_T, mask, lengths = batch.long_cam_T
        assert isinstance(long_cam_T, torch.Tensor)
        long_car_to_world = long_cam_T[:, ctx.start_frame].pinverse()
        torch.testing.assert_allclose(car_to_world.matmul(zero), zero)
        torch.testing.assert_allclose(long_car_to_world.matmul(zero), zero)

        return {
            "foo": torch.tensor(5.0),
        }


class TestBEV(unittest.TestCase):
    def test_bev_task_van(self) -> None:
        cam_shape = (48, 64)
        bev_shape = (4, 4)
        grid_shape = (4, 4, 4)
        cameras = ["left", "right"]
        dim = 8
        hr_dim = 1
        m = BEVTaskVan(
            tasks={"dummy": DummyBEVTask()},
            hr_tasks={"hr_dummy": DummyBEVTask()},
            cameras=cameras,
            dim=dim,
            hr_dim=hr_dim,
            num_encode_frames=2,
            num_backprop_frames=1,
            backbone=RiceBackbone(
                dim=dim,
                cam_dim=dim,
                hr_dim=hr_dim,
                grid_shape=grid_shape,
                input_shape=(48 // 16, 64 // 16),
                num_frames=2,
                cameras=cameras,
                num_upsamples=1,
            ),
            cam_encoder=lambda: RegNetEncoder(
                cam_shape=cam_shape, dim=dim, trunk=models.regnet_x_400mf
            ),
            transform=Compose(
                NormalizeCarPosition(start_frame=1),
                RandomRotation(),
            ),
        )
        writer = MagicMock()
        global_step = 500
        batch = dummy_batch()
        self.assertEqual(
            m.should_log(global_step=global_step, BS=batch.batch_size()),
            (True, True),
        )
        losses = m(batch, global_step=500, writer=writer, output="/tmp/out")
        self.assertCountEqual(losses.keys(), ["dummy-foo", "hr_dummy-foo"])
        self.assertIsNotNone(writer)
        self.assertEqual(writer.add_scalar.call_count, 2)
        self.assertEqual(
            writer.add_scalar.mock_calls,
            [
                call("dummy-test", 1234, global_step=500),
                call("hr_dummy-test", 1234, global_step=500),
            ],
        )

        groups = m.param_opts(lr=1e-4)
        names = [group["name"] for group in groups]
        self.assertCountEqual(
            names, ["backbone", "per_cam", "dummy/default", "hr_dummy/default"]
        )
