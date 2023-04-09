import unittest
from typing import Dict
from unittest.mock import call, MagicMock

import torch
from torchvision import models

from torchdrive.data import Batch, dummy_batch
from torchdrive.models.bev import RiceBackbone
from torchdrive.models.regnet import RegNetEncoder
from torchdrive.tasks.bev import BEVTask, BEVTaskVan, Context


class DummyBEVTask(BEVTask):
    def forward(
        self, ctx: Context, batch: Batch, bev: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        bev.mean().backward()
        ctx.add_scalar("test", bev.shape[-1])

        # check that start position is at zero
        cam_T = batch.cam_T[:, ctx.start_frame]
        zero = (
            cam_T.new_tensor([0.0, 0.0, 0.0, 1.0])
            .expand(cam_T.size(0), -1)
            .unsqueeze(-1)
        )
        long_cam_T, mask, lengths = batch.long_cam_T
        assert isinstance(long_cam_T, torch.Tensor)
        long_cam_T = long_cam_T[:, ctx.start_frame]
        torch.testing.assert_allclose(cam_T.matmul(zero), zero)
        torch.testing.assert_allclose(long_cam_T.matmul(zero), zero)

        return {
            "foo": torch.tensor(5.0),
        }


class TestBEV(unittest.TestCase):
    def test_bev_task_van(self) -> None:
        cam_shape = (48, 64)
        bev_shape = (4, 4)
        cameras = ["left", "right"]
        dim = 8
        hr_dim = 1
        m = BEVTaskVan(
            tasks={"dummy": DummyBEVTask()},
            hr_tasks={"hr_dummy": DummyBEVTask()},
            cam_shape=cam_shape,
            bev_shape=bev_shape,
            cameras=cameras,
            dim=dim,
            hr_dim=hr_dim,
            num_encode_frames=2,
            num_backprop_frames=1,
            writer=MagicMock(),
            backbone=RiceBackbone(
                dim=dim,
                cam_dim=dim,
                hr_dim=hr_dim,
                bev_shape=bev_shape,
                input_shape=(48 // 16, 64 // 16),
                num_frames=2,
                cameras=cameras,
                num_upsamples=1,
            ),
            cam_encoder=lambda: RegNetEncoder(
                cam_shape=cam_shape, dim=dim, trunk=models.regnet_x_400mf
            ),
        )
        losses = m(dummy_batch(), global_step=500)
        self.assertCountEqual(losses.keys(), ["dummy-foo", "hr_dummy-foo"])
        writer = m.writer
        self.assertIsNotNone(writer)
        self.assertEqual(writer.add_scalar.call_count, 2)
        self.assertEqual(
            writer.add_scalar.mock_calls,
            [
                call("dummy-test", 4, global_step=500),
                call("hr_dummy-test", 8, global_step=500),
            ],
        )
        self.assertEqual(len(m.param_opts(lr=1e-4)), 2)
