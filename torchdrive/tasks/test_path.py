import unittest
from unittest.mock import MagicMock

import torch
from parameterized import parameterized

from torchdrive.data import dummy_batch
from torchdrive.tasks.bev import Context
from torchdrive.tasks.path import PathTask, unflatten_strided


class TestPath(unittest.TestCase):
    @parameterized.expand([(True,), (False,)])
    def test_path_task(self, one_shot: bool) -> None:
        m = PathTask(
            bev_shape=(4, 4),
            bev_dim=8,
            dim=8,
            num_heads=2,
            num_layers=1,
            num_ar_iters=3,
            downsample=2,
            one_shot=one_shot,
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
        self.assertCountEqual(
            losses.keys(),
            [
                "position/0",
                "position/1",
                "position/2",
                # "ae/0",
                # "ae/1",
                # "ae/2",
                # "rel_dists/0",
                # "rel_dists/1",
                # "rel_dists/2",
            ],
        )
        self.assertEqual(losses["position/0"].shape, tuple())

        m.param_opts(lr=1e-4)

    def test_unflatten_strided(self) -> None:
        inp = torch.arange(12)
        out = unflatten_strided(inp, stride=3)
        self.assertEqual(out.shape, (3, 4))
        torch.testing.assert_close(
            out,
            torch.tensor(
                (
                    (0, 3, 6, 9),
                    (1, 4, 7, 10),
                    (2, 5, 8, 11),
                )
            ),
        )
