import random
import unittest
from unittest.mock import MagicMock, patch

import torch
from torchdrive.data import Batch, dummy_batch
from torchdrive.tasks.diff_traj import (
    compute_dream_pos,
    DiffTraj,
    random_traj,
    square_mask,
    XEmbedding,
    XYEmbedding,
    XYLinearEmbedding,
    XYMLPEncoder,
    XYSineMLPEncoder,
)


class TestDiffTraj(unittest.TestCase):
    def test_xy_embedding(self):
        torch.manual_seed(0)

        dim = 32

        traj = XYEmbedding(
            shape=(16, 24),
            dim=dim,
            scale=1.0,
        )

        input = torch.tensor(
            [
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (-1.0, 0.0),
                (0.0, -1.0),
            ]
        ).unsqueeze(0)

        output = traj(input)
        self.assertEqual(output.shape, (1, 5, dim))

        positions = traj.decode(output)
        self.assertEqual(positions.shape, (1, 5, 2))
        torch.testing.assert_close(positions, input)

    def test_xy_linear_embedding(self):
        torch.manual_seed(0)

        dim = 32

        traj = XYLinearEmbedding(
            shape=(16, 24),
            dim=dim,
            scale=1.0,
        )

        input = torch.tensor(
            [
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (-1.0, 0.0),
                (0.0, -1.0),
            ]
        ).unsqueeze(0)

        output = traj(input)
        self.assertEqual(output.shape, (1, 5, dim))

        positions = traj.decode(output)
        self.assertEqual(positions.shape, (1, 5, 2))
        torch.testing.assert_close(positions, input)

        loss = traj.ae_loss(input)
        self.assertEqual(
            loss.shape,
            (),
        )

        loss.backward()
        for param in traj.parameters():
            self.assertIsNotNone(param.grad)

    def test_x_embedding(self):
        torch.manual_seed(0)

        dim = 20

        traj = XEmbedding(shape=16, dim=dim, scale=1.0)

        input = torch.tensor(
            [
                0.0,
                -1.0,
                1.0,
            ]
        ).unsqueeze(0)

        output = traj(input)
        self.assertEqual(output.shape, (1, 3, dim))

        positions = traj.decode(output)
        self.assertEqual(positions.shape, (1, 3))
        torch.testing.assert_close(positions, input)

        loss = traj.ae_loss(input)
        self.assertEqual(
            loss.shape,
            (),
        )

        loss.backward()
        for param in traj.parameters():
            self.assertIsNotNone(param.grad)

    def test_diff_traj(self):
        torch.manual_seed(0)
        random.seed(0)

        m = DiffTraj(
            cameras=["left"],
            dim=32,
            dim_feedforward=8,
            num_layers=2,
            num_heads=1,
            cam_shape=(48, 64),
            num_inference_timesteps=2,
        )

        batch = dummy_batch()
        writer = MagicMock()
        losses = m(batch, global_step=0, writer=writer)

        param_opts = m.param_opts(1)
        all_params = set()
        for group in param_opts:
            for param in group["params"]:
                all_params.add(param)

        for name, param in m.named_parameters():
            self.assertIn(param, all_params, name)

    def test_xy_mlp_encoder(self):
        torch.manual_seed(0)

        m = XYMLPEncoder(
            dim=32,
            max_dist=1.0,
        )

        input = torch.tensor(
            [
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (-1.0, 0.0),
                (0.0, -1.0),
            ]
        ).unsqueeze(0)

        out = m(input)
        self.assertEqual(out.shape, (1, 5, 32))

        decoded = m.decode(out)
        self.assertEqual(decoded.shape, (1, 5, 2))

        loss = m.loss(out, input)
        self.assertEqual(loss.shape, (1, 5))
        loss.sum().backward()
        for param in m.parameters():
            self.assertIsNotNone(param.grad)

    def test_xy_sine_mlp_encoder(self):
        torch.manual_seed(0)

        m = XYSineMLPEncoder(
            dim=32,
            max_dist=128.0,
        )

        input = torch.tensor(
            [
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (-1.0, 0.0),
                (0.0, -1.0),
            ]
        ).unsqueeze(0)

        out = m(input)
        self.assertEqual(out.shape, (1, 5, 32))

        decoded = m.decode(out)
        self.assertEqual(decoded.shape, (1, 5, 2))

        loss = m.loss(out, input)
        self.assertEqual(loss.shape, (1, 5))
        loss.sum().backward()
        for param in m.parameters():
            self.assertIsNotNone(param.grad)

    def test_square_mask(self):
        input = torch.tensor(
            [
                [True, True],
                [True, False],
            ]
        )
        target = torch.tensor(
            [[[True, True], [True, True]], [[True, False], [False, True]]]
        )

        output = square_mask(input, num_heads=3)
        self.assertEqual(output.shape, (6, 2, 2))
        torch.testing.assert_close(output[:2], target)

    def test_compute_dream_pos(self):
        positions = torch.rand(2, 18, 2)
        mask = torch.ones(2, 18)
        pred_traj = torch.rand(2, 18, 2)

        dream_target, dream_mask, dream_positions, dream_pred = compute_dream_pos(
            positions, mask, pred_traj
        )
        self.assertEqual(dream_target.shape, (2, 16, 2))
        self.assertEqual(dream_mask.shape, (2, 16))
        self.assertEqual(dream_positions.shape, (2, 16, 2))
        self.assertEqual(dream_pred.shape, (2, 16, 2))

    def test_random_traj(self):
        BS = 10
        vel = torch.ones(BS, 1)
        seq_len = 18
        traj = random_traj(BS=BS, seq_len=seq_len, device="cpu", vel=vel)
        self.assertEqual(traj.shape, (BS, seq_len, 2))
