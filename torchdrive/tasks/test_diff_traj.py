import unittest
from unittest.mock import MagicMock, patch

import torch
from torchdrive.data import Batch, dummy_batch

from torchdrive.tasks.diff_traj import (
    DiffTraj,
    XEmbedding,
    XYEmbedding,
    XYLinearEmbedding,
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

        m = DiffTraj(
            cameras=["left"],
            dim=32,
            dim_feedforward=8,
            num_layers=2,
            num_heads=1,
            cam_shape=(48, 64),
        )

        batch = dummy_batch()
        writer = MagicMock()
        losses = m(batch, global_step=0, writer=writer)
