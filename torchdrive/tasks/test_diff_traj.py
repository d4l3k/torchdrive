import unittest

import torch
from torchdrive.tasks.diff_traj import XEmbedding, XYEmbedding, XYLinearEmbedding


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
