from torchdrive.tasks.diff_traj import XYEmbedding

import unittest

import torch

class TestDiffTraj(unittest.TestCase):
    def test_diff_traj(self):
        dim = 20

        traj = XYEmbedding(
            shape=(16, 24),
            dim=dim,
            scale=1.0,
        )

        input = torch.tensor([
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
        ]).unsqueeze(0)

        output = traj(input)
        self.assertEqual(output.shape, (1, 5, dim))

        positions = traj.decode(output)
        self.assertEqual(positions.shape, (1, 5, 2))
        torch.testing.assert_close(positions, input)
