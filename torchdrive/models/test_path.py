import unittest

import torch

from torchdrive.models.path import PathTransformer


class PathTest(unittest.TestCase):
    def test_path_transformer(self) -> None:
        bev_dim = 7
        dim = 6
        num_points = 10
        m = PathTransformer(
            bev_shape=(4, 4),
            bev_dim=bev_dim,
            dim=dim,
            num_heads=2,
        )
        out = m(
            bev=torch.rand(2, bev_dim, 4, 4),
            positions=torch.rand(2, 3, num_points),
        )
        self.assertEqual(out.shape, (2, 3, num_points))
