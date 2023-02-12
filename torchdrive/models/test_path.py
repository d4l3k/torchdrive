import unittest

import torch

from torchdrive.models.path import PathTransformer, pos_to_bucket, rel_dists


class PathTest(unittest.TestCase):
    def test_path_transformer(self) -> None:
        bev_dim = 7
        dim = 8
        num_points = 10
        m = PathTransformer(
            bev_shape=(4, 4),
            bev_dim=bev_dim,
            dim=dim,
            num_heads=2,
        )
        out, ae_prev = m(
            bev=torch.rand(2, bev_dim, 4, 4),
            positions=torch.rand(2, 3, num_points),
            final_pos=torch.rand(2, 3),
        )
        self.assertEqual(out.shape, (2, 3, num_points))
        self.assertEqual(ae_prev.shape, (2, 3, num_points))

    def test_path_transformer_infer(self) -> None:
        bev_dim = 7
        dim = 8
        m = PathTransformer(
            bev_shape=(4, 4),
            bev_dim=bev_dim,
            dim=dim,
            num_heads=2,
        )

        m.eval()
        with torch.no_grad():
            out = m.infer(
                bev=torch.rand(1, bev_dim, 4, 4),
                seq=torch.rand(1, 3, 2),
                final_pos=torch.rand(1, 3),
                n=3,
            )
        self.assertEqual(out.shape, (1, 3, 5))

    def test_rel_dists(self) -> None:
        out = rel_dists(torch.rand(2, 3, 6))
        self.assertEqual(out.shape, (2, 6))
        self.assertEqual(out[0, 0], 0)

        out = rel_dists(torch.arange(10).float().unsqueeze(0).unsqueeze(0))
        self.assertEqual(out.sum(), 9)

    def test_pos_to_bucket(self) -> None:
        pos = torch.tensor(
            (
                (10, -1),
                (10, 1),
                (1, 1),
                (1, -1),
                (0, 1),
                (0, -1),
            )
        ).float()
        self.assertEqual(
            pos_to_bucket(pos, buckets=14).tolist(),
            [10, 10, 8, 12, 7, 0],
        )
