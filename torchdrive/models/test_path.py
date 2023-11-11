import unittest

import torch

from torchdrive.models.path import (
    PathAutoRegressiveTransformer,
    PathOneShotTransformer,
    pos_to_bucket,
    rel_dists,
    XYEncoder,
)


class PathTest(unittest.TestCase):
    def test_path_transformer(self) -> None:
        bev_dim = 8
        dim = 8
        num_points = 7
        m = PathAutoRegressiveTransformer(
            bev_shape=(4, 4),
            bev_dim=bev_dim,
            dim=dim,
            num_heads=2,
            max_seq_len=num_points + 3,
        )
        out, ae_prev = m(
            bev=torch.rand(2, bev_dim, 4, 4),
            positions=torch.rand(2, 3, num_points),
            final_pos=torch.rand(2, 3),
        )
        self.assertEqual(out.shape, (2, 3, num_points))
        self.assertEqual(ae_prev.shape, (2, 3, num_points))

        m.set_dropout(0.2)

    def test_path_transformer_infer(self) -> None:
        bev_dim = 8
        dim = 8
        m = PathAutoRegressiveTransformer(
            bev_shape=(4, 4),
            bev_dim=bev_dim,
            dim=dim,
            num_heads=2,
            max_seq_len=10,
            pos_dim=200,
        )

        enc = XYEncoder(num_buckets=100, max_dist=10.0)

        m.eval()
        with torch.no_grad():
            out = PathAutoRegressiveTransformer.infer(
                enc,
                m,
                bev=torch.rand(1, bev_dim, 4, 4),
                seq=torch.rand(1, 3, 2),
                final_pos=torch.rand(1, 3),
                n=3,
            )
        self.assertEqual(out.shape, (1, 2, 5))

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

    def test_xy_encoder(self) -> None:
        enc = XYEncoder(num_buckets=100, max_dist=10.0)
        inp = torch.tensor((-11, -10, -5, 0, 5, 9.8, 11), dtype=torch.float).expand(
            2, 2, -1
        )

        x, y = enc.encode_labels(inp)
        self.assertEqual(x.shape, (2, 7))
        self.assertEqual(y.shape, (2, 7))
        torch.testing.assert_close(x, y)
        torch.testing.assert_close(x[0], torch.tensor((0, 0, 25, 50, 75, 99, 99)))

        xy1h = enc.encode_one_hot(inp)
        xy1h.requires_grad = True
        self.assertEqual(xy1h.shape, (2, 200, 7))

        xyd = enc.decode(xy1h)
        torch.testing.assert_close(xyd[:, 0], xyd[:, 1])
        self.assertEqual(xyd.shape, (2, 2, 7))
        torch.testing.assert_close(
            xyd[0, 0], torch.tensor((-10, -10, -5, 0, 5, 9.8, 9.8), dtype=torch.float)
        )

        loss = enc.loss(xy1h, inp)
        loss.mean().backward()
        self.assertIsNotNone(xy1h.grad)

    def test_path_one_shot_transformer(self) -> None:
        bev_dim = 8
        dim = 8
        num_points = 7
        m = PathOneShotTransformer(
            bev_shape=(4, 4),
            bev_dim=bev_dim,
            dim=dim,
            num_heads=2,
            max_seq_len=num_points + 3,
            num_queries=10,
        )
        out, _ = m(
            bev=torch.rand(2, bev_dim, 4, 4),
            positions=torch.rand(2, 3, num_points),
            final_pos=torch.rand(2, 3),
        )
        self.assertEqual(out.shape, (2, 3, num_points))
