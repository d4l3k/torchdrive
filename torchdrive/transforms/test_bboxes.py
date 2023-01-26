import unittest

import torch

from torchdrive.transforms.bboxes import bboxes3d_to_points, points_to_bboxes2d


class TestBBoxes(unittest.TestCase):
    def test_bboxes3d_to_points(self) -> None:
        points = torch.tensor((0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0))
        points = points.expand(1, 1, 9)
        out = bboxes3d_to_points(points)
        self.assertEqual(out.shape, (1, 1, 8, 3))
        self.assertEqual(out.abs().sum(), 4.0)

        points = torch.full((1, 1, 9), 1.0)
        out = bboxes3d_to_points(points)
        self.assertEqual(out.shape, (1, 1, 8, 3))
        self.assertTrue(
            torch.equal(
                out[0, 0],
                torch.tensor(
                    (
                        (160, 110, 17.5),
                        (160, 110, -2.5),
                        (160, 90, 17.5),
                        (160, 90, -2.5),
                        (140, 110, 17.5),
                        (140, 110, -2.5),
                        (140, 90, 17.5),
                        (140, 90, -2.5),
                    )
                ),
            ),
            out,
        )

    def test_points_to_bboxes2d(self) -> None:
        bboxes3d = torch.tensor((0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0))
        bboxes3d = bboxes3d.expand(1, 1, 9)
        points = bboxes3d_to_points(bboxes3d)
        pix_coords, bboxes2d, invalid_mask = points_to_bboxes2d(
            points, K=torch.rand(1, 4, 4), ex=torch.rand(1, 4, 4), w=64, h=48
        )
        self.assertEqual(pix_coords.shape, (1, 1, 8, 2))
        self.assertEqual(bboxes2d.shape, (1, 1, 4))
        self.assertEqual(invalid_mask.shape, (1, 1))
