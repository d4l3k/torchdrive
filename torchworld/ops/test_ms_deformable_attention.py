import unittest

import torch

from torchworld.ops.ms_deformable_attention import MSDeformableAttention2d


class TestMSDeformableAttention(unittest.TestCase):
    def test_ms_deform_attention2d(self) -> None:
        C = 16
        H = 4
        W = 5
        N = 2
        Q = 3

        m = MSDeformableAttention2d(d_model=C, n_levels=1, n_heads=1, n_points=4)
        query = torch.rand(2, Q, C)
        reference_points = torch.rand(2, Q, 1, 2)
        input_flatten = torch.rand(2, H * W, C)
        input_spatial_shapes = torch.tensor(((H, W),))
        input_level_start_index = torch.tensor((0,))
        out = m(
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
        )
        self.assertEqual(out.shape, (N, Q, C))
