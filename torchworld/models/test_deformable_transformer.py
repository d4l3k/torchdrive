import unittest

import torch

from torchworld.models.deformable_transformer import DeformableTransformer


class TestDeformableTransformer(unittest.TestCase):
    def test_transformer(self) -> None:
        BS = 2
        C = 3
        H = 4
        W = 5
        N = 6

        m = DeformableTransformer(
            d_model=C, dim_feedforward=24, num_feature_levels=1, nhead=1
        )
        srcs = [
            torch.rand(BS, C, H, W),
        ]
        masks = [torch.ones(BS, H, W, dtype=torch.bool)]
        pos_embeds = [
            torch.rand(BS, C, H, W),
        ]
        query_embed = torch.rand(BS, 2 * C)
        m(srcs, masks, pos_embeds, query_embed)
