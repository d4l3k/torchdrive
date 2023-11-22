import unittest

import torch

from torchworld.models.deformable_transformer import (
    collect_sampling_locations,
    DeformableTransformer,
)


class TestDeformableTransformer(unittest.TestCase):
    def test_transformer(self) -> None:
        BS = 2
        C = 3
        H = 4
        W = 5
        N = 6

        m = DeformableTransformer(
            d_model=C,
            dim_feedforward=24,
            num_feature_levels=1,
            nhead=1,
            num_encoder_layers=2,
            num_decoder_layers=2,
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

    def test_collect_sampling_locations(self) -> None:
        BS = 2
        C = 3
        H = 4
        W = 5
        N = 6

        m = DeformableTransformer(
            d_model=C,
            dim_feedforward=24,
            num_feature_levels=1,
            nhead=1,
            num_encoder_layers=2,
            num_decoder_layers=2,
        )
        srcs = [
            torch.rand(BS, C, H, W),
        ]
        masks = [torch.ones(BS, H, W, dtype=torch.bool)]
        pos_embeds = [
            torch.rand(BS, C, H, W),
        ]
        query_embed = torch.rand(BS, 2 * C)

        with collect_sampling_locations(m) as sampling_locations:
            m(srcs, masks, pos_embeds, query_embed)

        self.assertEqual(len(sampling_locations), 4)
        ref, samp = sampling_locations[-1]
        self.assertEqual(ref.shape, (2, 2, 1, 2))
        self.assertEqual(samp.shape, (2, 2, 1, 1, 4, 2))

    def test_dynamic_reference_points(self) -> None:
        BS = 2
        C = 3
        H = 4
        W = 5
        N = 6

        m = DeformableTransformer(
            d_model=C,
            dim_feedforward=24,
            num_feature_levels=2,
            nhead=1,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dynamic_reference_points=True,
        )
        srcs = [
            torch.rand(BS, C, H, W),
            torch.rand(BS, C, H * 2, W * 2),
        ]
        masks = [
            torch.ones(BS, H, W, dtype=torch.bool),
            torch.ones(BS, H * 2, W * 2, dtype=torch.bool),
        ]
        pos_embeds = [
            torch.rand(BS, C, H, W),
            torch.rand(BS, C, H * 2, W * 2),
        ]
        query_embed = torch.rand(BS, 2 * C)

        with collect_sampling_locations(m) as sampling_locations:
            m(srcs, masks, pos_embeds, query_embed)

        self.assertEqual(len(sampling_locations), 4)
        ref, samp = sampling_locations[-1]
        self.assertEqual(ref.shape, (2, 2, 2, 2))
        self.assertEqual(samp.shape, (2, 2, 1, 2, 4, 2))
        self.assertIsNotNone(m.decoder.layers[0].project_reference_points)
