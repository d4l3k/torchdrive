import unittest

import torch
from torch import nn

from torchworld.models.transformer import collect_cross_attention_weights


class TransformerTest(unittest.TestCase):
    def test_collect_cross_attention_weights(self) -> None:
        dim = 8
        transformer_model = nn.Transformer(
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=4,
            d_model=dim,
            dim_feedforward=16,
            batch_first=True,
        )
        src = torch.rand((32, 10, dim))
        tgt = torch.rand((32, 20, dim))
        with collect_cross_attention_weights(transformer_model) as attn_weights:
            out = transformer_model(src, tgt)
        self.assertEqual(len(attn_weights), 4)
        self.assertEqual(attn_weights[0].shape, (32, 20, 10))
