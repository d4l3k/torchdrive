import unittest
from typing import Dict, List
from unittest.mock import MagicMock, patch

import torch

from torchdrive.data import Batch, dummy_batch
from torchdrive.tasks.vit_jepa import ViTJEPA


class TestViTJEPA(unittest.TestCase):
    def test_vit_jepa(self) -> None:
        m = ViTJEPA(
            cameras=["left", "right"],
            num_frames=3,
            num_encode_frames=2,
            cam_shape=(48, 64),
            num_layers=2,
            dim_feedforward=16,
        )
        m.param_opts(lr=1e-4)
        batch = dummy_batch()
        writer = MagicMock()
        losses = m(batch, global_step=0, writer=writer)
        self.assertIsNotNone(m.encoders["left"].encoder.encoder.pos_embedding.grad)
