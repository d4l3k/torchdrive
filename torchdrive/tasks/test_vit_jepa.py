import unittest
from typing import Dict, List

from torchdrive.data import Batch, dummy_batch
from torchdrive.tasks.vit_jepa import ViTJEPA

import torch

class TestViTJEPA(unittest.TestCase):
    def test_vit_jepa(self) -> None:
        m = ViTJEPA(
            cameras=["left", "right"],
            num_frames=3,
            num_encode_frames=2,
            cam_shape=(48, 64),
        )
        batch = dummy_batch()
        losses = m(batch, global_step=0)
