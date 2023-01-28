import unittest

import torch

from torchdrive.data import Batch


class TesetData(unittest.TestCase):
    def test_batch(self) -> None:
        b = Batch(
            weight=torch.rand(0),
            distances=torch.rand(5),
            cam_T=torch.rand(5, 4, 4),
            frame_T=torch.rand(5, 4, 4),
            K={},
            T={},
            color={},
            mask={},
        )
