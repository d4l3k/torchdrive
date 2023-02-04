import unittest
from dataclasses import replace

import torch

from torchdrive.data import Batch, collate, dummy_batch, nonstrict_collate


class TesetData(unittest.TestCase):
    def test_dummy_batch(self) -> None:
        self.assertIsInstance(dummy_batch(), Batch)

    def test_collate(self) -> None:
        batch = collate([dummy_batch(), dummy_batch(), None])

    def test_collate_long_cam_T(self) -> None:
        a = dummy_batch()
        b = dummy_batch()
        a = replace(a, long_cam_T=torch.rand(3, 4, 4))
        b = replace(b, long_cam_T=torch.rand(5, 4, 4))
        batch = collate([a, b])
        self.assertIsNotNone(batch)
        self.assertEqual(batch.long_cam_T.shape, (2, 3, 4, 4))

    def test_nonstrict_colate(self) -> None:
        self.assertIsNone(nonstrict_collate([None]))
        with self.assertRaisesRegex(RuntimeError, "not enough data in batch"):
            collate([None])

    def test_batch_to(self) -> None:
        device = torch.device("cpu")
        batch = dummy_batch().to(device)
        self.assertEqual(batch.distances.device, device)
