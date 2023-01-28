import unittest

import torch

from torchdrive.data import Batch, collate, dummy_batch, nonstrict_collate


class TesetData(unittest.TestCase):
    def test_dummy_batch(self) -> None:
        self.assertIsInstance(dummy_batch(), Batch)

    def test_collate(self) -> None:
        batch = collate([dummy_batch(), dummy_batch(), None])

    def test_nonstrict_colate(self) -> None:
        self.assertIsNone(nonstrict_collate([None]))
        with self.assertRaisesRegex(RuntimeError, "not enough data in batch"):
            collate([None])

    def test_batch_to(self) -> None:
        device = torch.device("cpu")
        batch = dummy_batch().to(device)
        self.assertEqual(batch.distances.device, device)
