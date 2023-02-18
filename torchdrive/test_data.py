import unittest
from dataclasses import replace

import torch

from torchdrive.data import Batch, collate, dummy_batch, dummy_item, nonstrict_collate


class TestData(unittest.TestCase):
    def test_dummy_batch(self) -> None:
        self.assertIsInstance(dummy_batch(), Batch)

    def test_collate(self) -> None:
        batch = collate([dummy_item(), dummy_item(), None])

    def test_collate_long_cam_T(self) -> None:
        a = dummy_batch()
        b = dummy_batch()
        a = replace(a, long_cam_T=torch.rand(3, 4, 4))
        b = replace(b, long_cam_T=torch.rand(5, 4, 4))
        batch = collate([a, b])
        self.assertIsNotNone(batch)

        long_cam_T, mask, lengths = batch.long_cam_T
        self.assertEqual(long_cam_T.shape, (2, 5, 4, 4))
        self.assertEqual(mask.shape, (2, 5))
        self.assertEqual(lengths.tolist(), [3, 5])
        self.assertEqual(mask[0].tolist(), [True, True, True, False, False])
        self.assertEqual(long_cam_T[0, -1, 0, 0], 0)

    def test_nonstrict_collate(self) -> None:
        self.assertIsNone(nonstrict_collate([None]))
        with self.assertRaisesRegex(RuntimeError, "not enough data in batch"):
            collate([None])

    def test_batch_to(self) -> None:
        device = torch.device("cpu")
        batch = dummy_batch().to(device)
        self.assertEqual(batch.distances.device, device)

    def test_split(self) -> None:
        out = dummy_batch()
        self.assertEqual(len(out.split(1)), 2)
        self.assertEqual(len(out.split(2)), 1)
        self.assertEqual(len(out.split(3)), 1)

    def test_size(self) -> None:
        out = dummy_item()
        self.assertEqual(out.batch_size(), 1)

        out = dummy_batch()
        self.assertEqual(out.batch_size(), 2)

    def test_global_batch_size(self) -> None:
        self.assertEqual(dummy_item().global_batch_size, 1)
        batch = dummy_batch()
        self.assertEqual(batch.global_batch_size, 2)
        a, b = batch.split(1)
        print(a, b)
        self.assertEqual(a.global_batch_size, 2)
        self.assertEqual(a.batch_size(), 1)
