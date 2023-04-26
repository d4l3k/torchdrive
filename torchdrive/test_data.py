import unittest
from dataclasses import replace

import torch
from torch.utils.data import DataLoader, Dataset

from torchdrive.data import (
    Batch,
    collate,
    dummy_batch,
    dummy_item,
    nonstrict_collate,
    TransferCollator,
)


class DummyDataset(Dataset[Batch]):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, i: int) -> Batch:
        return dummy_item()


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
        self.assertEqual(a.global_batch_size, 2)
        self.assertEqual(a.batch_size(), 1)

    def test_weights(self) -> None:
        batch = dummy_batch()
        torch.testing.assert_close(batch.weight.sum(), torch.tensor(1.0))
        a, b = batch.split(1)
        torch.testing.assert_close(a.weight.sum(), torch.tensor(0.5))

    def test_transfer_collator(self) -> None:
        dataset = DummyDataset(5)
        dataloader = DataLoader(dataset, batch_size=None)
        device = torch.device("cpu")
        batch_size = 2
        collator = TransferCollator(dataloader, batch_size=batch_size, device=device)

        self.assertEqual(len(collator), 2)

        out = list(collator)
        self.assertEqual(len(out), 2)
        for batch in out:
            self.assertEqual(batch.batch_size(), batch_size)

    def test_world_to_car(self) -> None:
        batch = dummy_batch()
        out = batch.world_to_car(1)
        self.assertEqual(out.shape, (2, 4, 4))
        torch.testing.assert_allclose(out, batch.cam_T[:, 1].pinverse())

    def test_car_to_world(self) -> None:
        batch = dummy_batch()
        out = batch.car_to_world(1)
        self.assertEqual(out.shape, (2, 4, 4))
        torch.testing.assert_allclose(out, batch.cam_T[:, 1])

    def test_world_to_cam(self) -> None:
        batch = dummy_batch()
        cam = "left"
        frame = 1
        target = batch.T[cam].pinverse().matmul(batch.cam_T[:, frame].pinverse())
        out = batch.world_to_cam(cam, frame)
        self.assertEqual(out.shape, (2, 4, 4))
        torch.testing.assert_allclose(out, target)

    def test_cam_to_world(self) -> None:
        batch = dummy_batch()
        cam = "left"
        frame = 1
        target = (
            batch.T[cam].pinverse().matmul(batch.cam_T[:, frame].pinverse()).pinverse()
        )
        out = batch.cam_to_world(cam, frame)
        self.assertEqual(out.shape, (2, 4, 4))
        torch.testing.assert_allclose(out, target)
