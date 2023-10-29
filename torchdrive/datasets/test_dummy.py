import unittest

from torchdrive.data import Batch
from torchdrive.datasets.dummy import DummyDataset


class TestDummyDataset(unittest.TestCase):
    def test_sanity(self) -> None:
        dataset = DummyDataset()
        self.assertEqual(len(dataset), 10)
        # pyre-fixme[16]: `DummyDataset` has no attribute `__iter__`.
        for item in dataset:
            self.assertIsInstance(item, Batch)
