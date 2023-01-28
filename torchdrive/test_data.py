import unittest

from torchdrive.data import Batch, dummy_batch


class TesetData(unittest.TestCase):
    def test_dummy_batch(self) -> None:
        self.assertIsInstance(dummy_batch(), Batch)
