import unittest

from torchdrive.datasets.rice import compute_bin


class TestRice(unittest.TestCase):
    def test_compute_bin(self) -> None:
        self.assertEqual(compute_bin(5, [0, 2, 4, 6, 8]), 6)
