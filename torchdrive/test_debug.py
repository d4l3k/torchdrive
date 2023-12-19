import unittest

import torch

from torchdrive.debug import assert_not_nan, assert_not_nan_dict, is_nan


class DebugTest(unittest.TestCase):
    def test_is_nan(self) -> None:
        self.assertTrue(is_nan(torch.rand(1, 2, 3) * torch.nan))
        self.assertFalse(is_nan(torch.rand(1, 2, 3)))

    def test_assert_not_nan(self) -> None:
        assert_not_nan(torch.rand(1, 2, 3))
        with self.assertRaisesRegex(AssertionError, ".*is NaN:.*"):
            assert_not_nan(torch.rand(1, 2, 3) * torch.nan)

    def test_assert_not_nan_dict(self) -> None:
        assert_not_nan_dict({"a": torch.rand(1, 2, 3)})
        with self.assertRaisesRegex(AssertionError, ".*is NaN: testkey"):
            assert_not_nan_dict({"testkey": torch.rand(1, 2, 3) * torch.nan})
