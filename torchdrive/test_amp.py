import unittest

from torchdrive.amp import autocast


class TestAMP(unittest.TestCase):
    def test_autocast(self) -> None:
        with autocast():
            pass
