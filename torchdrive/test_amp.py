import unittest

from torchdrive.amp import autocast
from torchdrive.testing import skipIfNoCUDA


class TestAMP(unittest.TestCase):
    @skipIfNoCUDA()
    def test_autocast(self) -> None:
        with autocast():
            pass
