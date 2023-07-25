import unittest

def has_nuscenes() -> bool:
    try:
        import nuscenes
        return True
    except ModuleNotFoundError:
        return False

class TestNuscenes(unittest.TestCase):
    @unittest.skipUnless(has_nuscenes(), "requires nuscenes-devkit")
    def test_basic(self) -> None:
        from torchdrive.datasets import nuscenes_dataset
