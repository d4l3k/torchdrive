import unittest


def has_nuscenes() -> bool:
    try:
        import nuscenes  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


HAS_NUSCENES: bool = has_nuscenes()


class TestNuscenes(unittest.TestCase):
    @unittest.skipUnless(HAS_NUSCENES, "requires nuscenes-devkit")
    def test_basic(self) -> None:
        from torchdrive.datasets import nuscenes_dataset  # noqa: F401
