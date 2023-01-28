import unittest

import torch

from torchdrive.tasks.det import DetTask

class TestDet(unittest.TestCase):
    def test_det_task(self) -> None:
        m = DetTask(cameras=["left", "right"], cam_shape=(4, 6),
                    bev_shape=(4,4), dim=5, device=torch.device("cpu"))

