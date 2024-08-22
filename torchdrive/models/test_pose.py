import unittest

import torch

from torchdrive.models.pose import posenet18


class TestPose(unittest.TestCase):
    def test_posenet18(self) -> None:
        m = posenet18(num_images=2, output_channels=5)
        print(m)
        inp = torch.rand(2, 2, 3, 4, 6)

        out = m(inp)
        self.assertEqual(out.shape, (2, 5))
