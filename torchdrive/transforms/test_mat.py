import unittest

import torch

from torchdrive.transforms.mat import random_z_rotation, transformation_from_parameters


class TestMat(unittest.TestCase):
    def test_transformation_from_parameters(self) -> None:
        out = transformation_from_parameters(torch.rand(2, 1, 3), torch.rand(2, 1, 3))
        self.assertEqual(out.shape, (2, 4, 4))

    def test_random_z_rotation(self) -> None:
        out = random_z_rotation(2, torch.device("cpu"))
        self.assertEqual(out.shape, (2, 4, 4))
