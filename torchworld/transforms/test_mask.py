import unittest

import torch

from torchworld.transforms.mask import random_block_mask


class TestMask(unittest.TestCase):
    def test_random_block_mask(self) -> None:
        t = torch.rand(60, 80)
        mask = random_block_mask(t, block_size=(20, 20), num_blocks=8)
        self.assertEqual(t.shape, mask.shape)
