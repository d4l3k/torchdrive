import unittest

import torch

from torchworld.transforms.pca import structured_pca


class TestPCA(unittest.TestCase):
    def test_pca(self) -> None:
        A = torch.randn(2, 3, 4, 16)
        out = structured_pca(A, dim=3)
        self.assertEqual(out.shape, (2, 3, 4, 3))
