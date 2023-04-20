import unittest

import torch

from torchdrive.transforms.img import (
    normalize_img,
    normalize_img_cuda,
    normalize_mask,
    render_color,
)


class TestDet(unittest.TestCase):
    def test_normalize_img(self) -> None:
        out = normalize_img(torch.rand(3, 48, 64))
        self.assertEqual(out.shape, (3, 48, 64))

    def test_normalize_img_multiple_batches(self) -> None:
        out = normalize_img(torch.rand(3, 2, 48, 64))
        self.assertEqual(out.shape, (3, 2, 48, 64))

    def test_normalize_img_cuda(self) -> None:
        out = normalize_img_cuda(torch.rand(3, 48, 64))
        self.assertEqual(out.shape, (3, 48, 64))

    def test_render_color(self) -> None:
        out = render_color(torch.rand(48, 64))
        self.assertEqual(out.shape, (3, 48, 64))

    def test_normalize_mask(self) -> None:
        out = normalize_mask(torch.rand(2, 3, 48, 64), torch.rand(2, 1, 48, 64))
        self.assertEqual(out.shape, (2, 3, 48, 64))

        out = normalize_mask(torch.rand(2, 3, 48, 64), torch.zeros(2, 1, 48, 64))
        self.assertEqual(out.shape, (2, 3, 48, 64))
