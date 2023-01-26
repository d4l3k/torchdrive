import unittest

import torch

from torchdrive.transforms.img import normalize_img, normalize_img_cuda, render_color


class TestDet(unittest.TestCase):
    def test_normalize_img(self) -> None:
        out = normalize_img(torch.rand(3, 48, 64))
        self.assertEqual(out.shape, (3, 48, 64))

    def test_normalize_img_cuda(self) -> None:
        out = normalize_img_cuda(torch.rand(3, 48, 64))
        self.assertEqual(out.shape, (3, 48, 64))

    def test_render_color(self) -> None:
        out = render_color(torch.rand(48, 64))
        self.assertEqual(out.shape, (3, 48, 64))
