import unittest

import torch

from torchdrive.notebook import display_color, display_img


class TestNotebook(unittest.TestCase):
    def test_display_img(self) -> None:
        display_img(torch.rand(3, 4, 5))

    def test_display_color(self) -> None:
        display_color(torch.rand(4, 5))
