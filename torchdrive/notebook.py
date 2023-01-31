"""
This contains helper methods for visualizing data in Jupyter notebooks.
"""
import torch
from IPython.display import display
from torchvision.transforms.functional import to_pil_image

from torchdrive.transforms.img import normalize_img, render_color


def display_img(x: torch.Tensor) -> None:
    """
    Normalizes the provided [3, w, h] image and displays it.
    """
    display(to_pil_image(normalize_img(x)))


def display_color(x: torch.Tensor) -> None:
    """
    Renders a [w, h] grid into colors and displays it.
    """
    display(to_pil_image(render_color(x)))
