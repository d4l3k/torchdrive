"""
This contains helper methods for visualizing data in Jupyter notebooks.
"""
import torch
from IPython.display import display
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

from torchworld.transforms.img import normalize_img, render_color


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


def display_bboxes(img: torch.Tensor, bboxes: object, threshold: float = 0.5) -> None:
    """
    Displays bounding boxes in mmcv format on the provided image.
    """
    tboxes = []
    labels = []
    for i, box in enumerate(bboxes):
        if len(box) == 0:
            continue
        if not isinstance(box, torch.Tensor):
            box = torch.from_numpy(box)
        p = box[:, 4]
        valid = p > threshold
        box = box[valid]
        tboxes.append(box[:, :4])
        labels += [str(i)] * len(box)

    tboxes = torch.cat(tboxes, dim=0)

    img = normalize_img(img.float())
    img = (img.clamp(min=0, max=1) * 255).byte()

    display(to_pil_image(draw_bounding_boxes(image=img, boxes=tboxes, labels=labels)))
