from typing import Optional

import torch
from matplotlib import cm


@torch.no_grad()
def normalize_img_cuda(src: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the provided image range to lows P0.1 and highs P99 and returns
    the tensor.
    """
    has_bs = len(src.shape) == 4
    bs = len(src) if has_bs else 1
    src = src.detach()
    # q = 0.999
    flat = src.view((bs, 3, -1))
    quantiles = torch.quantile(
        flat, torch.tensor((0.001, 0.99), device=src.device), dim=2
    )
    max = quantiles[1].view(bs, 3, 1, 1)
    min = quantiles[0].view(bs, 3, 1, 1)
    new = (src - min).div_(max - min)
    new = new.clamp_(0, 1)
    if not has_bs:
        return new.squeeze(0)
    return new


@torch.no_grad()
def normalize_img(src: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the provided image and returns a CPU tensor.
    """
    return normalize_img_cuda(src).cpu()


@torch.no_grad()
def render_color(
    img: torch.Tensor,
    max: Optional[float] = None,
    min: Optional[float] = None,
    palette: str = "magma",
) -> torch.Tensor:
    """
    Renders an array into colors with the specified palette.

    Args:
        img: input tensor [H, W], float
    Returns:
        output tensor [3, H, W], float, cpu
    """
    img = img.detach()
    cmap = cm.get_cmap(palette)
    N = 1000
    colors = torch.tensor([cmap(i / N)[:3] for i in range(N)], device=img.device)

    if min is None:
        min = img.min()
    if max is None:
        max = img.max()

    if max == min:
        img = torch.zeros(img.shape)
    else:
        img = (img - min) / (max - min) * (N - 1)
    mapped = colors[img.long()]
    if len(mapped.shape) != 3:
        print(mapped.shape)
    return mapped.permute(2, 0, 1).cpu()
