from typing import Tuple

import torch

def random_block_mask(t: torch.Tensor, block_size: Tuple[int, int], num_blocks: int) -> torch.Tensor:
    """
    Randomly masks a tensor with blocks of zeros.

    True is where it is not masked.

    Returns
    -------
    mask : torch.Tensor
        A tensor of shape (h, w) where True is where it is not masked.
    """
    h, w = t.shape[-2:]
    mask = torch.ones(t.shape[-2:], dtype=torch.bool, device=t.device)
    mh, mw = block_size
    for _ in range(num_blocks):
        x = torch.randint(w, tuple())
        y = torch.randint(h, tuple())
        mask[max(y-mh//2, 0):y + mh//2, max(x-mw//2, 0):x + mw//2] = False
    return mask
