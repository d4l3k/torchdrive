from dataclasses import dataclass

import torch


@dataclass
class Points:
    """Points represents a set of points in world coordinates.

    Attributes
    ----------
    data: [bs, 3+, num_points]
        World coordinates with metadata [x, y, z, ... (intensity?)]
    """

    data: torch.Tensor
