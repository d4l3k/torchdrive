from dataclasses import dataclass

import torch

@dataclass
class Lidar:
    """Lidar represents a set of Lidar points.

    Attributes
    ----------
    data: [bs, 4, num_points]
        Lidar coordinates [x, y, z, intensity]
    """
    data: torch.Tensor
