from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Tuple

import torch
from torch import nn

from torchdrive.data import Batch


class BEVBackbone(nn.Module, ABC):
    """
    BEVBackbone is an abstract class for BEV backbones. It consumes multiple
    camera frames and a batch and returns two sets of BEV resolutions. A coarse
    grained grid and a high resolution grid.
    """

    @abstractmethod
    def forward(
        self, camera_features: Mapping[str, List[torch.Tensor]], batch: Batch
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        pass
