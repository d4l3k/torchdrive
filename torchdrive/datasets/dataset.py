from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional

from torch.utils.data import Dataset as TorchDataset

from torchdrive.data import Batch


class Datasets(str, Enum):
    """
    All the datasets supported by torchdrive.
    """

    RICE = "rice"
    NUSCENES = "nuscenes"


class Dataset(TorchDataset, ABC):
    """
    Base class for datasets used by TorchDrive.
    """

    NAME: Datasets
    cameras: List[str]
    CAMERA_OVERLAP: Dict[str, List[str]]

    @abstractmethod
    def __getitem__(self, idx: int) -> Optional[Batch]:
        ...
