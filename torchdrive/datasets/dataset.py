from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from strenum import StrEnum

from torch.utils.data import Dataset as TorchDataset

from torchdrive.data import Batch


class Datasets(StrEnum):
    """
    All the datasets supported by torchdrive.
    """

    RICE = "rice"
    NUSCENES = "nuscenes"
    DUMMY = "dummy"


class Dataset(TorchDataset, ABC):
    """
    Base class for datasets used by TorchDrive.
    """

    @property
    @abstractmethod
    def NAME(self) -> Datasets: ...

    @property
    @abstractmethod
    def cameras(self) -> List[str]: ...

    @property
    @abstractmethod
    def CAMERA_OVERLAP(self) -> Dict[str, List[str]]: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Optional[Batch]: ...
