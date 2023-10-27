from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, TypeVar, Union

import torch
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures.volumes import VolumeLocator
from pytorch3d.transforms import Transform3d

T = TypeVar("T")


@dataclass
class BaseGrid(ABC):
    @abstractmethod
    def to(self: T, target: Union[torch.device, str]) -> T:
        ...

    def cuda(self) -> T:
        return self.to(torch.device("cuda"))

    def cpu(self) -> T:
        return self.to(torch.device("cpu"))


@dataclass
class Grid3d(BaseGrid):
    """Grid3d represents a 3D grid of features in a certain location and time in
    world space.

    Birdseye view grids can be represented as a special case of Grid3d with the
    Z dimension only having a single element with an infinite height.

    Attributes
    ----------
    data: [bs, channels, z, y, x]
        The grid of features.
    transform:
        The 3d transform option that locates the Grid3d in space
        Voxel (-1 to 1) to world space.
    time: scalar or [bs]
        Time corresponding to the grid
    """

    data: torch.Tensor
    transform: Transform3d
    time: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.dim() != 5:
            raise TypeError(f"data must be 5 dimensional, got {self.data.shape}")
        if self.time.dim() not in (0, 1):
            raise TypeError(
                f"time must be scalar or 1-dimensional, got {self.time.shape}"
            )

        T = self.transform.get_matrix()
        if (BS := T.size(0)) != 1:
            if BS != self.data.size(0):
                raise TypeError(
                    f"data and transform batch sizes don't match: {T.shape, self.data.shape}"
                )

    @classmethod
    def from_volume(
        cls,
        data: torch.Tensor,
        voxel_size: Union[float, torch.Tensor] = 1.0,
        volume_translation: Union[Tuple[float, float, float], torch.Tensor] = (
            0.0,
            0.0,
            0.0,
        ),
        time: Union[float, torch.Tensor] = 0.0,
    ) -> "Grid3d":
        """from_volume creates a new grid using a similar interface to
        pytorch3d's Volume.

        Parameters
        ----------
        data: [bs, channels, z, y, x]
            The grid features.
        voxel_size:
            The size of the voxels.
        volume_translation:
            The center of the volume in world coordinates.
        """
        device = data.device
        grid_sizes = data.shape[2:5]
        locator = VolumeLocator(
            batch_size=len(data),
            grid_sizes=grid_sizes,
            voxel_size=voxel_size,
            volume_translation=volume_translation,
            device=device,
        )
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, dtype=torch.float, device=device)
        return cls(
            data=data,
            transform=locator.get_local_to_world_coords_transform(),
            time=time,
        )

    def to(self, target: Union[torch.device, str]) -> "Grid3d":
        return Grid3d(
            data=self.data.to(target),
            transform=self.transform.to(target),
            time=self.time.to(target),
        )


@dataclass
class GridImage(BaseGrid):
    """GridImage represents a 2D grid of features corresponding to certain
    camera (and thus location) and time.

    Attributes
    ----------
    data: [bs, channels, y, x]
        Grid of features.
    camera:
        camera corresponding to the image space features.
    time: [bs]
        Time corresponding to the grid.
    """

    data: torch.Tensor
    camera: CamerasBase
    time: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.dim() != 4:
            raise TypeError(f"data must be 4 dimensional, got {data.shape}")
        if self.time.dim() not in (0, 1):
            raise TypeError(
                f"time must be scalar or 1-dimensional, got {self.time.shape}"
            )

        T = self.camera.get_projection_transform().get_matrix()
        if (BS := T.size(0)) != 1:
            if BS != self.data.size(0):
                raise TypeError(
                    f"data and transform batch sizes don't match: {T.shape, self.data.shape}"
                )

    def to(self, target: Union[torch.device, str]) -> "GridImage":
        return GridImage(
            data=self.data.to(target),
            camera=self.camera.to(target),
            time=self.time.to(target),
        )
