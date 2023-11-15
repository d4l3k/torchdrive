from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING, TypeVar, Union

import torch
from pytorch3d.structures.volumes import VolumeLocator

from torchworld.structures.cameras import CamerasBase
from torchworld.transforms.transform3d import Transform3d

if TYPE_CHECKING:
    from typing import Self

T = TypeVar("T")


@dataclass
class BaseGrid(ABC):
    data: torch.Tensor
    time: torch.Tensor

    @abstractmethod
    def to(self, target: Union[torch.device, str]) -> "Self":
        ...

    def cuda(self) -> "Self":
        return self.to(torch.device("cuda"))

    def cpu(self) -> "Self":
        return self.to(torch.device("cpu"))

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    @abstractmethod
    def grid_shape(self) -> Tuple[int, ...]:
        ...


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
    local_to_world:
        The 3d transform option that locates the Grid3d in space
        Voxel (-1 to 1) to world space.
    time: scalar or [bs]
        Time corresponding to the grid
    """

    data: torch.Tensor
    local_to_world: Transform3d
    time: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.dim() != 5:
            raise TypeError(f"data must be 5 dimensional, got {self.data.shape}")
        if self.time.dim() not in (0, 1):
            raise TypeError(
                f"time must be scalar or 1-dimensional, got {self.time.shape}"
            )

        T = self.local_to_world.get_matrix()
        if (BS := T.size(0)) != 1:
            if BS != self.data.size(0):
                raise TypeError(
                    f"data and local_to_world batch sizes don't match: {T.shape, self.data.shape}"
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
        grid_sizes = tuple(data.shape[2:5])
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
            local_to_world=Transform3d(
                matrix=locator.get_local_to_world_coords_transform().get_matrix()
            ),
            time=time,
        )

    def to(self, target: Union[torch.device, str]) -> "Grid3d":
        return Grid3d(
            data=self.data.to(target),
            local_to_world=self.local_to_world.to(target),
            time=self.time.to(target),
        )

    def grid_shape(self) -> Tuple[int, int]:
        return tuple(self.data.shape[2:5])

    def replace(
        self,
        data: Optional[torch.Tensor] = None,
        local_to_world: Optional[Transform3d] = None,
        time: Optional[torch.Tensor] = None,
    ) -> "Grid3d":
        return Grid3d(
            data=data if data is not None else self.data,
            local_to_world=local_to_world
            if local_to_world is not None
            else self.local_to_world,
            time=time if time is not None else self.time,
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
    mask: [bs, 1, y, x]
        Optional mask for the features.
    """

    data: torch.Tensor
    camera: CamerasBase
    time: torch.Tensor
    mask: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.data.dim() != 4:
            raise TypeError(f"data must be 4 dimensional, got {self.data.shape}")
        if self.time.dim() not in (0, 1):
            raise TypeError(
                f"time must be scalar or 1-dimensional, got {self.time.shape}"
            )

        if (mask := self.mask) is not None:
            if self.data.shape[2:] != mask.shape[2:]:
                raise TypeError(
                    f"mask is not the same shape as data {self.data.shape} {mask.shape}"
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
            # pyre-fixme[6]: For 2nd argument expected `CamerasBase` but got
            #  `TensorProperties`.
            camera=self.camera.to(target),
            time=self.time.to(target),
        )

    def grid_shape(self) -> Tuple[int, int]:
        return tuple(self.data.shape[2:4])

    def replace(
        self,
        data: Optional[torch.Tensor] = None,
        camera: Optional[CamerasBase] = None,
        time: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> "GridImage":
        return GridImage(
            data=data if data is not None else self.data,
            camera=camera if camera is not None else self.camera,
            time=time if time is not None else self.time,
            mask=mask if mask is not None else self.mask,
        )
