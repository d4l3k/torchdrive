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

import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing

class Grid3d(torch.Tensor):
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

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        local_to_world: torch.Tensor,
        time: torch.Tensor,
        *,
        requires_grad: bool = False,
    ) -> "Grid3d":
        # new method instruct wrapper tensor from local_tensor and add
        # placement spec, it does not do actual distribution
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            data.shape,
            strides=data.stride(),
            dtype=data.dtype,
            device=data.device,
            layout=data.layout,
            requires_grad=requires_grad,
        )

        r._data = data
        r.local_to_world = local_to_world
        r.time = time

        r.__post_init__()

        return r

    @classmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args_data = pytree.tree_map_only(Grid3d, lambda x: x._data, args)
        local_to_world = pytree.tree_map_only(Grid3d, lambda x: x.local_to_world, args)
        local_to_world_flat = pytree.tree_leaves(local_to_world)
        time = pytree.tree_map_only(Grid3d, lambda x: x.time, args)
        time_flat = pytree.tree_leaves(time)
        kwargs_data = pytree.tree_map_only(Grid3d, lambda x: x._data, kwargs)

        out = func(*args_data, **kwargs_data)

        out_flat, spec = pytree.tree_flatten(out)
        out_flat = [
            Grid3d(data, l2w, time)
            for data, l2w, time in zip(out_flat, local_to_world_flat, time_flat)
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        return return_and_correct_aliasing(func, args, kwargs, out)


    def __post_init__(self) -> None:
        if self._data.dim() != 5:
            raise TypeError(f"data must be 5 dimensional, got {self.data.shape}")
        if self.time.dim() not in (0, 1):
            raise TypeError(
                f"time must be scalar or 1-dimensional, got {self.time.shape}"
            )

        T = self.local_to_world.get_matrix()
        if (BS := T.size(0)) != 1:
            if BS != self._data.size(0):
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
            data=self._data.to(target),
            local_to_world=self.local_to_world.to(target),
            time=self.time.to(target),
        )

    def grid_shape(self) -> Tuple[int, int]:
        return tuple(self._data.shape[2:5])


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
