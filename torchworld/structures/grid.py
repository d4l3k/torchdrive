from typing import Optional, Tuple, TypeVar, Union

import torch
from pytorch3d.structures import Volumes
from pytorch3d.structures.volumes import VolumeLocator

from torchworld.structures.cameras import CamerasBase
from torchworld.transforms.transform3d import Transform3d

T = TypeVar("T")


import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing

aten = torch.ops.aten

MASK_FUNCS = {
    aten.permute.default,
    aten.slice.Tensor,
}


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
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        out = super().__torch_function__(func, types, args, kwargs)
        if isinstance(out, Grid3d):
            if not hasattr(out, "_data"):
                raise TypeError(f"missing required variable in output from {func}")
            assert out._data is not None
        return out

    @classmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        def map_data(x):
            if not hasattr(x, "_data"):
                breakpoint()
            return x._data

        args_data = pytree.tree_map_only(cls, map_data, args)
        kwargs_data = pytree.tree_map_only(cls, map_data, kwargs)

        # get the first grid object from the args
        values, _ = pytree.tree_flatten(args)
        values = [v for v in values if isinstance(v, cls)]
        assert len(values) > 0, f"failed to find {cls} in args"
        grid = values[0]

        out = func(*args_data, **kwargs_data)

        out_flat, spec = pytree.tree_flatten(out)
        out_flat = [
            (
                cls(data=data, local_to_world=grid.local_to_world, time=grid.time)
                if isinstance(data, torch.Tensor)
                else data
            )
            for data in out_flat
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        return return_and_correct_aliasing(func, args, kwargs, out)

    def __tensor_flatten__(self):
        """
        protocol to inform how to flatten to tensor for PT2 tracing
        """
        return ["_data"], (self.local_to_world, self.time)

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, flatten_spec, *args):
        assert (
            flatten_spec is not None
        ), "Expecting spec to be not None from `__tensor_flatten__` return value!"
        data = inner_tensors["_data"]
        return cls(
            data,
            *flatten_spec,
        )

    def __post_init__(self) -> None:
        # if self._data.dim() != 5:
        #    raise TypeError(f"data must be 5 dimensional, got {self._data.shape}")
        if self.time.dim() not in (0, 1):
            raise TypeError(
                f"time must be scalar or 1-dimensional, got {self.time.shape}"
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

    def __repr__(self):
        return f"Grid3d(data={self._data}, local_to_world={self.local_to_world}, time={self.time})"

    def grid_shape(self) -> Tuple[int, int]:
        return tuple(self._data.shape[2:5])

    def numpy(self, **kwargs: object) -> object:
        return self._data.numpy(**kwargs)


class GridImage(torch.Tensor):
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

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        camera: CamerasBase,
        time: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        requires_grad: bool = False,
    ) -> "GridImage":
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
        r.camera = camera
        r.time = time
        r.mask = mask

        r.__post_init__()

        return r

    @classmethod
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        data_args = pytree.tree_map_only(cls, lambda x: x._data, args)
        data_kwargs = pytree.tree_map_only(cls, lambda x: x._data, kwargs)

        # get the first grid object from the args
        values, _ = pytree.tree_flatten(args)
        values = [v for v in values if isinstance(v, cls)]
        assert len(values) > 0, f"failed to find {cls} in args"
        grid = values[0]

        mask = grid.mask
        if mask is not None:
            if func in MASK_FUNCS:
                mask_args = pytree.tree_map_only(cls, lambda x: x.mask, args)
                mask_kwargs = pytree.tree_map_only(cls, lambda x: x.mask, kwargs)
                mask = func(*mask_args, **mask_kwargs)

        time = grid.time
        if func == aten.slice.Tensor:
            _, idx, start, end = args
            if idx == 0:
                time = func(time, idx, start, end)

        out = func(*data_args, **data_kwargs)
        out_flat, spec = pytree.tree_flatten(out)

        out_flat = [
            (
                cls(
                    data=out,
                    camera=grid.camera,
                    time=time,
                    mask=mask,
                )
                if isinstance(out, torch.Tensor)
                else out
            )
            for out in out_flat
        ]
        out = pytree.tree_unflatten(out_flat, spec)
        out = return_and_correct_aliasing(func, args, kwargs, out)
        return out

    def __tensor_flatten__(self):
        """
        protocol to inform how to flatten to tensor for PT2 tracing
        """
        return ["_data"], (self.camera, self.time, self.mask)

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, flatten_spec, *args):
        assert (
            flatten_spec is not None
        ), "Expecting spec to be not None from `__tensor_flatten__` return value!"
        data = inner_tensors["_data"]
        return cls(
            data,
            *flatten_spec,
        )

    def __post_init__(self) -> None:
        # if self._data.dim() != 4:
        #    raise TypeError(f"data must be 4 dimensional, got {self._data.shape}")
        if self.time.dim() not in (0, 1):
            raise TypeError(
                f"time must be scalar or 1-dimensional, got {self.time.shape}"
            )

    def grid_shape(self) -> Tuple[int, int]:
        return tuple(self._data.shape[2:4])

    def __repr__(self):
        return f"GridImage(data={self._data}, camera={self.camera}, time={self.time}), mask={self.mask}"

    def numpy(self, **kwargs: object) -> object:
        return self._data.numpy(**kwargs)
