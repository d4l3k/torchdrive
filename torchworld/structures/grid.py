from dataclasses import dataclass

import torch
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import Transform3d


@dataclass
class Grid3d:
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
        World to voxel space (-1 to 1).
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


@dataclass
class GridImage:
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
