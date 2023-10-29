from typing import Tuple

import torch
import torch.nn.functional as F
from pytorch3d.renderer.implicit.raysampling import RayBundle

from torchdrive.data import Batch


class LIDARRaySampler(torch.nn.Module):
    """
    LIDARRaySampler is a raysampler that consumes a Batch and returns a
    RayBundle with rays that match the directions of the LIDAR data contained in
    that batch.

    Intended for use in evaluating occupancy grids from LIDAR ground truth.
    """

    def __init__(self, n_pts_per_ray: int, min_depth: float, max_depth: float) -> None:
        super().__init__()

        self.n_pts_per_ray = n_pts_per_ray
        self.min_depth = min_depth
        self.max_depth = max_depth

        # pyre-fixme[4]: Attribute must be annotated.
        self.step = (self.max_depth - self.min_depth) / self.n_pts_per_ray
        assert self.step > 0

    def forward(self, batch: Batch) -> Tuple[RayBundle, torch.Tensor]:
        """
        Returns:
            RayBundle
            Tensor of depths corresponding to each LIDAR point.
        """
        T = batch.lidar_to_world()
        BS = batch.batch_size()
        # pyre-fixme[16]: Optional type has no attribute `size`.
        num_points = batch.lidar.size(2)

        origin = torch.tensor((0, 0, 0, 1.0), device=T.device, dtype=T.dtype).reshape(
            1, 4, 1
        )
        origin = T.matmul(origin)
        origin = origin[:, :3] / origin[:, 3:]
        origin = origin.permute(0, 2, 1).expand(-1, num_points, -1)

        # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
        coords = batch.lidar[:, :3]
        ones = torch.ones(BS, 1, num_points, dtype=T.dtype, device=T.device)
        coords = torch.cat((coords, ones), dim=1)
        coords = T.matmul(coords)
        coords = coords[:, :3] / coords[:, 3:]
        coords = coords.permute(0, 2, 1)

        directions = coords - origin
        distances = torch.linalg.vector_norm(directions, dim=2)
        directions = F.normalize(directions, dim=2)

        lengths = torch.arange(
            start=self.min_depth,
            end=self.max_depth,
            step=self.step,
            device=T.device,
            dtype=T.dtype,
        ).expand(BS, num_points, -1)

        return (
            RayBundle(
                origins=origin,
                directions=directions,
                lengths=lengths,
                # pyre-fixme[6]: For 4th argument expected `Tensor` but got `None`.
                xys=None,
            ),
            distances,
        )
