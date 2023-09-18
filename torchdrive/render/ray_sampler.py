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

        self.step = (self.max_depth - self.min_depth) / self.n_pts_per_ray
        assert self.step > 0

    def forward(self, batch: Batch) -> RayBundle:
        T = batch.lidar_to_world()
        BS = batch.batch_size()

        origin = torch.tensor((0, 0, 0, 1.0), device=T.device, dtype=T.dtype).reshape(
            1, 4, 1
        )
        origin = T.matmul(origin)
        origin = origin[:, :3] / origin[:, 3:]

        num_points = batch.lidar.size(2)
        coords = batch.lidar[:, :3]
        ones = torch.ones(BS, 1, num_points)
        coords = torch.cat((coords, ones), dim=1)
        coords = T.matmul(coords)
        coords = coords[:, :3] / coords[:, 3:]

        directions = coords - origin
        directions = F.normalize(directions, dim=1)

        lengths = torch.arange(
            start=self.min_depth,
            end=self.max_depth,
            step=self.step,
            device=T.device,
            dtype=T.dtype,
        )

        return RayBundle(
            origins=origin,
            directions=directions,
            lengths=lengths,
            xys=None,
        )
