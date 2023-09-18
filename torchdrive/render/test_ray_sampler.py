import unittest

from pytorch3d.renderer.implicit.utils import ray_bundle_to_ray_points

from torchdrive.data import dummy_batch
from torchdrive.render.ray_sampler import LIDARRaySampler


class TestRaySampler(unittest.TestCase):
    def test_lidar_ray_sampler(self) -> None:
        sampler = LIDARRaySampler(n_pts_per_ray=5, min_depth=0.1, max_depth=4.0)
        batch = dummy_batch()

        N_PTS = batch.lidar.size(2)

        ray_bundle, distances = sampler(batch)

        self.assertIsNone(ray_bundle.xys)
        self.assertEqual(ray_bundle.origins.shape, (2, N_PTS, 3))
        self.assertEqual(ray_bundle.directions.shape, (2, N_PTS, 3))
        self.assertEqual(ray_bundle.lengths.shape, (2, N_PTS, 5))
        self.assertEqual(distances.shape, (2, N_PTS))

        out = ray_bundle_to_ray_points(ray_bundle)
        self.assertEqual(out.shape, (2, N_PTS, 5, 3))
