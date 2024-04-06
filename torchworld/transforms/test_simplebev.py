import unittest

import torch

from torchworld.structures.cameras import PerspectiveCameras
from torchworld.structures.grid import Grid3d, GridImage
from torchworld.transforms.simplebev import lift_image_to_3d, merge_grids
from torchworld.transforms.transform3d import Transform3d


class TestSimpleBEV(unittest.TestCase):
    def test_lift_image_to_3d(self) -> None:
        device = torch.device("cpu")
        dtype = torch.half
        dst = Grid3d(
            data=torch.rand(0, 3, 1, 2, 3, device=device, dtype=dtype),
            local_to_world=Transform3d(device=device),
            time=torch.rand(2, device=device),
        )
        src = GridImage(
            data=torch.rand(2, 3, 1, 2, device=device, dtype=dtype),
            camera=PerspectiveCameras(device=device),
            time=torch.rand(2, device=device),
        )

        # compiled_lift = torch.compile(lift_image_to_3d, fullgraph=True, backend="eager")
        out, mask = lift_image_to_3d(src, dst)
        self.assertEqual(out.data.shape, (2, 3, 1, 2, 3))
        self.assertEqual(mask.data.shape, (2, 1, 1, 2, 3))
        self.assertIsInstance(out, Grid3d)
        self.assertIsInstance(mask, Grid3d)

    def test_merge_grids(self) -> None:
        torch.manual_seed(5)

        device = torch.device("cpu")
        dtype = torch.half
        grid = Grid3d(
            data=(torch.rand(2, 3, 1, 2, 3, device=device, dtype=dtype) * 10).round(),
            local_to_world=Transform3d(device=device),
            time=torch.rand(2, device=device),
        )
        self.assertEqual(grid.aminmax(), (0.0, 10.0))

        mask = Grid3d(
            data=torch.rand(2, 1, 1, 2, 3, device=device, dtype=dtype).round(),
            local_to_world=Transform3d(device=device),
            time=torch.rand(2, device=device),
        )
        self.assertEqual(mask.aminmax(), (0.0, 1.0))

        merged_grid, merged_mask = merge_grids([grid, grid], [mask, mask])
        self.assertEqual(merged_grid.shape, (2, 3, 1, 2, 3))
        self.assertEqual(merged_mask.shape, (2, 1, 1, 2, 3))

        self.assertEqual(merged_grid.aminmax(), (0.0, 10.0))
        self.assertEqual(merged_mask.aminmax(), (0.0, 1.0))
