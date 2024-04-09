from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

import torch
import torchvision
from torch import nn

from torchdrive.amp import autocast
from torchdrive.data import Batch
from torchdrive.models.bev_backbone import BEVBackbone

from torchworld.models.fpn import Resnet18FPN3d
from torchworld.structures.grid import Grid3d, GridImage
from torchworld.transforms.simplebev import lift_image_to_3d, merge_grids


class SimpleBEV3DBackbone(BEVBackbone):
    """
    A BEV backbone using the Simplebev/Segnet projections and a custom 3D convolution backbone.

    This extends Simple BEV to support multiple time frames.
    """

    def __init__(
        self,
        cam_dim: int,
        dim: int,
        hr_dim: int,
        num_frames: int,
        grid_shape: Tuple[int, int, int],
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.grid_shape = grid_shape
        self.num_frames = num_frames

        Z = grid_shape[2]

        self.project_cam_time = nn.ModuleList(
            [compile_fn(nn.Conv2d(cam_dim, cam_dim, 1)) for i in range(num_frames)]
        )
        self.fpn: nn.Module = compile_fn(Resnet18FPN3d(cam_dim))

        self.bev_project = nn.ModuleList(
            [
                nn.Conv2d(cam_dim * Z, dim, 1),
                nn.Conv2d(cam_dim * Z, dim, 1),
                nn.Conv2d(cam_dim * Z, dim, 1),
                nn.Conv2d(cam_dim * Z, dim, 1),
            ]
        )

    def forward(
        self,
        batch: Batch,
        camera_features: Mapping[str, List[GridImage]],
        target_grid: Grid3d,
    ) -> Tuple[Grid3d, List[Grid3d], Dict[str, torch.Tensor]]:
        BS = batch.batch_size()
        S = len(camera_features) * self.num_frames
        device = batch.device()

        Z = self.grid_shape[2]

        features = []
        masks = []

        # camera order doesn't matter for segnet
        for cam, time_feats in camera_features.items():
            for i, cam_feats in enumerate(time_feats[: self.num_frames]):
                with autocast():
                    proj_feats: GridImage = self.project_cam_time[i](cam_feats)

                voxel_feat, voxel_mask = lift_image_to_3d(proj_feats, target_grid)
                features.append(voxel_feat)
                masks.append(voxel_mask)

        x, _ = merge_grids(features, masks)

        with autocast():
            # run through FPN
            x1, x2, x3, x4 = self.fpn(x)
            assert x1.shape == x.shape

            x0 = x1

            # project to BEV grids
            x1 = x1.flatten(1, 2)
            x1 = self.bev_project[0](x1)
            x2 = x2.flatten(1, 2)
            x2 = self.bev_project[1](x2)
            x3 = x3.flatten(1, 2)
            x3 = self.bev_project[2](x3)
            x4 = x4.flatten(1, 2)
            x4 = self.bev_project[3](x4)

            return x0, (x1, x2, x3, x4), {}
