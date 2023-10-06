from typing import Callable, Dict, List, Mapping, Optional, Tuple, Type

import torch
from torch import nn

from torchdrive.amp import autocast

from torchdrive.attention import attention
from torchdrive.autograd import autograd_pause
from torchdrive.data import Batch
from torchdrive.models.bev_backbone import BEVBackbone
from torchdrive.models.regnet import (
    ConvPEBlock,
    RegNetEncoder,
    resnet_init,
    UpsamplePEBlock,
)
from torchdrive.positional_encoding import apply_sin_cos_enc2d


class GridTransformer(nn.Module):
    """
    This is a combination encoder and transformer. This takes in a stacked set
    of embeddings and uses multiheaded attention to produce a new output grid.

    This can be used for multiple cameras to BEV transformation but works for
    most grid to grid transforms. The camera frames should be using something
    like the RegNetEncoder provided by this repo but any similar encoder should
    work.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_shape: Tuple[int, int],
        input_dim: int,
        dim: int,
        num_inputs: int,
        num_heads: int = 12,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.output_shape = output_shape
        self.num_heads = num_heads

        self.context_encoder = nn.Sequential(
            nn.Conv2d(input_dim * num_inputs, dim, 1),
            nn.MaxPool2d(input_shape),
        )

        self.query_encoder = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
        )

        self.kv_encoder = nn.Sequential(
            nn.Conv1d(input_dim, 2 * dim, 1),
        )
        self.pos_encoding = apply_sin_cos_enc2d

    def forward(self, input_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs_feats: num_inputs * [BS, dim, *cam_shape]
        Returns:
            bev_feats: [BS, dim, *bev_shape]
        """
        merged_feats = torch.cat(input_feats, dim=1)
        BS = len(merged_feats)

        context = self.context_encoder(merged_feats)
        context = context.expand(-1, -1, *self.output_shape)
        context = self.pos_encoding(context)

        query = self.query_encoder(context).permute(0, 2, 3, 1).flatten(1, 2)
        q_seqlen = query.shape[1]

        x = torch.stack(input_feats, dim=-1).flatten(2, 4)
        kv = self.kv_encoder(x).permute(0, 2, 1)
        bev = attention(query, kv, dim=self.dim, num_heads=self.num_heads)
        bev = bev.unflatten(1, self.output_shape)
        bev = bev.permute(0, 3, 1, 2)
        return bev


class CamBEVEncoder(nn.Module):
    def __init__(
        self,
        cameras: List[str],
        bev_shape: Tuple[int, int],
        cam_shape: Tuple[int, int],
        dim: int,
        cam_encoder: Type[RegNetEncoder] = RegNetEncoder,
        transformer: Type[GridTransformer] = GridTransformer,
        conv: Type[ConvPEBlock] = ConvPEBlock,
    ) -> None:
        super().__init__()

        self.cameras = cameras
        self.cam_encoders = nn.ModuleDict(
            {cam: cam_encoder(cam_shape, dim=dim) for cam in cameras}
        )
        self.cam_transformers = nn.ModuleDict(
            {
                cam: transformer(
                    output_shape=bev_shape,
                    input_shape=self.cam_encoders[cameras[0]].output_shape,
                    input_dim=dim,
                    dim=dim,
                    num_inputs=1,
                )
                for cam in cameras
            }
        )
        resnet_init(self.cam_transformers)

        self.conv: nn.Module = conv(
            len(cameras) * dim,
            dim,
            input_shape=bev_shape,
        )
        resnet_init(self.conv)

    def per_cam_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.cam_transformers.parameters()) + list(
            self.cam_encoders.parameters()
        )

    def forward(
        self,
        camera_frames: Mapping[str, torch.Tensor],
        pause: bool = False,
        cam_feat_fn: Optional[Callable[[str, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        cam_feats = {
            cam: self.cam_encoders[cam](camera_frames[cam]) for cam in self.cameras
        }
        if pause:
            for k, v in cam_feats.items():
                cam_feats[k] = autograd_pause(v)
        ordered_grids = []
        for cam in self.cameras:
            cam_feat = cam_feats[cam]
            if cam_feat_fn is not None:
                cam_feat = cam_feat_fn(cam, cam_feat)
            ordered_grids.append(self.cam_transformers[cam]([cam_feat]))
        return cam_feats, self.conv(torch.cat(ordered_grids, dim=1))


class BEVMerger(nn.Module):
    """
    Encodes the multiple BEV maps into a single one. Useful for handling
    multiple frames across time.
    """

    def __init__(self, num_frames: int, bev_shape: Tuple[int, int], dim: int) -> None:
        super().__init__()

        self.merge = ConvPEBlock(
            dim * num_frames,
            dim,
            bev_shape,
        )
        resnet_init(self.merge)

    def forward(self, bevs: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(bevs, dim=1)
        return self.merge(x)


class BEVUpsampler(nn.Module):
    """
    Upsamples the BEV grid into a larger resolution but lower channel count.
    Each block halves the input dimension.
    """

    def __init__(
        self, num_upsamples: int, bev_shape: Tuple[int, int], dim: int, output_dim: int
    ) -> None:
        """
        Args:
            num_upsamples: number of UpsamplePEBlocks to use
            bev_shape: initial input shape
            dim: input dim
            output_dim: minimum dimension and output dimension
        """
        super().__init__()

        blocks: List[nn.Module] = []
        cur_dim = dim
        cur_shape = bev_shape
        for i in range(num_upsamples):
            next_dim = max(cur_dim // 2, output_dim)
            if i == (num_upsamples - 1):
                next_dim = output_dim
            blocks.append(
                UpsamplePEBlock(
                    in_ch=cur_dim,
                    out_ch=next_dim,
                    input_shape=cur_shape,
                )
            )
            cur_dim = next_dim
            cur_shape = (cur_shape[0] * 2, cur_shape[1] * 2)

        self.upsample = nn.Sequential(*blocks)
        resnet_init(self.upsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class RiceBackbone(BEVBackbone):
    """
    A BEV backbone using the custom Rice transformer and bev merger.
    """

    def __init__(
        self,
        dim: int,
        hr_dim: int,
        cam_dim: int,
        grid_shape: Tuple[int, int, int],  # [x, y, z]
        input_shape: Tuple[int, int],
        num_frames: int,
        cameras: List[str],
        num_upsamples: int,
    ) -> None:
        super().__init__()

        self.num_frames = num_frames
        bev_shape = grid_shape[:2]  # [x, y]
        self.out_Z = grid_shape[2] * 2**num_upsamples
        self.voxel_dim = max(hr_dim // self.out_Z, 1)

        self.cam_transformers = nn.ModuleDict(
            {
                cam: GridTransformer(
                    output_shape=bev_shape,
                    input_shape=input_shape,
                    dim=dim,
                    input_dim=cam_dim,
                    num_inputs=1,
                )
                for cam in cameras
            }
        )
        resnet_init(self.cam_transformers)

        self.frame_merger = BEVMerger(
            num_frames=num_frames, bev_shape=bev_shape, dim=dim
        )

        self.upsample = BEVUpsampler(
            num_upsamples=num_upsamples,
            bev_shape=bev_shape,
            dim=dim,
            output_dim=hr_dim,
        )
        self.project_voxel = nn.Conv2d(hr_dim, self.voxel_dim * self.out_Z, 1)

    def forward(
        self, camera_features: Mapping[str, List[torch.Tensor]], batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with autocast():
            bev_grids = []

            for i in range(self.num_frames):
                ordered_grids = []
                for cam, time_feats in camera_features.items():
                    cam_feat = time_feats[i]
                    ordered_grids.append(self.cam_transformers[cam]([cam_feat]))
                bev_grids.append(torch.stack(ordered_grids, dim=0).mean(dim=0))

            bev = self.frame_merger(bev_grids)

            hr_bev = self.upsample(bev)
            hr_bev = self.project_voxel(hr_bev)
            hr_bev = hr_bev.unflatten(1, (self.voxel_dim, self.out_Z))

            return hr_bev, bev
