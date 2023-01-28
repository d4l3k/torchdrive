from typing import List, Mapping, Tuple, Type

import torch
from torch import nn

from torchdrive.attention import attention
from torchdrive.models.regnet import ConvPEBlock, RegNetEncoder

from torchdrive.positional_encoding import positional_encoding


class CamBEVTransformer(nn.Module):
    """
    This is a combination encoder and transformer. This takes in a stacked set
    of per camera embeddings and uses multiheaded attention to produce a birdseye view
    map.

    The camera frames should be using something like the RegNetEncoder provided
    by this repo but any similar encoder should work.
    """

    def __init__(
        self,
        bev_shape: Tuple[int, int],
        cam_shape: Tuple[int, int],
        dim: int,
        num_cameras: int,
        num_heads: int = 12,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.bev_shape = bev_shape
        self.num_heads = num_heads

        self.register_buffer(
            "positional_encoding", positional_encoding(*bev_shape), persistent=False
        )

        self.context_encoder = nn.Sequential(
            nn.Conv2d(dim * num_cameras, dim, 1),
            nn.MaxPool2d(cam_shape),
        )

        self.query_encoder = nn.Sequential(
            nn.Conv2d(dim + 6, dim, 1),
        )

        self.kv_encoder = nn.Sequential(
            nn.Conv1d(dim, 2 * dim, 1),
        )

    def forward(self, camera_feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            camera_feats: num_cameras * [BS, dim, *cam_shape]
        Returns:
            bev_feats: [BS, dim, *bev_shape]
        """
        merged_camera_feats = torch.cat(camera_feats, dim=1)
        BS = len(merged_camera_feats)

        context = self.context_encoder(merged_camera_feats)
        context = torch.tile(context, (1, 1, *self.bev_shape))
        pos_enc = self.positional_encoding.expand(len(context), -1, -1, -1)
        context = torch.concat((context, pos_enc), dim=1)

        query = (
            self.query_encoder(context).permute(0, 2, 3, 1).reshape(BS, -1, self.dim)
        )
        q_seqlen = query.shape[1]

        x = merged_camera_feats.reshape(BS, self.dim, -1)
        kv = self.kv_encoder(x).permute(0, 2, 1)
        bev = attention(query, kv, dim=self.dim, num_heads=self.num_heads)
        bev = bev.reshape(BS, *self.bev_shape, self.dim)
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
        transformer: Type[CamBEVTransformer] = CamBEVTransformer,
        conv: Type[ConvPEBlock] = ConvPEBlock,
    ) -> None:
        super().__init__()

        self.cameras = cameras
        self.cam_encoders = nn.ModuleDict(
            {cam: cam_encoder(cam_shape, dim=dim) for cam in cameras}
        )

        self.transformer: nn.Module = transformer(
            bev_shape=bev_shape,
            cam_shape=self.cam_encoders[cameras[0]].output_shape,
            dim=dim,
            num_cameras=len(self.cameras),
        )
        self.conv: nn.Module = conv(
            dim,
            dim,
            bev_shape=bev_shape,
        )

    def forward(self, camera_frames: Mapping[str, torch.Tensor]) -> torch.Tensor:
        ordered_frames = [
            self.cam_encoders[cam](camera_frames[cam]) for cam in self.cameras
        ]
        return self.conv(self.transformer(ordered_frames))


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

    def forward(self, bevs: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(bevs, dim=1)
        return self.merge(x)
