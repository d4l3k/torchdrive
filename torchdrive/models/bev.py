from typing import List, Mapping, Tuple, Type

import torch
from torch import nn

from torchdrive.attention import attention
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
        dim: int,
        num_inputs: int,
        num_heads: int = 12,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.output_shape = output_shape
        self.num_heads = num_heads

        self.context_encoder = nn.Sequential(
            nn.Conv2d(dim * num_inputs, dim, 1),
            nn.MaxPool2d(input_shape),
        )

        self.query_encoder = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
        )

        self.kv_encoder = nn.Sequential(
            nn.Conv1d(dim, 2 * dim, 1),
        )

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
        context = apply_sin_cos_enc2d(context)

        query = (
            self.query_encoder(context).permute(0, 2, 3, 1).reshape(BS, -1, self.dim)
        )
        q_seqlen = query.shape[1]

        x = merged_feats.reshape(BS, self.dim, -1)
        kv = self.kv_encoder(x).permute(0, 2, 1)
        bev = attention(query, kv, dim=self.dim, num_heads=self.num_heads)
        bev = bev.reshape(BS, *self.output_shape, self.dim)
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

        self.transformer: nn.Module = transformer(
            output_shape=bev_shape,
            input_shape=self.cam_encoders[cameras[0]].output_shape,
            dim=dim,
            num_inputs=len(self.cameras),
        )
        resnet_init(self.transformer)
        self.conv: nn.Module = conv(
            dim,
            dim,
            input_shape=bev_shape,
        )
        resnet_init(self.conv)

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
