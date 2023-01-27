# pyre-ignore-all-errors[21]: missing optional imports

import warnings
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
except ImportError as e:
    warnings.warn(f"flash_attn not available: {e}")

try:
    from xformers.ops import memory_efficient_attention
except ImportError as e:
    warnings.warn(f"xformers not available: {e}")


from torchdrive.positional_encoding import positional_encoding


class BaseCamBEVTransformer(nn.Module, ABC):
    """
    This is a combination encoder and transformer. This takes in a stacked set
    of camera frames and uses multiheaded attention to produce a birdseye view
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

    def forward(self, camera_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            camera_feats: [BS, dim*num_cameras, *cam_shape]
        Returns:
            bev_feats: [BS, dim, *bev_shape]
        """
        BS = len(camera_feats)

        context = self.context_encoder(camera_feats)
        context = torch.tile(context, (1, 1, *self.bev_shape))
        pos_enc = self.positional_encoding.expand(len(context), -1, -1, -1)
        context = torch.concat((context, pos_enc), dim=1)

        query = (
            self.query_encoder(context).permute(0, 2, 3, 1).reshape(BS, -1, self.dim)
        )
        q_seqlen = query.shape[1]

        x = camera_feats.reshape(BS, self.dim, -1)
        kv = self.kv_encoder(x).permute(0, 2, 1)
        bev = self._transform(BS, query, kv)
        bev = bev.reshape(BS, *self.bev_shape, self.dim)
        bev = bev.permute(0, 3, 1, 2)
        return bev

    @abstractmethod
    def _transform(
        self, BS: int, query: torch.Tensor, kv: torch.Tensor
    ) -> torch.Tensor:
        ...


class NaiveCamBEVTransformer(BaseCamBEVTransformer):
    """
    This implements the BaseCamBEVTransformer with the stock pytorch
    MultiheadedAttention implementation. Not recommended.
    """

    def __init__(
        self,
        bev_shape: Tuple[int, int],
        cam_shape: Tuple[int, int],
        dim: int,
        num_cameras: int,
        num_heads: int = 12,
    ) -> None:
        super().__init__(
            bev_shape=bev_shape,
            cam_shape=cam_shape,
            dim=dim,
            num_cameras=num_cameras,
            num_heads=num_heads,
        )

        self.transformer = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            batch_first=True,
            bias=False,
        )

    def _transform(
        self, BS: int, query: torch.Tensor, kv: torch.Tensor
    ) -> torch.Tensor:
        key = kv[..., : self.dim]
        value = kv[..., self.dim :]
        bev, _ = self.transformer(
            query,
            key,
            value,
            need_weights=False,
        )
        return bev


class FlashAttnCamBEVTransformer(BaseCamBEVTransformer):
    """
    Implements BaseCamBEVTransformer with flash_attn. Only supports CUDA and
    fp16/bfloat16.
    """

    def _transform(
        self, BS: int, query: torch.Tensor, kv: torch.Tensor
    ) -> torch.Tensor:
        q_seqlen = query.shape[1]
        k_seqlen = kv.shape[1]

        return flash_attn_unpadded_kvpacked_func(
            q=query.reshape(
                -1, self.num_heads, self.dim // self.num_heads
            ).contiguous(),
            kv=kv.reshape(
                -1, 2, self.num_heads, self.dim // self.num_heads
            ).contiguous(),
            cu_seqlens_q=torch.arange(
                0,
                (BS + 1) * q_seqlen,
                step=q_seqlen,
                device=query.device,
                dtype=torch.int32,
            ),
            cu_seqlens_k=torch.arange(
                0,
                (BS + 1) * k_seqlen,
                step=k_seqlen,
                device=query.device,
                dtype=torch.int32,
            ),
            max_seqlen_q=q_seqlen,
            max_seqlen_k=k_seqlen,
            dropout_p=0.0,
        )


class XformersCamBEVTransformer(BaseCamBEVTransformer):
    """
    Implements BaseCamBEVTransformer with xformers. This should be the most
    performant option and has better support across devices than flash_attn.
    """

    def _transform(
        self, BS: int, query: torch.Tensor, kv: torch.Tensor
    ) -> torch.Tensor:
        key = kv[..., : self.dim]
        value = kv[..., self.dim :]
        return memory_efficient_attention(
            query=query.contiguous(),
            key=key.contiguous(),
            value=value.contiguous(),
        )
