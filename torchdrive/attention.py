# pyre-ignore-all-errors[21]: missing optional imports

from typing import Optional

import torch

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    import xformers
    from xformers.ops import memory_efficient_attention

    HAS_XFORMERS = True
except ImportError as e:
    HAS_XFORMERS = False


def flash_attention(
    q: torch.Tensor, kv: torch.Tensor, dim: int, num_heads: int, dropout_p: float = 0.0
) -> torch.Tensor:
    BS = q.shape[0]
    q_seqlen = q.shape[1]
    k_seqlen = kv.shape[1]

    out = flash_attn_unpadded_kvpacked_func(
        q=q.reshape(-1, num_heads, dim // num_heads).contiguous(),
        kv=kv.reshape(-1, 2, num_heads, dim // num_heads).contiguous(),
        cu_seqlens_q=torch.arange(
            0,
            (BS + 1) * q_seqlen,
            step=q_seqlen,
            device=q.device,
            dtype=torch.int32,
        ),
        cu_seqlens_k=torch.arange(
            0,
            (BS + 1) * k_seqlen,
            step=k_seqlen,
            device=q.device,
            dtype=torch.int32,
        ),
        max_seqlen_q=q_seqlen,
        max_seqlen_k=k_seqlen,
        dropout_p=0.0,
    )
    return out.reshape(BS, -1, dim)


def xformers_attention(
    q: torch.Tensor, kv: torch.Tensor, dim: int, num_heads: int, dropout_p: float = 0.0
) -> torch.Tensor:
    key = kv[..., :dim]
    value = kv[..., dim:]
    return memory_efficient_attention(
        query=q.contiguous(),
        key=key.contiguous(),
        value=value.contiguous(),
        p=dropout_p,
    )


def naive_attention(
    q: torch.Tensor, kv: torch.Tensor, dim: int, num_heads: int, dropout_p: float = 0.0
) -> torch.Tensor:
    key = kv[..., :dim]
    value = kv[..., dim:]
    return _ref_attention(
        q=q.contiguous(),
        k=key.contiguous(),
        v=value.contiguous(),
        p=dropout_p,
    )


def attention(
    q: torch.Tensor, kv: torch.Tensor, dim: int, num_heads: int, dropout_p: float = 0.0
) -> torch.Tensor:
    if HAS_XFORMERS:
        return xformers_attention(q, kv, dim, num_heads, dropout_p)
    if HAS_FLASH_ATTN and q.is_cuda and q.dtype in (torch.half, torch.bfloat16):
        return flash_attention(q, kv, dim, num_heads, dropout_p)
    return naive_attention(q, kv, dim, num_heads, dropout_p)


def _ref_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    drop_mask: Optional[torch.Tensor] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    From https://github.com/facebookresearch/xformers/blob/main/tests/test_mem_eff_attention.py#L188
    Copyright (c) Facebook, Inc. and its affiliates
    BSD 3-Clause License
    """
    if q.ndim == 4:
        assert p == 0.0
        return _ref_attention_bmhk(q, k, v, attn_bias=attn_bias)
    q = q
    k = k
    v = v

    scale = scale if scale is not None else (1 / q.shape[-1] ** 0.5)
    q = q * scale

    attn = q @ k.transpose(-2, -1)
    if attn_bias is not None:
        if isinstance(attn_bias, xformers.ops.AttentionBias):
            # Always create in B,H,Mq,Mk format
            attn_bias_tensor = attn_bias.materialize(
                (q.shape[0], 1, q.shape[1], k.shape[1]),
                device=q.device,
                dtype=torch.float32,
            )
        else:
            attn_bias_tensor = attn_bias
        if attn_bias_tensor.ndim == 4:
            assert q.shape[0] == attn_bias_tensor.shape[0] * attn_bias_tensor.shape[1]
            attn_bias_tensor = attn_bias_tensor.reshape(
                [-1, *attn_bias_tensor.shape[2:]]
            )
        attn = attn + attn_bias_tensor.float()
    attn = attn.softmax(-1)
    if drop_mask is not None:
        attn = attn * (drop_mask / (1 - p))
    return attn @ v


def _ref_attention_bmhk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    From https://github.com/facebookresearch/xformers/blob/main/tests/test_mem_eff_attention.py#L188
    Copyright (c) Facebook, Inc. and its affiliates
    BSD 3-Clause License
    """
    assert q.ndim == 4

    def T(t: torch.Tensor) -> torch.Tensor:
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    if isinstance(attn_bias, xformers.ops.AttentionBias):
        attn_bias = attn_bias.materialize(
            (q.shape[0], q.shape[2], q.shape[1], k.shape[1]),
            device=q.device,
            dtype=torch.float32,
        ).reshape([q.shape[0] * q.shape[2], q.shape[1], k.shape[1]])
    out = _ref_attention(T(q), T(k), T(v), attn_bias, scale=scale)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))
