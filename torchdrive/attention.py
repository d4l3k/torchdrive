# pyre-ignore-all-errors[21]: missing optional imports

import torch
import torch.nn.functional as F


def attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    dim: int,
    num_heads: int,
    dropout_p: float = 0.0,
    causal: bool = False,
) -> torch.Tensor:
    """
    attention is a dispatcher for multiheaded attention and will use the most
    efficient option that's available in the current environment for the
    specific dtype/device.

    Backends (in order of preference):
    * xformers (everything)
    * flash_attn (cuda, fp16/bfloat16 only)
    * naive pytorch (everything)

    Args:
        q: [BS, num_queries, dim]
        kv: [BS, num_kvs, dim*2]
    Returns:
        [BS, num_queries, dim]
    """
    assert q.size(-1) % 8 == 0, q.shape
    assert (kv.size(-1) // 2) % 8 == 0, q.shape

    # pyre-fixme[16]: no attributed scaled_dot_product_attention
    return F.scaled_dot_product_attention(
        query=q,
        key=kv[..., :dim],
        value=kv[..., dim:],
        dropout_p=dropout_p,
        is_causal=causal,
    )
