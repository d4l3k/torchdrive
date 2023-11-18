from contextlib import contextmanager
from typing import Dict, Generator, List, Tuple

import torch
from torch import nn


@contextmanager
def collect_cross_attention_weights(
    m: nn.Module,
) -> Generator[List[torch.Tensor], None, None]:
    """
    This is a context manager that captures the cross attention weights used by
    the provided module in a list that is returned.
    """
    out: List[torch.Tensor] = []

    def forward_pre_hook(
        module: nn.Module, args: List[object], kwargs: Dict[str, object]
    ) -> Tuple[List[object], Dict[str, object]]:
        kwargs["need_weights"] = True
        return args, kwargs

    def forward_hook(
        module: nn.Module, input: object, outputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        _, weights = outputs
        out.append(weights)

    handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(m: nn.Module) -> None:
        for name, module in m.named_children():
            if isinstance(module, nn.MultiheadAttention) and name == "multihead_attn":
                handles.append(
                    module.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)
                )
                handles.append(module.register_forward_hook(forward_hook))
            else:
                register(module)

    register(m)

    try:
        yield out
    finally:
        for handle in handles:
            handle.remove()
