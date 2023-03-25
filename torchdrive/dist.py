from typing import Iterable

import torch.distributed as dist

from torch import nn


def run_ddp(params: Iterable[nn.Parameter]) -> None:
    """
    This does an all reduce on the gradients for the provided parameters.
    Equivalent to DistributedDataParallel but runs at the end.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    handles = []
    for param in params:
        if param.requires_grad and param.grad is not None:
            handles.append(dist.all_reduce(param.grad, async_op=True))
    for handle in handles:
        handle.wait()
