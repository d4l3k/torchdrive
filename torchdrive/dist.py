from typing import Iterable, List

import torch
import torch.distributed as dist
from torch import nn
from torch._C._distributed_c10d import Work


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


def _all_reduce_params(group_params: List[nn.Parameter]) -> Work:
    grads = torch.cat([param.grad.view(-1) for param in group_params])
    size = 0
    for p in group_params:
        grad_size = p.numel()
        p.grad = grads[size : size + grad_size].view_as(p.grad)
        size += grad_size

    return dist.all_reduce(grads, async_op=True)


def run_ddp_concat(
    params: Iterable[nn.Parameter], bucket_cap_elem: int = 6250000
) -> None:
    """
    This does an all reduce on the gradients for the provided parameters.
    Equivalent to DistributedDataParallel but runs at the end.

    Concatenates all gradients before all reducing.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    handles: List[Work] = []

    group_size: int = 0
    group_params: List[nn.Parameter] = []

    for param in params:
        if param.requires_grad and param.grad is not None:
            group_params.append(param)
            group_size += param.grad.numel()
            if group_size >= bucket_cap_elem:
                handles.append(_all_reduce_params(group_params))
                group_params = []
                group_size = 0
    if len(group_params) > 0:
        handles.append(_all_reduce_params(group_params))

    for handle in handles:
        handle.wait()
