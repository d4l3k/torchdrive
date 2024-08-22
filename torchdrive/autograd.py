import itertools
from contextlib import contextmanager
from typing import cast, Generator, List, Optional, overload, Tuple, TypeVar, Union

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def freeze(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = False
    return module


def autograd_pause(tensor: torch.Tensor) -> torch.Tensor:
    """
    autograd_pause returns a new tensor with requires_grad set. The original
    tensor is available as the .parent attribute.

    See autograd_resume.
    """
    if not torch.is_grad_enabled():
        return tensor
    assert tensor.requires_grad, f"tensor does not require grad {tensor}"
    detatched = tensor.detach()
    detatched.requires_grad = True
    detatched.parent = tensor
    return detatched


def _fail_resumed(grad: torch.Tensor) -> None:
    raise RuntimeError("tensor has already been resumed")


def autograd_resume(*tensors: torch.Tensor) -> None:
    """
    autograd_resume resumes the backwards pass for the provided tensors that were
    graph broken via autograd_pause.
    """
    if not torch.is_grad_enabled():
        return
    # ignore zero sized tensors
    tensors = tuple([t for t in tensors if t.numel() > 0])

    for t in tensors:
        assert t.grad is not None, "missing grad"
    # pyre-ignore
    parents: List[torch.Tensor] = [t.parent for t in tensors]
    grad_tensors: List[torch.Tensor] = [cast(torch.Tensor, t.grad) for t in tensors]
    # TODO: run in separate CUDA stream
    torch.autograd.backward(
        tensors=parents,
        grad_tensors=grad_tensors,
    )
    for t in tensors:
        t.register_hook(_fail_resumed)


@overload
@contextmanager
def autograd_context(tensor: torch.Tensor) -> Generator[torch.Tensor, None, None]:
    ...


@overload
@contextmanager
def autograd_context(
    tensor: torch.Tensor, *tensors: torch.Tensor
) -> Generator[Tuple[torch.Tensor, ...], None, None]:
    ...


@contextmanager
def autograd_context(
    tensor: torch.Tensor, *tensors: torch.Tensor
) -> Generator[Union[torch.Tensor, Tuple[torch.Tensor, ...]], None, None]:
    """
    autograd_context pauses the passed in tensors and resumes then when the
    manager exits.
    """
    paused_tensors = tuple(
        autograd_pause(t) for t in itertools.chain([tensor], tensors)
    )
    if len(paused_tensors) == 1:
        yield paused_tensors[0]
    else:
        yield paused_tensors
    autograd_resume(*paused_tensors)


T = TypeVar("T")


@contextmanager
def autograd_optional(tensor: T) -> Generator[T, None, None]:
    if isinstance(tensor, torch.Tensor):
        with autograd_context(tensor) as tensor:
            yield tensor
    else:
        yield tensor


def register_log_grad_norm(
    t: torch.Tensor,
    writer: Optional[SummaryWriter],
    key: str,
    tag: str,
    global_step: int,
) -> None:
    if writer is None:
        return
    if not torch.is_grad_enabled():
        return
    nonopt_writer: SummaryWriter = writer

    def backward_hook(grad: torch.Tensor) -> None:
        nonopt_writer.add_scalars(
            key, {tag: torch.linalg.vector_norm(grad).float()}, global_step=global_step
        )

    t.register_hook(backward_hook)


def log_grad_norm(
    t: torch.Tensor,
    writer: Optional[SummaryWriter],
    key: str,
    tag: str,
    global_step: int,
) -> torch.Tensor:
    """
    log_grad_norm returns a tensor where the backwards gradient norm will be
    logged to Tensorboard.
    """
    if writer is None:
        return t
    # soft clone without copying data
    t = t.view_as(t)

    register_log_grad_norm(
        t=t, writer=writer, key=key, tag=tag, global_step=global_step
    )
    return t
