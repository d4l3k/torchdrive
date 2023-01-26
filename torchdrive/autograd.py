import itertools
from contextlib import contextmanager
from typing import cast, Generator, List, overload, Tuple, Union

import torch


def autograd_pause(tensor: torch.Tensor) -> torch.Tensor:
    """
    autograd_pause returns a new tensor with requires_grad set. The original
    tensor is available as the .parent attribute.

    See autograd_resume.
    """
    detatched = tensor.detach()
    detatched.requires_grad = True
    detatched.parent = tensor
    return detatched


def autograd_resume(*tensors: torch.Tensor) -> None:
    """
    autograd_resume resumes the backwards pass for the provided tensors that were
    graph broken via autograd_pause.
    """
    for t in tensors:
        assert t.grad is not None, "missing grad"
    # pyre-ignore
    parents: List[torch.Tensor] = [t.parent for t in tensors]
    grad_tensors: List[torch.Tensor] = [cast(torch.Tensor, t.grad) for t in tensors]
    torch.autograd.backward(
        tensors=parents,
        grad_tensors=grad_tensors,
    )


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
