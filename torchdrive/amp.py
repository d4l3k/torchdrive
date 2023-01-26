from contextlib import contextmanager
from typing import Generator

import torch


@contextmanager
def autocast() -> Generator[None, None, None]:
    """
    This does a bfloat16 autocast using torch.cuda.amp.autocast.
    """
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        yield
