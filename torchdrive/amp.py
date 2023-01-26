from contextlib import contextmanager
from typing import Generator

import torch


@contextmanager
def autocast() -> Generator[None, None, None]:
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        yield
