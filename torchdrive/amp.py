from contextlib import contextmanager
from typing import Generator

import torch


if not torch.cuda.is_available():

    @contextmanager
    def autocast() -> Generator[None, None, None]:
        """
        noop if no cuda
        """
        yield

else:

    @contextmanager
    def autocast() -> Generator[None, None, None]:
        """
        This does a bfloat16 autocast using torch.cuda.amp.autocast.
        """
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            yield
