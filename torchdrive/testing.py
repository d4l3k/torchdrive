import unittest
from contextlib import contextmanager
from typing import Generator

import torch


@contextmanager
def skipIfNoCUDA(reason: str = "test requires CUDA") -> Generator[None, None, None]:
    if not torch.cuda.is_available():
        raise unittest.SkipTest(reason)
    yield
