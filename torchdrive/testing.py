import random
import unittest
from contextlib import contextmanager
from typing import Generator

import numpy as np

import torch


@contextmanager
def skipIfNoCUDA(reason: str = "test requires CUDA") -> Generator[None, None, None]:
    if not torch.cuda.is_available():
        raise unittest.SkipTest(reason)
    yield


def manual_seed(seed: int) -> None:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
