# pyre-ignore-all-errors[21]: missing optional imports

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


def xformers_available() -> bool:
    try:
        import xformers  # noqa: F401

        return True
    except ImportError:
        return False


def manual_seed(seed: int) -> None:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
