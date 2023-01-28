import unittest

import torch
from torch import nn

from torchdrive.checkpoint import remap_state_dict


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.a = nn.Parameter(torch.rand(1, 2))
        self.b = nn.Parameter(torch.rand(2, 3))


class TestCheckpoint(unittest.TestCase):
    def test_remap_state_dict(self) -> None:
        m = DummyModel()
        state_dict = {
            "foo.a": torch.rand(1, 2),
            "foo.b": torch.rand(2, 3),
        }
        self.assertCountEqual(
            remap_state_dict(state_dict, m).keys(), ["a", "b", "foo.a", "foo.b"]
        )
