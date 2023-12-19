from typing import Dict, Union

import torch


def is_nan(x: torch.Tensor) -> Union[torch.Tensor, bool]:
    if x is None:
        return False
    return x.isnan().any() or x.isinf().any()


def assert_not_nan(x: torch.Tensor, msg: str = "") -> None:
    assert not is_nan(x), f"{x} is NaN: {msg}"


def assert_not_nan_dict(x: Dict[str, torch.Tensor]) -> None:
    for k, v in x.items():
        assert_not_nan(v, k)
