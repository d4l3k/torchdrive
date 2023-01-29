from difflib import SequenceMatcher
from typing import Dict
from collections import OrderedDict

import torch
from torch import nn


def similarity(a: str, b: str) -> float:
    """
    similarity returns a similarity score between the provided strings.
    """
    return SequenceMatcher(None, a, b).ratio()


def remap_state_dict(
    state_dict: Dict[str, torch.Tensor], m: nn.Module
) -> OrderedDict[str, torch.Tensor]:
    """
    remap_state_dict maps the parameters from provided state_dict to best match
    any new/renamed parameters in the new module. It uses shape information and
    similarity search to find the best candidate in the new model.
    """
    to_load = OrderedDict[str, torch.Tensor](state_dict)

    for k, v in m.state_dict().items():
        if "frozen" in k:
            continue
        if k in state_dict and v.shape == state_dict[k].shape:
            continue
        # avoid transferring batch normalization stats
        if ".num_batches_tracked" in k or ".running_mean" in k or "running_var" in k:
            continue
        _, _, ksuffix = k.rpartition(".")
        found = None
        found_key = None
        found_score = None
        for k2, v2 in state_dict.items():
            _, _, k2suffix = k2.rpartition(".")
            if ksuffix != k2suffix:
                continue
            if v.shape == v2.shape:
                score = similarity(k2, k)
                if found_score is None or score > found_score:
                    found = v2
                    found_key = k2
                    found_score = score
        if found is not None:
            to_load[k] = found
            print(f"remapping {k} to {found_key}, ratio {found_score} {found.shape}")
        else:
            print(f"failed to find {k} {v.shape}")

    return to_load
