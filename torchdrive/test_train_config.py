import glob
import importlib
import os.path
import sys
import unittest
from typing import List

import torch
from parameterized import parameterized


CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
sys.path.append(CONFIG_DIR)


def get_module_names() -> List[str]:
    configs = glob.glob(os.path.join(CONFIG_DIR, "*.py"))
    assert len(configs) > 0
    module_names = []
    for c in configs:
        module_name, _ = os.path.splitext(os.path.basename(c))
        module_names.append((module_name,))
    return module_names


class TestTrainConfig(unittest.TestCase):
    @parameterized.expand(get_module_names())
    def test_configs(self, module_name: str) -> None:
        config_module = importlib.import_module(module_name)
        config = config_module.CONFIG
        model = config.create_model(device=torch.device("cpu"))
        self.assertIsNotNone(model)
