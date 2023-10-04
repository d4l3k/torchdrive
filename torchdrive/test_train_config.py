import glob
import importlib
import os.path
import sys
import unittest

import torch


class TestTrainConfig(unittest.TestCase):
    def test_configs(self):
        config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
        sys.path.append(config_dir)

        configs = glob.glob(os.path.join(config_dir, "*.py"))
        self.assertGreater(len(configs), 0)
        for c in configs:
            module_name, _ = os.path.splitext(os.path.basename(c))
            config_module = importlib.import_module(module_name)
            config = config_module.CONFIG
            model = config.create_model(device=torch.device("cpu"))
            self.assertIsNotNone(model)
