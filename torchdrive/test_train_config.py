import glob
import importlib
import os.path
import sys
import unittest
from typing import List

import torch
from parameterized import parameterized

from torchdrive.datasets.dataset import Datasets
from torchdrive.train_config import create_parser, TrainConfig


# pyre-fixme[5]: Global expression must be annotated.
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
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torchdrive.test_train_config.get_module_names()` to decorator factory
    #  `parameterized.parameterized.expand`.
    # pyre-fixme[16]: Module `parameterized` has no attribute `expand`.
    @parameterized.expand(get_module_names())
    def test_configs(self, module_name: str) -> None:
        config_module = importlib.import_module(f"configs.{module_name}")
        print(module_name, config_module)
        config = config_module.CONFIG
        model = config.create_model(device=torch.device("cpu"))
        self.assertIsNotNone(model)

        config.dataset = Datasets.DUMMY
        dataset = config.create_dataset()

    def test_parser(self) -> None:
        parser = create_parser()
        args = parser.parse_args(
            [
                "--output=foo",
                "--config=simplebev3d",
                "--config.lr=1234",
                "--config.ae=true",
            ]
        )
        self.assertIsInstance(args.config, TrainConfig)
        self.assertEqual(args.config.lr, 1234)
        self.assertEqual(args.config.ae, True)
