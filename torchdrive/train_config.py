import argparse
import importlib
from dataclasses import dataclass, fields
from typing import Callable, List, Optional, Tuple

import torch
from dataclasses_json import dataclass_json
from torch import nn

from torchdrive.autograd import freeze

from torchdrive.datasets.dataset import Dataset, Datasets
from torchdrive.tasks.bev import BEVTask, BEVTaskVan
from torchdrive.tasks.diff_traj import DiffTraj
from torchdrive.tasks.vit_jepa import ViTJEPA


@dataclass
class DatasetConfig:
    # shared
    cameras: List[str]
    num_frames: int
    cam_shape: Tuple[int, int]

    # dataset only params
    dataset: Datasets
    dataset_path: str
    mask_path: str
    batch_size: int
    num_workers: int
    autolabel_path: str
    autolabel: bool

    def create_dataset(self, smoke: bool = False) -> Dataset:
        if self.dataset == Datasets.RICE:
            from torchdrive.datasets.rice import MultiCamDataset

            dataset = MultiCamDataset(
                index_file=self.dataset_path,
                mask_dir=self.mask_path,
                cameras=self.cameras,
                dynamic=True,
                cam_shape=self.cam_shape,
                # 3 encode frames, 3 decode frames, overlap last frame
                nframes_per_point=self.num_frames,
                # pyre-fixme[16]: `TrainConfig` has no attribute `limit_size`.
                limit_size=self.limit_size,
            )
        elif self.dataset == Datasets.NUSCENES:
            from torchdrive.datasets.nuscenes_dataset import NuscenesDataset

            dataset = NuscenesDataset(
                data_dir=self.dataset_path,
                version="v1.0-mini" if smoke else "v1.0-trainval",
                lidar=False,
                num_frames=self.num_frames,
            )
        elif self.dataset == Datasets.DUMMY:
            from torchdrive.datasets.dummy import DummyDataset

            dataset = DummyDataset()
            dataset.cameras = self.cameras
        else:
            raise ValueError(f"unknown dataset type {self.dataset}")

        if self.autolabel:
            from torchdrive.datasets.autolabeler import AutoLabeler

            dataset = AutoLabeler(dataset, path=self.autolabel_path)

        for cam in self.cameras:
            assert cam in dataset.cameras, "invalid camera"
        return dataset


@dataclass
class OptimizerConfig:
    epochs: int
    lr: float
    grad_clip: float
    step_size: int


@dataclass_json
@dataclass
class TrainConfig(DatasetConfig, OptimizerConfig):
    # backbone settings
    dim: int
    cam_dim: int
    hr_dim: int
    num_upsamples: int  # number of upsamples to do after the backbone
    grid_shape: Tuple[int, int, int]  # [x, y, z]
    backbone: str
    cam_encoder: str
    num_encode_frames: int

    # tasks
    det: bool
    ae: bool
    voxel: bool
    voxelsem: List[str]
    path: bool
    voxel_jepa: bool

    # det config
    det_num_queries: int = 1000

    start_offsets: Tuple[int, ...] = (0,)
    freeze_cam_backbone: bool = False
    cam_features_mask_ratio: float = 0.0

    def create_model(
        self,
        device: torch.device,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> BEVTaskVan:
        from torchdrive.transforms.batch import (
            Compose,
            NormalizeCarPosition,
            RandomRotation,
            RandomTranslation,
        )

        adjust: int = 2**self.num_upsamples
        adjusted_grid_shape = tuple(v // adjust for v in self.grid_shape)

        if self.backbone == "rice":
            from torchdrive.models.bev import RiceBackbone

            h: int
            w: int
            h, w = self.cam_shape
            # pyre-fixme[10]: Name `BEVBackbone` is used but not defined.
            backbone: BEVBackbone = RiceBackbone(
                dim=self.dim,
                cam_dim=self.cam_dim,
                grid_shape=adjusted_grid_shape,
                input_shape=(h // adjust, w // adjust),
                hr_dim=self.hr_dim,
                num_frames=self.num_encode_frames,
                cameras=self.cameras,
                num_upsamples=self.num_upsamples,
            )
        elif self.backbone == "simple_bev":
            from torchdrive.models.simple_bev import SegnetBackbone

            backbone = SegnetBackbone(
                grid_shape=adjusted_grid_shape,
                dim=self.dim,
                hr_dim=self.hr_dim,
                cam_dim=self.cam_dim,
                num_frames=self.num_encode_frames,
                scale=3 / adjust,
                num_upsamples=self.num_upsamples,
                compile_fn=compile_fn,
            )
        elif self.backbone == "simple_bev3d":
            from torchworld.models.simplebev_3d import SimpleBEV3DBackbone

            backbone = SimpleBEV3DBackbone(
                grid_shape=self.grid_shape,
                dim=self.dim,
                hr_dim=self.hr_dim,
                cam_dim=self.cam_dim,
                num_frames=3,
                compile_fn=compile_fn,
            )
        else:
            raise ValueError(f"unknown backbone {self.backbone}")

        if self.cam_encoder == "regnet":
            from torchdrive.models.regnet import RegNetEncoder

            h: int
            w: int
            h, w = self.cam_shape

            cam_feats_shape: Tuple[int, int] = (h // 16, w // 16)

            def cam_encoder() -> RegNetEncoder:
                return RegNetEncoder(
                    cam_shape=self.cam_shape,
                    dim=self.cam_dim,
                )

        elif self.cam_encoder == "simple_regnet":
            from torchvision import models

            from torchdrive.models.simple_bev import RegNetEncoder

            h: int
            w: int
            h, w = self.cam_shape

            cam_feats_shape = (h // 8, w // 8)

            def cam_encoder() -> RegNetEncoder:
                cam_backbone = models.regnet_x_800mf(pretrained=True)
                if self.freeze_cam_backbone:
                    cam_backbone = freeze(cam_backbone)
                return RegNetEncoder(C=self.cam_dim, regnet=cam_backbone)

        else:
            raise ValueError(f"unknown cam encoder {self.cam_encoder}")

        # pyre-fixme[10]: Name `Dict` is used but not defined.
        tasks: Dict[str, BEVTask] = {}
        hr_tasks: Dict[str, BEVTask] = {}
        bev_shape = tuple(v // 16 for v in self.grid_shape[:2])
        if self.path:
            from torchdrive.tasks.path import PathTask

            tasks["path"] = PathTask(
                bev_shape=bev_shape,
                bev_dim=self.dim,
                dim=256,
                # compile_fn=compile_fn,
            )
        if self.det:
            from torchdrive.tasks.det import DetTask

            tasks["det"] = DetTask(
                cameras=self.cameras,
                cam_shape=self.cam_shape,
                bev_shape=bev_shape,
                dim=self.dim,
                device=device,
                # compile_fn=compile_fn,
                num_queries=self.det_num_queries,
            )
        if self.ae:
            from torchdrive.tasks.ae import AETask

            tasks["ae"] = AETask(
                cameras=self.cameras,
                cam_shape=self.cam_shape,
                bev_shape=bev_shape,
                dim=self.dim,
            )
        if self.voxel:
            from torchdrive.tasks.voxel import VoxelTask

            hr_tasks["voxel"] = VoxelTask(
                cameras=self.cameras,
                cam_shape=self.cam_shape,
                dim=self.dim,
                hr_dim=self.hr_dim,
                cam_dim=self.cam_dim,
                cam_feats_shape=cam_feats_shape,
                height=self.grid_shape[2],  # z
                z_offset=0.4,  # TODO: share across SimpleBev and here
                device=device,
                semantic=self.voxelsem,
                # camera_overlap=dataset.CAMERA_OVERLAP,
                compile_fn=compile_fn,
                start_offsets=self.start_offsets,
            )
        if self.voxel_jepa:
            from torchdrive.tasks.voxel_jepa import VoxelJEPATask

            hr_tasks["voxel_jepa"] = VoxelJEPATask(
                cameras=self.cameras,
                cam_shape=self.cam_shape,
                dim=self.dim,
                hr_dim=self.hr_dim,
                cam_dim=self.cam_dim,
                cam_feats_shape=cam_feats_shape,
                height=self.grid_shape[2],  # z
                z_offset=0.4,  # TODO: share across SimpleBev and here
                device=device,
                # camera_overlap=dataset.CAMERA_OVERLAP,
                compile_fn=compile_fn,
                start_offsets=self.start_offsets,
            )

        model = BEVTaskVan(
            tasks=tasks,
            hr_tasks=hr_tasks,
            cameras=self.cameras,
            dim=self.dim,
            hr_dim=self.hr_dim,
            cam_dim=self.cam_dim,
            grid_shape=self.grid_shape,
            scale=3.0,
            cam_features_mask_ratio=self.cam_features_mask_ratio,
            compile_fn=compile_fn,
            num_encode_frames=self.num_encode_frames,
            backbone=backbone,
            cam_encoder=cam_encoder,
            transform=Compose(
                NormalizeCarPosition(start_frame=self.num_encode_frames - 1),
                RandomRotation(),
                RandomTranslation(distances=(5.0, 5.0, 0.0)),
            ),
        )

        model = model.to(device)
        return model


@dataclass_json
@dataclass
class ViTJEPATrainConfig(DatasetConfig, OptimizerConfig):
    num_encode_frames: int

    def create_model(
        self,
        device: torch.device,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> ViTJEPA:
        return ViTJEPA(
            cameras=self.cameras,
            num_encode_frames=self.num_encode_frames,
            cam_shape=self.cam_shape,
            num_frames=self.num_frames,
        ).to(device)


@dataclass_json
@dataclass
class DiffTrajTrainConfig(DatasetConfig, OptimizerConfig):
    num_encode_frames: int

    def create_model(
        self,
        device: torch.device,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> DiffTraj:
        model = DiffTraj(
            cameras=self.cameras,
            num_encode_frames=self.num_encode_frames,
            cam_shape=self.cam_shape,
            num_frames=self.num_frames,
        ).to(device)

        for cam_encoder in model.encoders.values():
            cam_encoder.freeze_pretrained_weights()

        return model


class _ConfigAction(argparse.Action):
    def __init__(self, dest: str, *args: object, **kwargs: object) -> None:
        # pyre-fixme[6]: For 1st argument expected `Sequence[str]` but got `object`.
        # pyre-fixme[6]: For 2nd argument expected `Union[None, int, str]` but got
        #  `object`.
        # pyre-fixme[6]: For 5th argument expected `Union[typing.Callable[[str],
        #  Variable[_T]], None, FileType]` but got `object`.
        # pyre-fixme[6]: For 6th argument expected
        #  `Optional[Iterable[Variable[_T]]]` but got `object`.
        # pyre-fixme[6]: For 7th argument expected `bool` but got `object`.
        # pyre-fixme[6]: For 8th argument expected `Optional[str]` but got `object`.
        # pyre-fixme[6]: For 9th argument expected `Union[None, str,
        #  typing.Tuple[str, ...]]` but got `object`.
        super().__init__(*args, dest=dest, **kwargs)
        self.dest = dest

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        value: str,
        option_string: Optional[str] = None,
    ) -> None:
        config_module = importlib.import_module(f"configs.{value}")
        config = config_module.CONFIG

        setattr(namespace, self.dest, config)


class _ConfigFieldAction(argparse.Action):
    def __init__(self, dest: str, *args: object, **kwargs: object) -> None:
        # pyre-fixme[6]: For 1st argument expected `Sequence[str]` but got `object`.
        # pyre-fixme[6]: For 2nd argument expected `Union[None, int, str]` but got
        #  `object`.
        # pyre-fixme[6]: For 5th argument expected `Union[typing.Callable[[str],
        #  Variable[_T]], None, FileType]` but got `object`.
        # pyre-fixme[6]: For 6th argument expected
        #  `Optional[Iterable[Variable[_T]]]` but got `object`.
        # pyre-fixme[6]: For 7th argument expected `bool` but got `object`.
        # pyre-fixme[6]: For 8th argument expected `Optional[str]` but got `object`.
        # pyre-fixme[6]: For 9th argument expected `Union[None, str,
        #  typing.Tuple[str, ...]]` but got `object`.
        super().__init__(*args, dest=dest, **kwargs)
        self.dest = dest

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        value: str,
        option_string: Optional[str] = None,
    ) -> None:
        target, _, field = self.dest.partition(".")
        config = getattr(namespace, target)
        setattr(config, field, value)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument("--output", required=True, type=str, default="out")
    parser.add_argument("--load", type=str)
    parser.add_argument("--skip_load_optim", default=False, action="store_true")
    parser.add_argument("--anomaly_detection", default=False, action="store_true")
    parser.add_argument("--limit_size", type=int)
    parser.add_argument("--checkpoint_every", type=int, default=2000)
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument(
        "--grad_sizes", default=False, action="store_true", help="log grad sizes"
    )
    parser.add_argument(
        "--compile", default=False, action="store_true", help="use torch.compile"
    )
    parser.add_argument(
        "--smoke",
        default=False,
        action="store_true",
        help="run with a smaller smoke test config",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="the config file name to use",
        action=_ConfigAction,
    )

    for field in fields(TrainConfig):
        parser.add_argument(
            f"--config.{field.name}", type=field.type, action=_ConfigFieldAction
        )

    return parser
