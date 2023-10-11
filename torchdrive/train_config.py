import argparse
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
from torch import nn

from torchdrive.datasets.dataset import Dataset, Datasets
from torchdrive.tasks.bev import BEVTask, BEVTaskVan


@dataclass
class TrainConfig:
    # backbone settings
    cameras: List[str]
    dim: int
    cam_dim: int
    hr_dim: int
    num_upsamples: int  # number of upsamples to do after the backbone
    grid_shape: Tuple[int, int, int]  # [x, y, z]
    cam_shape: Tuple[int, int]
    backbone: str
    cam_encoder: str
    num_frames: int
    num_encode_frames: int

    # optimizer settings
    epochs: int
    lr: float
    grad_clip: float
    step_size: int

    # dataset config
    dataset: Datasets
    dataset_path: str
    mask_path: str
    batch_size: int
    num_workers: int

    # tasks
    det: bool
    ae: bool
    voxel: bool
    voxelsem: List[str]
    path: bool

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
                limit_size=self.limit_size,
            )
        elif self.dataset == Datasets.NUSCENES:
            from torchdrive.datasets.nuscenes_dataset import NuscenesDataset

            dataset = NuscenesDataset(
                data_dir=self.dataset_path,
                version="v1.0-mini" if smoke else "v1.0-trainval",
                lidar=True,
                num_frames=self.num_frames,
            )
        else:
            raise ValueError(f"unknown dataset type {self.dataset}")

        assert set(dataset.cameras) == set(self.cameras)
        return dataset

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
            from torchdrive.models.simple_bev import Segnet3DBackbone

            backbone = Segnet3DBackbone(
                grid_shape=adjusted_grid_shape,
                dim=self.dim,
                hr_dim=self.hr_dim,
                cam_dim=self.cam_dim,
                num_frames=3,
                scale=3 / adjust,
                num_upsamples=self.num_upsamples,
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
                return RegNetEncoder(
                    C=self.cam_dim, regnet=models.regnet_x_800mf(pretrained=True)
                )

        else:
            raise ValueError(f"unknown cam encoder {self.cam_encoder}")

        tasks: Dict[str, BEVTask] = {}
        hr_tasks: Dict[str, BEVTask] = {}
        if self.path:
            from torchdrive.tasks.path import PathTask

            tasks["path"] = PathTask(
                bev_shape=self.bev_shape,
                bev_dim=self.dim,
                dim=self.dim,
                # compile_fn=compile_fn,
            )
        if self.det:
            from torchdrive.tasks.det import DetTask

            tasks["det"] = DetTask(
                cameras=self.cameras,
                cam_shape=self.cam_shape,
                bev_shape=self.bev_shape,
                dim=self.dim,
                device=device,
                compile_fn=compile_fn,
            )
        if self.ae:
            from torchdrive.tasks.ae import AETask

            tasks["ae"] = AETask(
                cameras=self.cameras,
                cam_shape=self.cam_shape,
                bev_shape=self.bev_shape,
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
            )

        model = BEVTaskVan(
            tasks=tasks,
            hr_tasks=hr_tasks,
            cameras=self.cameras,
            dim=self.dim,
            hr_dim=self.hr_dim,
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


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="train")

    parser.add_argument("--output", required=True, type=str, default="out")
    parser.add_argument("--load", type=str)
    parser.add_argument("--skip_load_optim", default=False, action="store_true")
    parser.add_argument("--anomaly-detection", default=False, action="store_true")
    parser.add_argument("--limit_size", type=int)
    parser.add_argument("--checkpoint_every", type=int, default=2000)
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument(
        "--grad_sizes", default=False, action="store_true", help="log grad sizes"
    )
    parser.add_argument(
        "--compile", default=False, action="store_true", help="use torch.compile"
    )
    parser.add_argument("--config", required=True, help="the config file name to use")

    return parser
