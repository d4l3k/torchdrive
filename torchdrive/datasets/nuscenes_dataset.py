import os
from bisect import bisect_left
from datetime import timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, TypedDict

import torch
import torchvision.transforms as transforms
from nuscenes.nuscenes import NuScenes
from PIL import Image
from pytorch3d.transforms import quaternion_to_matrix
from torch.utils.data import ConcatDataset, DataLoader, Dataset

Tensor = torch.Tensor

from torchdrive.data import Batch, collate


class SampleData(TypedDict):
    timestamp: int
    filename: str
    ego_pose_token: str
    calibrated_sensor_token: str


def normalize01(tensor: Tensor) -> Tensor:
    return transforms.functional.normalize(
        tensor,
        [0.3504, 0.4324, 0.2892],
        [0.0863, 0.1097, 0.0764],
        inplace=True,
    )


class CamTypes(str, Enum):
    CAM_FRONT = "CAM_FRONT"
    CAM_FRONT_LEFT = "CAM_FRONT_LEFT"
    CAM_FRONT_RIGHT = "CAM_FRONT_RIGHT"
    CAM_BACK = "CAM_BACK"
    CAM_BACK_LEFT = "CAM_BACK_LEFT"
    CAM_BACK_RIGHT = "CAM_BACK_RIGHT"


NUM_FRAMES = 5


def calculate_timestamp_index(
    cam_samples: Dict[str, List[SampleData]]
) -> Dict[int, Dict[str, Tuple[SampleData, int]]]:
    timestamp_index = {}
    for cam, samples in cam_samples.items():
        for i, sample in enumerate(samples):
            timestamp = sample["timestamp"]
            if timestamp not in timestamp_index:
                timestamp_index[timestamp] = {}
            timestamp_index[timestamp][cam] = (sample, i)
    return timestamp_index


def calculate_nearest_data_within_epsilon(
    cam_samples: Dict[str, List[SampleData]],
    epsilon: int,
    timestamp_index: Dict[int, Dict[str, Tuple[SampleData, int]]],
    sorted_timestamps: List[int],
) -> Dict[int, Tuple[SampleData, Dict[str, int]]]:
    nearest_data_within_epsilon = {}
    for cam_front_timestamp in sorted_timestamps:
        nearest_data = {}
        nearest_data_idxs = {}
        min_timestamp = cam_front_timestamp - epsilon
        max_timestamp = cam_front_timestamp + epsilon

        for cam, samples in cam_samples.items():
            if cam == CamTypes.CAM_FRONT:
                continue

            timestamp_range_start = bisect_left(sorted_timestamps, min_timestamp)
            timestamp_range_end = bisect_left(sorted_timestamps, max_timestamp)

            for i in range(timestamp_range_start, timestamp_range_end):
                timestamp = sorted_timestamps[i]
                sample_data = timestamp_index[timestamp]
                if cam in sample_data:
                    nearest_data[cam], nearest_data_idxs[cam] = sample_data[cam]
                    break

        nearest_data_within_epsilon[cam_front_timestamp] = (
            (nearest_data, nearest_data_idxs)
            if len(nearest_data) == len(cam_samples) - 1
            else (None, None)
        )

    return nearest_data_within_epsilon


class TimestampMatcher:
    def __init__(
        self, cam_samples: Dict[str, List[SampleData]], epsilon: timedelta
    ) -> None:
        self.cam_samples = cam_samples
        self.epsilon = int(epsilon.total_seconds() * 1e6)
        self.timestamp_index: Dict[
            int, Dict[str, Tuple[SampleData, int]]
        ] = calculate_timestamp_index(cam_samples)
        self.sorted_timestamps: List[int] = sorted(self.timestamp_index.keys())
        self.nearest_data_within_epsilon: Dict[
            int, Tuple[SampleData, Dict[str, int]]
        ] = calculate_nearest_data_within_epsilon(
            cam_samples, self.epsilon, self.timestamp_index, self.sorted_timestamps
        )

    def get_nearest_data_within_epsilon(
        self, idx: int
    ) -> Tuple[SampleData, Dict[str, int]]:
        cam_front_samples = self.cam_samples[CamTypes.CAM_FRONT]
        if idx < 0 or idx >= len(cam_front_samples):
            raise IndexError("Index out of range")

        cam_front_timestamp = cam_front_samples[idx]["timestamp"]
        return self.nearest_data_within_epsilon[cam_front_timestamp]


class SceneDataset(Dataset):
    """A "scene" is all the sample data from first (the one with no prev) to last (the one with no next) for a single camera."""

    def __init__(
        self, dataroot: str, nusc: NuScenes, samples: List[SampleData]
    ) -> None:
        self.dataroot = dataroot
        self.nusc = nusc
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples) - NUM_FRAMES - 1

    def _getitem(self, sample_data: SampleData) -> Dict[str, object]:
        # Get world to car translation matrix cam_T
        pose_token = sample_data["ego_pose_token"]
        pose = self.nusc.get("ego_pose", pose_token)
        cam_T = torch.eye(4)
        translation = torch.tensor(pose["translation"])
        cam_T[:3, 3] = translation
        cam_T = cam_T.inverse()  # convert to car_to_world

        # apply rotation matrix
        rotation_mat = torch.eye(4)
        quat = torch.tensor(pose["rotation"])
        rotation = quaternion_to_matrix(quat)
        rotation_mat[:3, :3] = rotation

        cam_T = rotation_mat.inverse().matmul(cam_T)

        timestamp = sample_data["timestamp"]

        # Get the image
        img_path = os.path.join(self.dataroot, sample_data["filename"])  # current image
        img = Image.open(img_path)
        width, height = img.size
        # TODO: Resize to (640, 480) [H, W]
        # img = img.resize((640, 480))
        transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
                normalize01,
                transforms.ConvertImageDtype(
                    torch.bfloat16
                ),  # drop precision to save memory
            ]
        )
        img = transform(img)

        # Get the camera_intrinsic (K)
        calibrated_sensor_token = sample_data["calibrated_sensor_token"]
        calibrated_sensor = self.nusc.get("calibrated_sensor", calibrated_sensor_token)
        camera_intrinsic = torch.tensor(calibrated_sensor["camera_intrinsic"])
        K = torch.eye(4)
        K[:3, :3] = camera_intrinsic
        K[0] /= width
        K[1] /= height

        # Get car to camera local translation matrix T
        rotation = quaternion_to_matrix(torch.tensor(calibrated_sensor["rotation"]))
        translation = calibrated_sensor["translation"]
        T = torch.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = torch.tensor(translation)

        return {
            "weight": 1,
            "distance": 0,  # Must be computed relative to frame #1
            "cam_T": cam_T,
            "frame_T": None,  # Must be computed relative to frame #1
            "frame_time": timestamp,
            "K": K,
            "T": T,
            "color": img,
            "mask": None,
        }

    def __getitem__(self, idx: int) -> Dict[str, object]:
        frame_dicts = []
        for i in range(idx, idx + NUM_FRAMES):
            sample = self.samples[i]
            frame_dict = self._getitem(sample)
            frame_dicts.append(frame_dict)

        dists = [torch.tensor(0)]
        for i in range(1, len(frame_dicts)):
            # Get the relative distance from the prev sample to the curr sample
            curr_fd = frame_dicts[i]
            prev_fd = frame_dicts[i - 1]
            dist = torch.linalg.norm(curr_fd["cam_T"][:3, 3] - prev_fd["cam_T"][:3, 3])
            dists.append(dist)

        cam_Ts = [fd["cam_T"] for fd in frame_dicts]

        frame_Ts = [torch.eye(4)]
        for i in range(1, len(frame_dicts)):
            # Get the relative frame_T for this frame by comparing the current cam_T to the previous cam_T
            frame_T = cam_Ts[i - 1].inverse().matmul(cam_Ts[i])
            # TODO use torch.solve instead
            # frame_T = torch.linalg.solve(cam_Ts[i], cam_Ts[i - 1])[0]
            frame_Ts.append(frame_T)

        imgs = [fd["color"] for fd in frame_dicts]

        return {
            "weight": torch.tensor(frame_dicts[0]["weight"]),
            "distance": torch.stack(dists),
            "cam_T": torch.stack(cam_Ts),
            "frame_T": torch.stack(frame_Ts),
            "frame_time": torch.tensor([fd["frame_time"] for fd in frame_dicts]),
            "K": frame_dicts[0][
                "K"
            ],  # only one cam to pixel matrix is required as it doesn't change during drive
            "T": frame_dicts[0][
                "T"
            ],  # only one cam to car translation matrix is required as it doesn't change during drive
            "color": torch.stack(imgs),
            "mask": torch.ones(1, 480, 640),
        }


class NuscenesDataset(Dataset):
    CAMERA_OVERLAP: Dict[str, List[str]] = {
        CamTypes.CAM_FRONT: [CamTypes.CAM_FRONT_LEFT, CamTypes.CAM_FRONT_RIGHT],
        CamTypes.CAM_FRONT_LEFT: [CamTypes.CAM_FRONT, CamTypes.CAM_BACK_LEFT],
        CamTypes.CAM_FRONT_RIGHT: [CamTypes.CAM_FRONT, CamTypes.CAM_BACK_RIGHT],
        CamTypes.CAM_BACK: [CamTypes.CAM_BACK_LEFT, CamTypes.CAM_BACK_RIGHT],
        CamTypes.CAM_BACK_LEFT: [CamTypes.CAM_BACK, CamTypes.CAM_FRONT_LEFT],
        CamTypes.CAM_BACK_RIGHT: [CamTypes.CAM_BACK, CamTypes.CAM_FRONT_RIGHT],
    }

    def __init__(self, data_dir: str, version: str = "v1.0-trainval") -> None:
        self.data_dir = data_dir
        self.version = version
        self.nusc = NuScenes(version=version, dataroot=data_dir, verbose=True)
        self.cam_types: List[str] = [
            CamTypes.CAM_FRONT,
            CamTypes.CAM_FRONT_LEFT,
            CamTypes.CAM_FRONT_RIGHT,
            CamTypes.CAM_BACK,
            CamTypes.CAM_BACK_LEFT,
            CamTypes.CAM_BACK_RIGHT,
        ]

        # Organize all the sample_data into scenes by camera type
        self.cam_scenes: Dict[str, ConcatDataset] = {}
        self.cam_samples: Dict[str, List[SampleData]] = {}
        for cam in self.cam_types:
            self.cam_scenes[cam], self.cam_samples[cam] = self._cam2scenes(cam)
            print(f"Found {len(self.cam_scenes[cam])} scenes for {cam}")
            print(f"Found {len(self.cam_samples[cam])} samples for {cam}")

        # Create a timestamp matcher to match the timestamps of each camera (addresses the issue of cameras being out of sync)
        self.timestamp_matcher = TimestampMatcher(
            self.cam_samples, epsilon=timedelta(milliseconds=51)
        )

    def _cam2scenes(
        self, cam: str
    ) -> Tuple[ConcatDataset[SceneDataset], List[Dict[str, object]]]:
        """Takes in a camera, returns the data split up into SceneDatasets of information."""
        # Get all the sample_data with no prev
        starts = [
            sd
            for sd in self.nusc.sample_data
            if sd["channel"] == cam and sd["prev"] == ""
        ]

        assert len(starts) > 0, f"No starts found for {cam}"

        # For each start, follow the .next until there are no more. Store these samples in a SceneDataset.
        ds_scenes, scenes = [], []
        for start in starts:
            samples = []
            sample_data = start
            while sample_data["next"] != "":
                samples.append(sample_data)
                sample_data = self.nusc.get("sample_data", sample_data["next"])
            samples.append(sample_data)

            scenes.append(samples)
            ds_scenes.append(SceneDataset(self.data_dir, self.nusc, samples))

        scene_samples = [sample for scene in scenes for sample in scene]
        ds = ConcatDataset(ds_scenes)
        return ds, scene_samples

    def __len__(self) -> int:
        return len(self.cam_scenes[CamTypes.CAM_FRONT])

    def _getitem(self, idx: int) -> Optional[Batch]:
        """Returns one row of a Batch of data for the given index."""
        # Do timestamp matching for this idx
        sample_data, idxs = self.timestamp_matcher.get_nearest_data_within_epsilon(
            idx
        )  # Returns { cam: sample_data, ... } for all cams except CAM_FRONT
        if sample_data is None:
            return None

        # Now get processed sample data using SceneDatasets from the cam_scenes
        data = {}
        for cam in idxs:
            adj_idx = idxs[cam]
            data[cam] = self.cam_scenes[cam][adj_idx]
        data[CamTypes.CAM_FRONT] = self.cam_scenes[CamTypes.CAM_FRONT][idx]

        weight: Tensor = data[CamTypes.CAM_FRONT]["weight"]
        distances: Tensor = data[CamTypes.CAM_FRONT]["distance"]
        cam_Ts: Tensor = data[CamTypes.CAM_FRONT]["cam_T"]
        frame_Ts: Tensor = data[CamTypes.CAM_FRONT]["frame_T"]
        frame_times: Tensor = data[CamTypes.CAM_FRONT]["frame_time"]
        Ks: Dict[str, Tensor] = {}
        Ts: Dict[str, Tensor] = {}
        colors: Dict[str, Tensor] = {}
        masks: Dict[str, Tensor] = {}

        for cam in self.cam_types:
            sample_dict = data[cam]
            Ks[cam] = sample_dict["K"]
            Ts[cam] = sample_dict["T"]
            cam_colors = []
            for i in range(NUM_FRAMES):
                cam_colors.append(sample_dict["color"][i])
            colors[cam] = torch.stack(cam_colors, dim=0)
            masks[cam] = sample_dict["mask"]

        return Batch(
            weight=weight.float(),
            distances=distances,
            cam_T=cam_Ts,
            frame_T=frame_Ts,
            frame_time=frame_times,
            K=Ks,
            T=Ts,
            color=colors,
            mask=masks,
            long_cam_T=cam_Ts,
        )

    def __getitem__(self, idx: int) -> Optional[Batch]:
        """Aggregates the data from each camera type into an element of a Batch."""
        return self._getitem(idx)


if __name__ == "__main__":
    import sys

    dataroot: str = sys.argv[-1]
    version: str = "v1.0-mini"
    ds = NuscenesDataset(dataroot, version=version)
    dl: DataLoader = DataLoader(
        ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate
    )
    for batch in dl:
        torch.save(batch, "nuscenes_batch.pt")
        break
