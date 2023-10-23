import os
import time
from bisect import bisect_left
from datetime import timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, TypedDict

import orjson

import pandas as pd
import pyarrow as pa
import torch
import torchvision.transforms as transforms
from nuscenes.nuscenes import NuScenes as UnpatchedNuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
from pytorch3d.transforms import quaternion_to_matrix
from torch.utils.data import ConcatDataset, DataLoader, Dataset as TorchDataset
from tqdm import tqdm

from torchdrive.datasets.dataset import Dataset, Datasets

Tensor = torch.Tensor

from torchdrive.data import Batch, collate


def as_str(data: object) -> str:
    if isinstance(data, str):
        return data
    elif isinstance(data, pa.StringScalar):
        return data.as_py()
    else:
        raise TypeError(f"unknown type of {data}")


def as_num(data: object) -> str:
    if isinstance(data, (int, float)):
        return data
    elif isinstance(data, pa.Int64Scalar):
        return data.as_py()
    else:
        raise TypeError(f"unknown type of {data}")


def as_py(data: object) -> object:
    if hasattr(data, "as_py"):
        return data.as_py()
    return data


# NuScenes = UnpatchedNuScenes
class NuScenes(UnpatchedNuScenes):
    def __init__(
        self,
        version: str = "v1.0-mini",
        dataroot: str = "/data/sets/nuscenes",
        verbose: bool = True,
        map_resolution: float = 0.1,
    ):
        super().__init__(version, dataroot, verbose, map_resolution)

        for table in self.table_names:
            if table == "map":
                continue
            setattr(self, table, pa.array(getattr(self, table)))

    def __load_table__(self, table_name) -> dict:
        """Loads a table."""
        with open(os.path.join(self.table_root, f"{table_name}.json")) as f:
            table = orjson.loads(f.read())
        return table

    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.
        :param verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            index = {}

            for ind, member in enumerate(getattr(self, table)):
                index[member["token"]] = ind

            self._token2ind[table] = pd.Series(index.values(), index=index.keys())

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get("calibrated_sensor", record["calibrated_sensor_token"])
            sensor_record = self.get("sensor", cs_record["sensor_token"])
            record["sensor_modality"] = sensor_record["modality"]
            record["channel"] = sensor_record["channel"]

    def get(self, table_name: str, token: str) -> dict:
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, "Table {} not found".format(table_name)

        return getattr(self, table_name)[self.getind(table_name, as_str(token))]

    def getind(self, table_name: str, token: str) -> int:
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        return self._token2ind[table_name][token]


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


class SensorTypes(str, Enum):
    LIDAR_TOP = "LIDAR_TOP"


def calculate_timestamp_index(
    cam_samples: Dict[str, List[SampleData]]
) -> Dict[int, Dict[str, Tuple[SampleData, int]]]:
    timestamp_index = {}
    for cam, samples in cam_samples.items():
        for i, sample in enumerate(samples):
            timestamp = as_num(sample["timestamp"])
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
        timestamp_index: Dict[
            int, Dict[str, Tuple[SampleData, int]]
        ] = calculate_timestamp_index(cam_samples)
        sorted_timestamps: List[int] = sorted(timestamp_index.keys())
        self.nearest_data_within_epsilon: Dict[
            int, Tuple[SampleData, Dict[str, int]]
        ] = calculate_nearest_data_within_epsilon(
            cam_samples, self.epsilon, timestamp_index, sorted_timestamps
        )

    def get_nearest_data_within_epsilon(
        self, idx: int
    ) -> Tuple[SampleData, Dict[str, int]]:
        cam_front_samples = self.cam_samples[CamTypes.CAM_FRONT]
        if idx < 0 or idx >= len(cam_front_samples):
            raise IndexError("Index out of range")

        cam_front_timestamp = as_num(cam_front_samples[idx]["timestamp"])
        return self.nearest_data_within_epsilon[cam_front_timestamp]


def get_ego_T(nusc: NuScenes, sample_data: SampleData) -> torch.Tensor:
    """
    Get world to car translation matrix cam_T

    Returns [4, 4]
    """
    pose_token = sample_data["ego_pose_token"]
    pose = nusc.get("ego_pose", pose_token)
    cam_T = torch.eye(4)
    translation = torch.tensor(as_py(pose["translation"]))
    cam_T[:3, 3] = translation
    # cam_T = cam_T.inverse()  # convert to car_to_world

    # apply rotation matrix
    rotation_mat = torch.eye(4)
    quat = torch.tensor(as_py(pose["rotation"]))
    rotation = quaternion_to_matrix(quat)
    rotation_mat[:3, :3] = rotation

    # cam_T = rotation_mat.inverse().matmul(cam_T)
    return cam_T.matmul(rotation_mat).inverse()


def get_sensor_calibration_K(
    nusc: NuScenes, sample_data: SampleData, width: int, height: int
) -> torch.Tensor:
    """
    Camera intrinsic matrix.
    """

    calibrated_sensor_token = sample_data["calibrated_sensor_token"]
    calibrated_sensor = nusc.get("calibrated_sensor", calibrated_sensor_token)
    camera_intrinsic = torch.tensor(as_py(calibrated_sensor["camera_intrinsic"]))
    K = torch.eye(4)
    K[:3, :3] = camera_intrinsic
    K[0] /= width
    K[1] /= height

    return K


def get_sensor_calibration_T(nusc: NuScenes, sample_data: SampleData) -> torch.Tensor:
    """
    Get car to camera local translation matrix T
    """
    calibrated_sensor_token = sample_data["calibrated_sensor_token"]
    calibrated_sensor = nusc.get("calibrated_sensor", calibrated_sensor_token)

    rotation = quaternion_to_matrix(torch.tensor(as_py(calibrated_sensor["rotation"])))
    translation = as_py(calibrated_sensor["translation"])
    rot_T = torch.eye(4)
    rot_T[:3, :3] = rotation
    trans_T = torch.eye(4)
    trans_T[:3, 3] = torch.tensor(translation)

    return trans_T.matmul(rot_T)


class CameraDataset(TorchDataset):
    """A "scene" is all the sample data from first (the one with no prev) to last (the one with no next) for a single camera."""

    def __init__(
        self,
        dataroot: str,
        nusc: NuScenes,
        samples: List[SampleData],
        num_frames: int,
    ) -> None:
        self.dataroot = dataroot
        self.nusc = nusc
        self.samples = samples
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.samples) - self.num_frames - 1

    def _getitem(self, sample_data: SampleData) -> Dict[str, object]:
        cam_T = get_ego_T(self.nusc, sample_data)
        timestamp = as_py(sample_data["timestamp"])

        # Get the image
        img_path = os.path.join(
            self.dataroot, as_str(sample_data["filename"])
        )  # current image
        img = Image.open(img_path)
        width, height = img.size

        # this is fine since K is normalized to 1.0
        img = img.resize((640, 480))

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

        T = get_sensor_calibration_T(self.nusc, sample_data)
        K = get_sensor_calibration_K(self.nusc, sample_data, width=width, height=height)

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
            "token": sample_data["sample_token"],
        }

    def __getitem__(self, idx: int) -> Dict[str, object]:
        frame_dicts = []
        for i in range(idx, idx + self.num_frames):
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
            frame_T = torch.linalg.solve(cam_Ts[i - 1], cam_Ts[i])
            # TODO use torch.solve instead
            # frame_T = torch.linalg.solve(cam_Ts[i], cam_Ts[i - 1])[0]
            frame_Ts.append(frame_T)

        imgs = [fd["color"] for fd in frame_dicts]
        token = [fd["token"] for fd in frame_dicts]

        # timestamp is in microseconds, need to convert it to seconds and
        # normalize to first frame
        frame_time = torch.tensor(
            [fd["frame_time"] for fd in frame_dicts], dtype=torch.int64
        )
        frame_time = frame_time - frame_time[0]
        frame_time = frame_time.float() / 1e6

        long_cam_Ts = [
            get_ego_T(self.nusc, sample_data) for sample_data in self.samples[idx:]
        ]

        return {
            "weight": torch.tensor(frame_dicts[0]["weight"]),
            "distance": torch.stack(dists),
            "cam_T": torch.stack(cam_Ts),
            "frame_T": torch.stack(frame_Ts),
            "long_cam_T": torch.stack(long_cam_Ts),
            "frame_time": frame_time,
            "K": frame_dicts[0][
                "K"
            ],  # only one cam to pixel matrix is required as it doesn't change during drive
            "T": frame_dicts[0][
                "T"
            ],  # only one cam to car translation matrix is required as it doesn't change during drive
            "color": torch.stack(imgs),
            "mask": torch.ones(1, 480, 640),
            "token": token,
        }


class LidarDataset(TorchDataset):
    def __init__(
        self, dataroot: str, nusc: NuScenes, samples: List[SampleData], num_frames: int
    ) -> None:
        self.dataroot = dataroot
        self.nusc = nusc
        self.samples = samples
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.samples) - self.num_frames - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Account for FPS difference between cameras (15ps) and lidar
        # (20fps)
        idx += self.num_frames // 2 * 15 // 20
        sample_data = self.samples[idx]

        calibrated_sensor_token = sample_data["calibrated_sensor_token"]
        calibrated_sensor = self.nusc.get("calibrated_sensor", calibrated_sensor_token)

        ego_T = get_ego_T(self.nusc, sample_data)
        sensor_T = get_sensor_calibration_T(self.nusc, sample_data)

        pcl = LidarPointCloud.from_file(
            os.path.join(self.dataroot, as_str(sample_data["filename"]))
        )
        return torch.from_numpy(pcl.points), sensor_T.matmul(ego_T)


class NuscenesDataset(Dataset):
    NAME = Datasets.NUSCENES
    CAMERA_OVERLAP: Dict[str, List[str]] = {
        CamTypes.CAM_FRONT: [CamTypes.CAM_FRONT_LEFT, CamTypes.CAM_FRONT_RIGHT],
        CamTypes.CAM_FRONT_LEFT: [CamTypes.CAM_FRONT, CamTypes.CAM_BACK_LEFT],
        CamTypes.CAM_FRONT_RIGHT: [CamTypes.CAM_FRONT, CamTypes.CAM_BACK_RIGHT],
        CamTypes.CAM_BACK: [CamTypes.CAM_BACK_LEFT, CamTypes.CAM_BACK_RIGHT],
        CamTypes.CAM_BACK_LEFT: [CamTypes.CAM_BACK, CamTypes.CAM_FRONT_LEFT],
        CamTypes.CAM_BACK_RIGHT: [CamTypes.CAM_BACK, CamTypes.CAM_FRONT_RIGHT],
    }
    cameras: List[str] = list(CamTypes)

    def __init__(
        self,
        data_dir: str,
        num_frames: int,
        version: str = "v1.0-trainval",
        lidar: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.version = version
        self.nusc = NuScenes(version=version, dataroot=data_dir, verbose=True)
        self.num_frames = num_frames
        self.cam_types: List[str] = [
            CamTypes.CAM_FRONT,
            CamTypes.CAM_FRONT_LEFT,
            CamTypes.CAM_FRONT_RIGHT,
            CamTypes.CAM_BACK,
            CamTypes.CAM_BACK_LEFT,
            CamTypes.CAM_BACK_RIGHT,
        ]
        self.sensor_types = self.cam_types
        if lidar:
            self.sensor_types = self.sensor_types + [SensorTypes.LIDAR_TOP]
        self.lidar = lidar
        self.cameras: List[str] = list(self.CAMERA_OVERLAP.keys())

        # Organize all the sample_data into scenes by camera type
        self.cam_scenes: Dict[str, ConcatDataset] = {}
        self.cam_samples: Dict[str, List[SampleData]] = {}
        for cam in self.sensor_types:
            self.cam_scenes[cam], self.cam_samples[cam] = self._cam2scenes(cam)
            print(f"Found {len(self.cam_scenes[cam])} scenes for {cam}")
            print(f"Found {len(self.cam_samples[cam])} samples for {cam}")

        # Create a timestamp matcher to match the timestamps of each camera (addresses the issue of cameras being out of sync)
        self.timestamp_matcher = TimestampMatcher(
            self.cam_samples, epsilon=timedelta(milliseconds=51)
        )

    def _cam2scenes(
        self, cam: str
    ) -> Tuple[ConcatDataset[CameraDataset], List[Dict[str, object]]]:
        """Takes in a camera, returns the data split up into CameraDatasets of information."""
        # Get all the sample_data with no prev
        starts = [
            sd
            for sd in self.nusc.sample_data
            if as_str(sd["channel"]) == cam and as_str(sd["prev"]) == ""
        ]

        assert len(starts) > 0, f"No starts found for {cam}"

        # For each start, follow the .next until there are no more. Store these samples in a CameraDataset.
        ds_scenes, scenes = [], []
        for start in starts:
            samples = []
            sample_data = start
            while as_str(sample_data["next"]) != "":
                samples.append(sample_data)
                sample_data = self.nusc.get("sample_data", sample_data["next"])
            samples.append(sample_data)

            scenes.append(samples)
            sensor_modality = as_str(sample_data["sensor_modality"])
            if sensor_modality == "camera":
                kls = CameraDataset
            elif sensor_modality == "lidar":
                kls = LidarDataset
            else:
                raise RuntimeError(f"unsupported sensor_modality {sensor_modality}")
            samples = pa.array(samples)
            ds_scenes.append(
                kls(self.data_dir, self.nusc, samples, num_frames=self.num_frames)
            )

        scene_samples = pa.array([sample for scene in scenes for sample in scene])
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

        # Now get processed sample data using CameraDatasets from the cam_scenes
        data = {}
        for cam, adj_idx in idxs.items():
            cam_scene = self.cam_scenes[cam]
            if adj_idx < 0 or adj_idx >= len(cam_scene):
                # TODO: figure out why index is invalid
                return None
            data[cam] = cam_scene[adj_idx]
        data[CamTypes.CAM_FRONT] = self.cam_scenes[CamTypes.CAM_FRONT][idx]

        token: Tensor = data[CamTypes.CAM_FRONT]["token"]
        weight: Tensor = data[CamTypes.CAM_FRONT]["weight"]
        distances: Tensor = data[CamTypes.CAM_FRONT]["distance"]
        cam_Ts: Tensor = data[CamTypes.CAM_FRONT]["cam_T"]
        long_cam_Ts: Tensor = data[CamTypes.CAM_FRONT]["long_cam_T"]
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
            for i in range(self.num_frames):
                cam_colors.append(sample_dict["color"][i])
            colors[cam] = torch.stack(cam_colors, dim=0)
            masks[cam] = sample_dict["mask"]

        if self.lidar:
            lidar, lidar_T = data[SensorTypes.LIDAR_TOP]
            lidar_T = cam_Ts[0].matmul(lidar_T.inverse()).inverse()
        else:
            lidar = None
            lidar_T = torch.eye(4)

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
            long_cam_T=long_cam_Ts,
            lidar=lidar,
            lidar_T=lidar_T,
            token=[token],
        )

    def __getitem__(self, idx: int) -> Optional[Batch]:
        """Aggregates the data from each camera type into an element of a Batch."""
        return self._getitem(idx)


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1]
    dataroot: str = sys.argv[2]
    version: str = sys.argv[3]  # "v1.0-mini"
    if cmd == "single":
        ds = NuscenesDataset(dataroot, version=version)
        dl: DataLoader = DataLoader(
            ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate
        )
        for batch in dl:
            torch.save(batch, "nuscenes_batch.pt")
            print(batch)
            break
    elif cmd == "bulk":
        ds = NuscenesDataset(dataroot, version=version)
        for batch in tqdm(ds):
            pass
    else:
        raise ValueError(f"unknown command {cmd}")
