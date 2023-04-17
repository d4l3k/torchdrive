import functools
import glob
import io
import logging
import os
import os.path
import random
from collections import defaultdict
from typing import Callable, cast, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import av
import cv2
import numpy as np
import numpy.typing
import orjson
import pytorch3d.transforms
import torch
from av.filter import Graph
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from torchdrive.data import Batch
from torchdrive.transforms.mat import transformation_from_parameters

av.logging.set_level(logging.DEBUG)  # pyre-fixme

KPH_TO_MPS: float = 1000 / (60 * 60)
FPS = 36

SPEED_BINS = [15, 30, 45, 60, 80]
HEADING_BINS = [5, 15, 45, 90, 180]


def compute_bin(v: float, bins: List[int]) -> int:
    for b in bins:
        if v < b:
            return b
    return -1


def bin_weights(bins: Dict[int, int]) -> Dict[int, float]:
    mean = sum(bins.values()) / len(bins)
    return {k: mean / v for k, v in bins.items()}


def cv2_remap(
    img: torch.Tensor, map1: object, map2: object, border_mode: int
) -> Tensor:
    remap_input = img.permute(1, 2, 0).numpy()
    cv_color = cv2.remap(
        remap_input,
        map1,
        map2,
        interpolation=cv2.INTER_CUBIC,
        borderMode=border_mode,
    )
    out = torch.from_numpy(cv_color)
    if len(out.shape) == 2:
        # opencv drops single dimension if present, add it back
        return out.unsqueeze(0)
    return out.permute(2, 0, 1)


def useful_array(
    plane: av.video.plane.VideoPlane, bytes_per_pixel: int = 1, dtype: str = "uint8"
) -> np.typing.NDArray[np.uint8]:
    """
    Return the useful part of the VideoPlane as a single dimensional array.
    We are simply discarding any padding which was added for alignment.
    """
    import numpy as np

    total_line_size = abs(plane.line_size)
    useful_line_size = plane.width * bytes_per_pixel
    arr = np.frombuffer(plane, np.dtype(dtype))
    if total_line_size != useful_line_size:
        arr = arr.reshape(-1, total_line_size)[:, 0:useful_line_size].reshape(-1)
    return arr


def normalize01(tensor: Tensor) -> Tensor:
    return transforms.functional.normalize(
        tensor,
        [0.3504, 0.4324, 0.2892],
        [0.0863, 0.1097, 0.0764],
        inplace=True,
    )


def link_nodes(*nodes: Optional["av.filter.FilterContext"]) -> None:
    # pyre-fixme[9]: Unable to unpack List[FilterContext]
    nodes = [node for node in nodes if node is not None]
    for c, n in zip(nodes, nodes[1:]):
        c.link_to(n)


def heading_diff(a: float, b: float) -> float:
    """
    computes the heading difference between two angles.
    """
    diff = abs(a - b)
    if diff > 180:
        diff = 360 - diff
    return diff


class MultiCamDataset(Dataset):
    CAMERA_OVERLAP = {
        "main": ["narrow", "fisheye"],
        "narrow": ["main", "fisheye"],
        "fisheye": ["main", "narrow", "leftpillar", "rightpillar"],
        "leftpillar": ["leftrepeater", "fisheye"],
        "rightpillar": ["rightrepeater", "fisheye"],
        "leftrepeater": ["backup", "leftpillar"],
        "rightrepeater": ["backup", "rightpillar"],
        "backup": ["leftrepeater", "rightrepeater"],
    }

    def __init__(
        self,
        index_file: str,
        mask_dir: str,
        cameras: List[str],
        cam_shape: Tuple[int, int],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dynamic: bool = False,
        localization: bool = False,
        nframes_per_point: int = 2,
        limit_size: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.frames: List[Tuple[str, int]] = []
        self.dynamic = dynamic
        self.dim: Tuple[int, int] = tuple(reversed(cam_shape))
        self.nframes_per_point = nframes_per_point
        self.dtype = dtype

        self.cameras = cameras

        self.per_path_frame_count: Dict[str, int] = {}

        root = os.path.dirname(index_file)
        indexes = glob.glob(os.path.join(root, "*", "info_noradar.json"))

        self.path_heading_bin: Dict[str, int] = {}
        self.speed_bins: Dict[int, int] = defaultdict(lambda: 0)
        self.heading_bins: Dict[int, int] = defaultdict(lambda: 0)

        DROP_FIRST_N = 10
        DROP_LAST_N = 20 + self.nframes_per_point
        if not dynamic:
            DROP_LAST_N += 30
        MIN_DIST_M = 10

        for path in indexes:
            path = os.path.dirname(path)
            infos = self._get_raw_infos(path, 0, -1)
            if infos is None or len(infos) == 0:
                continue
            frame_counts = [len(infos)]
            try:
                for camera in self.cameras:
                    _, _, offsets, _ = self._load_offsets(path, camera)
                    frame_counts.append(len(offsets))
            except FileNotFoundError:
                continue
            frame_count = min(frame_counts)
            infos = infos[:frame_count]
            self.per_path_frame_count[path] = frame_count

            speeds = []
            for i in range(DROP_FIRST_N, frame_count - DROP_LAST_N):
                info = infos[i]
                speed = info["Speed"]
                if speed < 5 or speed > 80:  # kph
                    continue

                # minimum travel distance
                dist = sum(infos[j]["Speed"] / FPS for j in range(i, frame_count))
                if dist < MIN_DIST_M:
                    continue

                speeds.append(speed)

                self.frames.append((path, i))

            if len(speeds) == 0:
                continue
            heading_change = heading_diff(
                infos[0]["Heading"],
                infos[-1]["Heading"],
            )
            speed = sum(speeds) / len(speeds)
            self.speed_bins[compute_bin(speed, SPEED_BINS)] += 1
            heading_bin = compute_bin(heading_change, HEADING_BINS)
            self.heading_bins[heading_bin] += 1
            self.path_heading_bin[path] = heading_bin

            if limit_size is not None and len(self.frames) > limit_size:
                break

        self.heading_weights: Dict[int, float] = bin_weights(self.heading_bins)
        print("heading_weights", self.heading_weights)

        self.transform = transform
        self.localization = localization

        self._load_masks(mask_dir)

        graph = Graph()
        link_nodes(
            graph.add_buffer(
                height=1280, width=960, format="yuv420p10le", time_base=1 / 1000
            ),
            graph.add("scale", f"{self.dim[0]}:{self.dim[1]}"),
            graph.add("swapuv"),
            graph.add("buffersink"),
        )
        graph.configure()
        self.graph: Graph = graph

    def _load_masks(self, mask_dir: str) -> None:
        # load masks
        self.masks = {}
        for camera in self.cameras:
            mask = Image.open(os.path.join(mask_dir, f"{camera}.png"))
            mask = transforms.functional.to_tensor(mask)
            self.masks[camera] = transforms.functional.resize(
                mask[:1], list(reversed(self.dim))
            )

    def __len__(self) -> int:
        return len(self.frames)

    def _get_calibration(
        self, path: str, cam: str
    ) -> Tuple[torch.Tensor, np.typing.NDArray[np.float32], torch.Tensor]:
        """
        returns K*calibration, fisheye distortion, extrinsic translation
        """
        calibration_path = os.path.join(path, f"field_calibration_{cam}.json")
        with open(calibration_path, "rb") as f:
            data = orjson.loads(f.read())
        intrinsics = data["intrinsics"]
        x_focal_length = intrinsics[0]
        y_focal_length = intrinsics[1]
        K = torch.tensor(
            [
                [x_focal_length / 640, 0, 0.5, 0],
                [0, y_focal_length / 480, 0.5, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float,
        )
        D = np.array([[v] for v in intrinsics[4:8]])

        extrinsics = data["extrinsics"]
        T = torch.tensor(extrinsics[:12] + [0, 0, 0, 1]).reshape((4, 4))

        # pitch, yaw, roll
        calibration = -torch.tensor(extrinsics[12:], dtype=torch.float)
        rot = torch.eye(4)
        rot[:3, :3] = pytorch3d.transforms.euler_angles_to_matrix(
            calibration, convention="XYZ"
        )
        K = K.matmul(rot)

        return K, D, T

    @functools.lru_cache(maxsize=16)  # noqa: B019
    def _load_offsets(
        self, path: str, cam: str
    ) -> Tuple[str, int, Sequence[int], Sequence[int]]:
        index_path = os.path.join(path, f"{cam}_index.csv")
        with open(index_path, "rt") as f:
            first_line = f.readline()
            try:
                file: str = os.path.join(path, f"{cam}.h265")
                start_i = int(first_line)
            except ValueError:
                file: str = os.path.join(path, "../..", first_line.strip())
                start_i = int(f.readline())

            data = np.loadtxt(f, dtype=np.int64, delimiter=" ")
            # pyre-fixme[9]
            offsets: Sequence[int] = data[:, 0]
            # pyre-fixme[9]
            sizes: Sequence[int] = data[:, 1]

        assert len(sizes) == len(offsets)

        return file, start_i, offsets, sizes

    def _nearest_iframe(self, idx: int, start_i: int) -> int:
        iframe = (idx - start_i) // 9 * 9 + start_i
        return iframe

    def _get_frames(
        self,
        path: str,
        cam: str,
        frames: Union[List[int], torch.Tensor],
        dim: Optional[Tuple[int, int]] = None,
    ) -> List[torch.Tensor]:
        num_frames = len(frames)
        assert num_frames > 0, "got zero frames requested"
        if dim is None:
            dim = self.dim
        h265_path, start_i, offsets, sizes = self._load_offsets(path, cam)

        assert max(frames) < len(offsets), f"{frames}, {len(offsets)}"

        frames = list(frames) if isinstance(frames, list) else frames.tolist()
        decode_idxs = []
        cur_frame = -1
        for frame in frames:
            iframe = self._nearest_iframe(frame, start_i)
            if iframe > cur_frame:
                decode_idxs += list(range(iframe, frame + 1))
            else:
                decode_idxs += list(range(cur_frame + 1, frame + 1))
            cur_frame = frame

        with io.BytesIO() as data:
            with open(h265_path, "rb") as f:
                for idx in decode_idxs:
                    offset = offsets[idx]
                    size = sizes[idx]
                    f.seek(offset)
                    buf = f.read(size)
                    assert len(buf) == size
                    data.write(buf)

            data.seek(0)
            nbytes = data.getbuffer().nbytes
            assert nbytes > 0

            out = []
            with av.open(data, format="hevc") as vid:
                # todo seek to each frame
                idx = 0
                for frame in vid.decode():
                    if idx == 0 and not frame.key_frame:
                        raise IndexError(f"{h265_path} first frame not iframe {frame}")
                    frame_idx = decode_idxs[idx]
                    if frame_idx in frames:
                        self.graph.push(frame)
                        frame = self.graph.pull()
                        frame = frame.reformat(format="rgb48le")
                        frame = frame.to_ndarray()
                        frame = (
                            torch.from_numpy(frame.astype(np.float32))
                            .permute(2, 0, 1)
                            .div_(2**16)
                        )
                        out.append(normalize01(frame))
                    idx += 1
                    if idx >= len(decode_idxs):
                        break

        assert len(out) == num_frames, f"{len(out)}, {num_frames}"
        return out

    def _get_rect_frames(
        self, path: str, cam: str, frames: List[int]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns rectified frame, mask and calibrations
        """
        K, D, T = self._get_calibration(path, cam)

        mask = self.masks[cam]

        K = K.clone().numpy()[:3, :3]
        # convert to image space
        K[0] *= self.dim[0]
        K[1] *= self.dim[1]

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, self.dim, np.eye(3), balance=1
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, self.dim, cv2.CV_16SC2
        )

        K = torch.zeros((4, 4))
        K[:3, :3] = torch.tensor(new_K)
        # convert to uv space
        K[0] /= self.dim[0]
        K[1] /= self.dim[1]
        K[2, 2] = 1
        K[3, 3] = 1

        out = []
        for frame in self._get_frames(path, cam, frames):
            if cam == "backup":
                # fill in the black part of the backup camera frame with mean
                # color to avoid screwing up normalization
                maxes = frame.mean(dim=(1, 2))
                frame[:, self.dim[0] // 3 * 2 :, :] = maxes.reshape(3, 1, 1)
            frame = cv2_remap(frame, map1, map2, border_mode=cv2.BORDER_REPLICATE)
            out.append(frame)

        mask = cv2_remap(mask, map1, map2, border_mode=cv2.BORDER_CONSTANT)
        return out, mask, K, T

    def _get_info(self, path: str, idx: int) -> Dict[str, object]:
        info_path = os.path.join(path, f"info_{idx:03d}.json")
        with open(info_path, "rb") as f:
            return orjson.loads(f.read())

    def __getitem__(self, idx: int) -> Optional[Batch]:
        try:
            return self._getitem(idx)
        except RuntimeError as e:
            if "Stream.decode" not in str(e):
                raise
            print(e)
        except av.error.InvalidDataError as e:
            print(e)
        except IndexError as e:
            print(e)
        except av.error.MemoryError as e:
            print(e)

    def _get_alignment(self, path: str) -> Dict[str, int]:
        path = os.path.join(path, "alignment.json")
        with open(path, "rb") as f:
            return orjson.loads(f.read())

    def _get_raw_infos(
        self, path: str, a: int, b: int
    ) -> Optional[List[Dict[str, float]]]:
        info_path = os.path.join(path, "info_noradar.json")
        with open(info_path, "rb") as f:
            info = orjson.loads(f.read())
            if info is None:
                return None
            return info[a:b]

    def _get_infos(self, path: str, a: int, b: int) -> Dict[str, torch.Tensor]:
        fields = {
            "Speed": KPH_TO_MPS,
            "SteeringAngle": 1,
            "SteeringAngleOffset": 1,
            "PitchRate": 1,
            "RollRate": 1,
            "YawRate": 1,
        }
        out = defaultdict(lambda: [])
        raw_infos = self._get_raw_infos(path, a, b)
        assert raw_infos is not None
        for info in raw_infos:
            for k, scale in fields.items():
                out[k].append(info[k] * scale)
        return {k: torch.tensor(v, dtype=torch.float) for k, v in out.items()}

    def _cam_T(self, infos: Dict[str, torch.Tensor]) -> Tuple[Tensor, Tensor]:
        frames = len(infos["Speed"])
        speed = infos["Speed"] / FPS
        roll = infos["RollRate"] / FPS
        pitch = infos["PitchRate"] / FPS
        yaw = infos["YawRate"] / FPS

        translation = torch.zeros(frames, 3, dtype=torch.float)
        translation[:, 0] = speed

        axisangle = torch.stack((roll, pitch, yaw), dim=1)
        assert translation.shape == axisangle.shape, translation.shape

        frame_T = transformation_from_parameters(
            axisangle.unsqueeze(1), translation.unsqueeze(1)
        )

        cam_T = torch.zeros(frames, 4, 4, dtype=torch.float)
        cam_T[0] = torch.eye(4)
        for i in range(1, frames):
            cam_T[i] = torch.matmul(cam_T[i - 1], frame_T[i - 1])
        return cam_T, frame_T

    def _getitem(self, idx: int) -> Batch:
        path: str
        camera: str
        idx: int
        path, idx = self.frames[idx]

        # metadata
        frame_count = self.per_path_frame_count[path]
        infos = self._get_infos(path, idx, frame_count)
        alignment: Mapping[str, int] = self._get_alignment(path)

        speeds = infos["Speed"]
        dists = (speeds / FPS).cumsum(dim=0)

        _, start_i, _, _ = self._load_offsets(path, "main")

        max_dist = 65
        max_idxs = (dists > max_dist).nonzero()
        if max_idxs.numel() == 0:
            max_frame = frame_count - 1
        else:
            max_frame = idx + cast(int, max_idxs[0].item())

        if self.dynamic:
            points = [idx]
        else:
            mid_frame = (idx + max_frame) // 2
            # need to use 10 frame gap to avoid iframe collisions
            far_frame1 = random.randrange(idx + 10, mid_frame - 10)
            far_frame2 = random.randrange(mid_frame, max_frame - 10)

            points = [
                idx,
                self._nearest_iframe(far_frame1, start_i),
                self._nearest_iframe(far_frame2, start_i),
            ]
        frames = []
        for p in points:
            for i in range(self.nframes_per_point):
                frames.append(p + i)

        if len(set(frames)) != len(frames):
            raise RuntimeError(f"duplicate frame index in {frames}")

        cam_Ts, frame_T = self._cam_T(infos)

        info_idxs = [i - idx for i in frames]
        dists = dists[info_idxs]
        cam_T = cam_Ts[info_idxs]
        frame_T = frame_T[info_idxs]
        frame_time = torch.tensor(info_idxs, dtype=torch.float) / 36

        Ks: Dict[str, torch.Tensor] = {}
        Ts: Dict[str, torch.Tensor] = {}
        colors: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}

        def load(cam: str, frames: List[int]) -> None:
            label = cam

            # align camera frames
            cam_alignment = alignment[cam]
            assert cam_alignment >= 0
            frames = [i - cam_alignment for i in frames]

            color, mask, K, T = self._get_rect_frames(path, cam, frames)
            # mask[:, 0:240, :] = 0
            Ks[label] = K
            # out["inv_K", label] = K.pinverse()
            Ts[label] = T
            colors[label] = torch.stack(color).to(self.dtype)
            masks[label] = mask.to(self.dtype)

        for camera in self.cameras:
            load(camera, frames)

        return Batch(
            weight=torch.tensor(self.heading_weights[self.path_heading_bin[path]]),
            K=Ks,
            T=Ts,
            color=colors,
            mask=masks,
            cam_T=cam_T,
            long_cam_T=cam_Ts,
            distances=dists,
            frame_T=frame_T,
            frame_time=frame_time,
        )
