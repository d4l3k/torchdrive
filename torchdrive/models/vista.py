import os.path
from typing import Tuple
import time

import torch
import torch.nn.functional as F

from omegaconf import ListConfig, OmegaConf
from torchvision.transforms.functional import to_pil_image
from torchworld.transforms.img import normalize_img

from vwm.sample_utils import (
    do_sample,
    init_embedder_options,
    init_sampling,
    load_model_from_config,
)

from torchdrive.data import collate
from torchdrive.datasets.nuscenes_dataset import NuscenesDataset
from torchdrive.transforms.batch import NormalizeCarPosition


class VistaSampler:
    def __init__(
        self,
        config_path: str = "~/Vista/configs/inference/vista.yaml",
        ckpt_path: str = "~/Vista/ckpts/vista.safetensors",
        device: str = "cuda",
        steps: int = 50,
        cfg_scale: float = 2.5,
        num_frames: int = 10,
        cond_aug: float = 0.0,
        render_size: Tuple[int, int] = (320, 576),
    ) -> None:
        """
        Args:
            config_path: path to the config file
            ckpt_path: path to the checkpoint file
            device: device to run inference on
            steps: number of diffusion steps
            cfg_scale: scale of the config
            num_frames: number of frames to generate
                NOTE Vista is trained at 10hz and Nuscenes is 12hz
            cond_aug: augmentation strength (extra noise)
            render_size: size of the image passed into Vista
        """
        config_path = os.path.expanduser(config_path)
        ckpt_path = os.path.expanduser(ckpt_path)

        self.cond_aug = cond_aug
        self.num_frames = num_frames
        self.render_size = render_size

        start = time.perf_counter()

        config = OmegaConf.load(config_path)
        model = load_model_from_config(config, ckpt_path)
        self.model = model.bfloat16().to(device).eval()

        print(f"loaded vista in {time.perf_counter() - start:.2f}s")

        guider = "VanillaCFG"
        self.sampler = init_sampling(
            guider=guider,
            steps=steps,
            cfg_scale=cfg_scale,
            num_frames=num_frames,
        )

    def generate(
        self, cond_img: torch.Tensor, trajectory: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates the next num_frames prediction.

        Args:
            cond_img: (1, 3, H, W)
                the list of positions
                Should be -1 to 1 value range
                320x576 or 576x1024
            trajectory: (1, 5, 2)
                trajectory including start position at (0, 0)
                (x, y) -- x+ is forward
                meters
                every 0.5s
        """
        device = cond_img.device

        # TODO: support conditioning on multiple frames
        assert cond_img.size(0) == 1
        h, w = cond_img.shape[2:]

        assert trajectory.size(-1) == 2
        # downsample to 4 frames or 2 seconds
        trajectory = trajectory.squeeze(0)[1:5]
        # switch axis so y+ is forward, and x+ is right
        trajectory = torch.stack([-trajectory[:, 1], trajectory[:, 0]], dim=-1)
        trajectory = trajectory.flatten()
        assert trajectory.shape == (8,)

        cond_img = F.interpolate(cond_img, size=self.render_size, mode="bilinear")

        amin, amax = cond_img.aminmax()
        center = (amax + amin) / 2
        dist = (amax - amin) / 2

        # recenter to (-1, 1)
        cond_img = (cond_img - center) / dist

        unique_keys = set([x.input_key for x in self.model.conditioner.embedders])

        value_dict = init_embedder_options(unique_keys)
        value_dict["cond_frames_without_noise"] = cond_img
        value_dict["cond_aug"] = self.cond_aug
        value_dict["cond_frames"] = cond_img + self.cond_aug * torch.randn_like(
            cond_img, device=device
        )
        value_dict["trajectory"] = trajectory

        uc_keys = [
            "cond_frames",
            "cond_frames_without_noise",
            "command",
            "trajectory",
            "speed",
            "angle",
            "goal",
        ]

        images = cond_img.expand(self.num_frames, -1, -1, -1)

        out = do_sample(
            images,
            self.model,
            self.sampler,
            value_dict,
            num_rounds=1,
            num_frames=self.num_frames,
            force_uc_zero_embeddings=uc_keys,
            initial_cond_indices=[0],  # only condition on first frame
        )
        samples, samples_z, inputs = out

        out_min, out_max = samples.aminmax()
        out_center = (out_max + out_min) / 2
        out_dist = (out_max - out_min) / 2

        # restore original range
        samples = (samples - out_center) / out_dist * dist + center

        return F.interpolate(samples, (h, w), mode="bilinear")


if __name__ == "__main__":
    dataset = NuscenesDataset(
        data_dir="~/nuscenes",
        version="v1.0-mini",
        lidar=False,
        num_frames=1,
    )
    device = torch.device("cuda")

    sample = dataset[0]
    batch = collate([sample]).to(device)

    transform = NormalizeCarPosition(start_frame=0)
    batch = transform(batch)

    trajectory = batch.positions()
    # down sample to 0.5s resolution 12 hz
    trajectory = trajectory[:, ::6, ::2]
    cond_img = batch.color["CAM_FRONT"][:1, 0]

    print(trajectory)

    sampler = VistaSampler(device=device)
    out = sampler.generate(cond_img, trajectory)
    print(out.shape)
    assert out.shape == (10, 3, 480, 640)

    for i, img in enumerate(out):
        img = to_pil_image(normalize_img(img))
        img.save(f"vista_{i}.png")
