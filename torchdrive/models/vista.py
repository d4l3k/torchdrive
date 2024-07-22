import torch

from omegaconf import ListConfig, OmegaConf

from torchdrive.data import collate
from torchdrive.transforms.batch import NormalizeCarPosition
from torchdrive.datasets.nuscenes_dataset import NuscenesDataset

from vwm.sample_utils import load_model_from_config, init_sampling, do_sample

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
    ) -> None:
        self.cond_aug = cond_aug
        self.num_frames = num_frames

        config = OmegaConf.load(config_path)
        model = load_model_from_config(config, ckpt_path)
        self.model = model.bfloat16().to(device)

        guider = "VanillaCFG"
        self.sampler = init_sampling(
            guider=guider, 
            steps=steps, 
            cfg_scale=cfg_scale, 
            num_frames=num_frames,
        )
        
    def generate(self, cond_img: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
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

        assert cond_img.size(0) == 1

        unique_keys = set([x.input_key for x in model.conditioner.embedders])

        value_dict = init_embedder_options(unique_keys)
        value_dict["cond_frames_without_noise"] = cond_img
        value_dict["cond_aug"] = self.cond_aug
        value_dict["cond_frames"] = cond_img + self.cond_aug * torch.randn_like(cond_img, device=device)
        value_dict["trajectory"] = trajectory.squeeze(0)[1:5].flatten()

        uc_keys = ["cond_frames", "cond_frames_without_noise", "command", "trajectory", "speed", "angle", "goal"]

        images = cond_img.expand(self.num_frames, -1, -1, -1)

        out = do_sample(
            images,
            self.model,
            self.sampler,
            value_dict,
            num_rounds=1,
            num_frames=self.num_frames,
            force_uc_zero_embeddings=uc_keys,
            initial_cond_indices=[0], # only condition on first frame
        )
        samples, samples_z, inputs = out


if __name__ == "__main__":
    dataset = NuscenesDataset(
        data_dir="~/nuscenes",
        version="v1.0-mini",
        lidar=False,
        num_frames=1,
    )

    sample = dataset[0]
    batch = collate([sample])


    transform = NormalizeCarPosition(start_frame=0)
    batch = transform(batch)

    trajectory = batch.positions()
    # down sample to 0.5s resolution 12 hz
    trajectory = trajectory[:, ::6, :]

    sampler = VistaSampler()
    out = sampler.generate(cond_img, trajectory)
    print(out.shape)
    print(out)
