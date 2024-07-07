from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchdrive.amp import autocast
from torchdrive.autograd import autograd_context, register_log_grad_norm
from torchdrive.data import Batch
from torchdrive.losses import losses_backward
from torchdrive.tasks.van import Van

from torchdrive.transforms.batch import NormalizeCarPosition
from torchtune.modules import RotaryPositionalEmbeddings
from torchworld.models.vit import MaskViT
from torchworld.transforms.img import (
    normalize_img,
    normalize_mask,
    render_color,
    render_pca,
)
from torchworld.transforms.mask import random_block_mask, true_mask


class Denoiser(nn.Module):
    """Transformer denoising model for 1d sequences"""

    def __init__(
        self,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        dim: int,
        mlp_dim: int,
        attention_dropout: float,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.positional_embedding = RotaryPositionalEmbeddings(
            dim // num_heads, max_seq_len=max_seq_len
        )

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=attention_dropout,
                batch_first=True,
                layer_norm_eps=1e-6,
            )
        self.layers = nn.Sequential(layers)

    def forward(self, input: torch.Tensor, condition: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        torch._assert(
            condition.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )

        x = input

        # apply rotary embeddings
        # RoPE applies to each head separately
        x = x.unflatten(-1, (self.num_heads, self.dim // self.num_heads))
        x = self.positional_embedding(x)
        x = x.flatten(-2, -1)

        for layer in self.layers:
            x = layer(x, condition)

        return x


class XYEmbedding(nn.Module):
    def __init__(self, shape: Tuple[int, int], scale: float, dim: int):
        """
        Initialize the XYEmbedding

        Arguments:
            shape: the size of the embedding grid [x, y], the center is 0,0
            scale: the max coordinate value
            dim: dimension of the embedding
        """
        super().__init__()

        self.scale = scale
        self.shape = shape

        self.embedding = nn.Parameter(torch.empty(*shape, dim).normal_(std=0.02))

    def forward(self, pos: torch.Tensor):
        """
        Args:
            pos: the list of positions(..., 2)

        Returns:
            the embedding of the position (..., dim)
        """

        dx = (self.shape[0] - 1) // 2
        dy = (self.shape[1] - 1) // 2
        x = (pos[..., 0] * dx / self.scale + dx).long()
        y = (pos[..., 1] * dy / self.scale + dy).long()

        x = x.clamp(min=0, max=self.shape[0] - 1)
        y = y.clamp(min=0, max=self.shape[1] - 1)

        return self.embedding[x, y]

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Convert the embedding back to the position using a cosine similarity distance function.

        Args:
            input: input embedding to decode (bs, seq_len, dim)

        Returns:
            the position (bs, seq_len, 2)
        """

        bs = input.size(0)
        flattened_embedding = self.embedding.flatten(0, 1)

        # (bs, seq_len, dim) @ (x*y, dim) -> (bs, seq_len, x*y)
        similarity = torch.einsum("bsd,xd->bsx", input, flattened_embedding)

        # (bs, seq_len, x*y) -> (bs, seq_len, xy index)
        classes = torch.argmax(similarity, dim=-1)

        # (bs, seq_len, xy index) -> (bs, seq_len, 2)
        x = torch.div(classes, self.shape[1], rounding_mode="floor")
        y = torch.remainder(classes, self.shape[1])

        dx = (self.shape[0] - 1) // 2
        dy = (self.shape[1] - 1) // 2

        x = (x.float() - dx) * self.scale / dx
        y = (y.float() - dy) * self.scale / dy

        # 2x (bs, seq_len) -> (bs, seq_len, 2)
        return torch.stack([x, y], dim=-1)


class XEmbedding(nn.Module):
    def __init__(self, shape: int, scale: float, dim: int):
        """
        Initialize the XEmbedding, which is a linear embedding.

        Arguments:
            shape: the size of the embedding grid [x], the center is 0.0
            scale: the max coordinate value
            dim: dimension of the embedding
        """
        super().__init__()

        self.scale = scale
        self.shape = shape

        self.embedding = nn.Parameter(torch.empty(shape, dim).normal_(std=0.02))

    def _calculate_index(self, pos: torch.Tensor) -> torch.Tensor:
        dx = (self.shape - 1) // 2
        x = (pos * dx / self.scale + dx).long()

        x = x.clamp(min=0, max=self.shape - 1)

        return x

    def forward(self, pos: torch.Tensor):
        """
        Args:
            pos: the list of positions (...)

        Returns:
            the embedding of the position (..., dim)
        """

        x = self._calculate_index(pos)
        return self.embedding[x]

    def _decode_ll(self, input: torch.Tensor) -> torch.Tensor:
        # (bs, seq_len, dim) @ (x, dim) -> (bs, seq_len, x)
        similarity = torch.einsum("bsd,xd->bsx", input, self.embedding)

        return similarity

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Convert the embedding back to the position using a cosine similarity distance function.

        Args:
            input: input embedding to decode (bs, seq_len, dim)

        Returns:
            the position (bs, seq_len)
        """

        similarity = self._decode_ll(input)

        x = similarity.argmax(dim=-1)

        dx = (self.shape - 1) // 2
        x = (x.float() - dx) * self.scale / dx
        return x

    def ae_loss(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute the autoencoder loss for the embedding.

        Args:
            input: input embedding to decode (bs, seq_len, dim)

        Returns:
            the position (bs, seq_len)
        """

        embedding = self(input)

        similarity = self._decode_ll(embedding)
        target = self._calculate_index(input)

        print(similarity.shape, target.shape)

        return F.cross_entropy(similarity.flatten(0, -2), target.flatten())


class XYLinearEmbedding(nn.Module):
    def __init__(self, shape: Tuple[int, int], scale: float, dim: int):
        """
        Initialize the XYLinearEmbedding which is a 2d embedding comprised of two linear XEmbeddings.

        Arguments:
            shape: the size of the embedding grid [x, y], the center is 0.0
            scale: the max coordinate value
            dim: dimension of the embedding (split in 2 for the two child embeddings)
        """
        super().__init__()

        self.dim = dim // 2

        self.x = XEmbedding(shape[0], scale, dim // 2)
        self.y = XEmbedding(shape[1], scale, dim // 2)

    def forward(self, pos: torch.Tensor):
        x = self.x(pos[..., 0])
        y = self.y(pos[..., 1])
        return torch.cat([x, y], dim=-1)

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        x = self.x.decode(input[..., : self.dim])
        y = self.y.decode(input[..., self.dim :])
        return torch.stack([x, y], dim=-1)

    def ae_loss(self, input: torch.Tensor) -> torch.Tensor:
        x = self.x.ae_loss(input[..., 0])
        y = self.y.ae_loss(input[..., 1])
        return x + y


class DiffTraj(nn.Module, Van):
    """
    A diffusion model for trajectory detection.
    """

    def __init__(
        self,
        cameras: List[str],
        cam_shape: Tuple[int, int],
        dim: int = 1024,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        num_layers: int = 24,
        num_heads: int = 16,
        num_encode_frames: int = 1,
        num_frames: int = 1,
    ):
        super().__init__()

        self.cameras = cameras
        self.num_frames = num_frames
        self.num_encode_frames = num_encode_frames
        self.cam_shape = cam_shape
        self.feat_shape = (cam_shape[0] // 16, cam_shape[1] // 16)
        self.encoders = nn.ModuleDict(
            {
                cam: MaskViT(
                    cam_shape=cam_shape,
                    dim=dim,
                    attention_dropout=0.1,
                )
                for cam in cameras
            }
        )

        # embedding
        # 2*100m/512 == 0.39 meters
        self.xy_embedding = XYLinearEmbedding(
            shape=(512, 512),
            scale=100,  # 100 meters
            dim=dim,
        )

        self.denoiser = Denoiser(
            max_seq_len=256,
            num_layers=num_layers,
            num_heads=num_heads,
            dim=dim,
            mlp_dim=dim_feedforward,
            attention_dropout=dropout,
        )

        self.static_features_encoder = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.batch_transform = NormalizeCarPosition(start_frame=0)

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        return [
            {
                "name": "encoders",
                "params": list(self.encoders.parameters()),
                "lr": lr / len(self.encoders),
            },
            {
                "name": "static_features",
                "params": list(self.static_features_encoder.parameters()),
                "lr": lr,
            },
            {
                "name": "denoiser",
                "params": list(self.denoiser.parameters()),
                "lr": lr,
            },
            {
                "name": "xy_embedding",
                "params": list(self.xy_embedding.parameters()),
                "lr": lr,
            },
        ]

    def forward(
        self,
        batch: Batch,
        global_step: int,
        writer: Optional[SummaryWriter] = None,
        output: str = "out",
    ) -> Dict[str, torch.Tensor]:
        batch = self.batch_transform(batch)

        losses = {}

        BS = len(batch.distances)
        device = batch.device()

        log_img, log_text = self.should_log(global_step, BS)

        # for size, device only
        empty_mask = torch.empty(self.feat_shape, device=device)

        all_feats = []

        for cam in self.cameras:
            feats = batch.color[cam][:, : self.num_encode_frames]
            block_size = min(*self.feat_shape) // 3
            mask = random_block_mask(
                empty_mask,
                block_size=(block_size, block_size),
                num_blocks=8,
            )

            if writer is not None and log_text:
                writer.add_scalar(
                    f"{cam}/count",
                    mask.long().sum(),
                    global_step=global_step,
                )
            if writer is not None and log_img:
                writer.add_image(
                    f"{cam}/mask",
                    render_color(mask),
                    global_step=global_step,
                )

            with autocast():
                # checkpoint encoders to save memory
                encoder = self.encoders[cam]
                cam_feats = torch.utils.checkpoint.checkpoint(
                    encoder,
                    feats.flatten(0, 1),
                    mask,
                    use_reentrant=False,
                )
                assert cam_feats.requires_grad, f"missing grad for cam {cam}"

            if writer is not None and log_text:
                register_log_grad_norm(
                    t=cam_feats,
                    writer=writer,
                    key="gradnorm/cam-encoder",
                    tag=cam,
                    global_step=global_step,
                )

            # (n, seq_len, hidden_dim) -> (bs, num_encode_frames, seq_len, hidden_dim)
            cam_feats = cam_feats.unflatten(0, feats.shape[:2])

            if writer is not None and log_img:
                writer.add_image(
                    f"{cam}/pca",
                    render_color(cam_feats[0, 0]),
                    global_step=global_step,
                )

            # flatten time
            # (bs, num_encode_frames, seq_len, hidden_dim) -> (bs, num_encode_frames * seq_len, hidden_dim)
            cam_feats = cam_feats.flatten(1, 2)

            all_feats.append(cam_feats)

        input_tokens = torch.cat(all_feats, dim=1)

        world_to_car, mask, lengths = batch.long_cam_T
        car_to_world = torch.zeros_like(world_to_car)
        car_to_world[mask] = world_to_car[mask].inverse()

        assert mask.int().sum() == lengths.sum(), (mask, lengths)

        zero_coord = torch.zeros(1, 4, device=device, dtype=torch.float)
        zero_coord[:, -1] = 1

        positions = torch.matmul(car_to_world, zero_coord.T).squeeze(-1)
        positions /= positions[..., -1:] + 1e-8  # perspective warp
        positions = positions[..., :2]

        losses["xy_embedding/ae"] = self.xy_embedding(positions)

        velocity = positions[:, 1] - positions[:, 0]
        assert positions.size(-1) == 2
        velocity = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)

        static_features = self.static_features_encoder(velocity)

        lengths = mask.sum(dim=-1)
        pos_len = lengths.amax()

        # we need to be aligned to size 8
        align = 8
        if positions.size(1) % align != 0:
            pad = align - positions.size(1) % align
            mask = F.pad(mask, (0, pad), value=True)
            positions = F.pad(positions, (0, 0, 0, pad), value=0)

        assert positions.size(1) % align == 0
        assert mask.size(1) % align == 0
        assert positions.size(1) == mask.size(1)

        num_elements = mask.float().sum()

        if writer and log_text:
            writer.add_scalar(
                "paths/seq_len",
                pos_len,
                global_step=global_step,
            )
            writer.add_scalar(
                "paths/num_elements",
                num_elements,
                global_step=global_step,
            )

        posmax = positions.abs().amax()
        assert posmax < 1000, positions

        traj_embed = self.xy_embedding(positions)
        # (bs, seq_len, dim) -> (bs, dim, seq_len)
        traj_embed = traj_embed

        noise = torch.randn(traj_embed.shape, device=traj_embed.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (BS,),
            device=traj_embed.device,
            dtype=torch.int64,
        )
        traj_embed = self.noise_scheduler.add_noise(traj_embed, noise, timesteps)

        with autocast():
            pred_noise = self.denoiser(traj_embed, input_tokens)

        noise_loss = F.mse_loss(pred_noise, noise, reduction="none")
        noise_loss = noise_loss[mask]
        losses["diffusion"] = noise_loss.mean()

        if writer and log_img:
            # calculate cross_attn_weights
            with torch.no_grad():
                fig = plt.figure()
                target = positions[0].detach().cpu()
                plt.plot(target[..., 0], target[..., 1], label="target")

            fig.legend()
            plt.gca().set_aspect("equal")
            writer.add_figure(
                "paths/target",
                fig,
                global_step=global_step,
            )

        losses_backward(losses)

        return losses
