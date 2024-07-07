import os.path
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from diffusers import EulerDiscreteScheduler
from safetensors.torch import load_model
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchdrive.amp import autocast
from torchdrive.autograd import autograd_context, register_log_grad_norm
from torchdrive.data import Batch
from torchdrive.debug import assert_not_nan, is_nan
from torchdrive.losses import losses_backward
from torchdrive.models.mlp import MLP
from torchdrive.models.path import XYEncoder
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


def square_mask(mask: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Create a squared mask from a sequence mask.

    Arguments:
        mask: the sequence mask (bs, seq_len)
        num_heads: the number of heads

    Returns:
        the squared mask (bs*num_heads, seq_len, seq_len)
    """

    bs, seq_len = mask.shape

    # (bs, seq_len) -> (bs, 1, seq_len)
    x = mask.unsqueeze(1)
    # (bs, 1, seq_len) -> (bs, seq_len, seq_len)
    x = x.expand(-1, seq_len, seq_len)

    # (bs, seq_len) -> (bs, seq_len, 1)
    y = mask.unsqueeze(2)
    # (bs, seq_len, 1) -> (bs, seq_len, seq_len)
    y = y.expand(-1, seq_len, seq_len)

    mask = torch.logical_and(x, y).repeat(num_heads, 1, 1)

    diagonal = torch.arange(seq_len, device=mask.device)
    mask[:, diagonal, diagonal] = True
    return mask


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

    def forward(
        self, input: torch.Tensor, input_mask: torch.Tensor, condition: torch.Tensor
    ):
        torch._assert(
            input_mask.dim() == 2,
            f"Expected (batch_size, seq_length) got {input_mask.shape}",
        )
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        torch._assert(
            condition.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {condition.shape}",
        )

        x = input

        # apply rotary embeddings
        # RoPE applies to each head separately
        x = x.unflatten(-1, (self.num_heads, self.dim // self.num_heads))
        x = self.positional_embedding(x)
        x = x.flatten(-2, -1)

        attn_mask = square_mask(input_mask, num_heads=self.num_heads)
        # True values are ignored so need to flip the mask
        attn_mask = torch.logical_not(attn_mask)

        for i, layer in enumerate(self.layers):
            x = layer(tgt=x, tgt_mask=attn_mask, memory=condition)

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


class XYMLPEncoder(nn.Module):
    def __init__(
        self, dim: int, max_dist: float, dropout: float = 0.1, pretrained: bool = False
    ) -> None:
        super().__init__()

        self.embedding = XYEncoder(num_buckets=dim // 2, max_dist=max_dist)
        self.encoder = MLP(dim, dim, dim, num_layers=3, dropout=dropout)
        self.decoder = MLP(dim, dim, dim, num_layers=3, dropout=dropout)

        if pretrained:
            path = os.path.join(
                os.path.dirname(__file__),
                "../../data/xy_mlp_vae.safetensors",
            )
            print(f"loading {path}")
            load_model(self, path)

            for param in self.embedding.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xy: the list of positions (..., 2)

        Returns:
            the embedding of the position (..., dim)
        """
        xy = xy.permute(0, 2, 1)
        one_hot = self.embedding.encode_one_hot(xy)
        return self.encoder(one_hot).permute(0, 2, 1)

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        emb = self.decoder(input.permute(0, 2, 1))
        xy = self.embedding.decode(emb).permute(0, 2, 1)
        return xy

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted = predicted.permute(0, 2, 1)
        target = target.permute(0, 2, 1)
        emb = self.decoder(predicted)
        return self.embedding.loss(emb, target)


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
        num_inference_timesteps: int = 50,
        num_train_timesteps: int = 1000,
    ):
        super().__init__()

        self.cameras = cameras
        self.num_frames = num_frames
        self.num_encode_frames = num_encode_frames
        self.cam_shape = cam_shape
        self.feat_shape = (cam_shape[0] // 16, cam_shape[1] // 16)
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps

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
        self.xy_embedding = XYMLPEncoder(dim=dim, max_dist=128, pretrained=True)

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

        self.noise_scheduler = EulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps
        )
        self.noise_scheduler.set_timesteps(num_train_timesteps)
        self.eval_noise_scheduler = EulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps
        )
        self.eval_noise_scheduler.set_timesteps(self.num_inference_timesteps)

        self.batch_transform = NormalizeCarPosition(start_frame=0)

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        return [
            {
                "name": "encoders",
                "params": list(self.encoders.parameters()),
                "lr": lr / 10,
            },
            {
                "name": "static_features",
                "params": list(self.static_features_encoder.parameters()),
                "lr": lr / 10,
            },
            {
                "name": "denoiser",
                "params": list(self.denoiser.parameters()),
                "lr": lr,
            },
            {
                "name": "xy_embedding",
                "params": list(self.xy_embedding.parameters()),
                "lr": lr / 10,
            },
        ]

    def should_log(self, global_step: int, BS: int) -> Tuple[bool, bool]:
        log_text_interval = 1000 // BS
        # log_text_interval = 1
        # It's important to scale the less frequent interval off the more
        # frequent one to avoid divisor issues.
        log_img_interval = log_text_interval * 10
        log_img = (global_step % log_img_interval) == 0
        log_text = (global_step % log_text_interval) == 0

        return log_img, log_text

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
                unmasked, cam_feats = torch.utils.checkpoint.checkpoint(
                    encoder,
                    feats.flatten(0, 1),
                    mask,
                    use_reentrant=False,
                )
                assert cam_feats.requires_grad, f"missing grad for cam {cam}"

            if writer is not None and log_img:
                writer.add_image(
                    f"{cam}/color",
                    normalize_img(feats[0, 0]),
                    global_step=global_step,
                )
                writer.add_image(
                    f"{cam}/pca",
                    render_pca(unmasked[0].permute(1, 2, 0)),
                    global_step=global_step,
                )

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

        # calculate velocity between first two frames to allow model to understand current speed
        # TODO: convert this to a categorical embedding
        velocity = positions[:, 1] - positions[:, 0]
        assert positions.size(-1) == 2
        velocity = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)

        static_features = self.static_features_encoder(velocity).unsqueeze(1)

        lengths = mask.sum(dim=-1)
        min_len = lengths.amin()
        assert min_len > 0, "got example with zero sequence length"

        # truncate to shortest sequence
        # pos_len = lengths.amin()
        # if pos_len % align != 0:
        #    pos_len -= pos_len % align
        # assert pos_len >= 8
        # positions = positions[:, :pos_len]
        # mask = mask[:, :pos_len]

        # we need to be aligned to size 8
        # pad length
        align = 8
        if positions.size(1) % align != 0:
            pad = align - positions.size(1) % align
            mask = F.pad(mask, (0, pad), value=True)
            positions = F.pad(positions, (0, 0, 0, pad), value=0)
        pos_len = positions.size(1)

        assert positions.size(1) % align == 0
        assert mask.size(1) % align == 0
        assert positions.size(1) == mask.size(1)

        num_elements = mask.float().sum()

        if writer and log_text:
            writer.add_scalar(
                "paths/pos_len",
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

        noise = torch.randn(traj_embed.shape, device=traj_embed.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (BS,),
            device=traj_embed.device,
            dtype=torch.int64,
        )
        traj_embed_noise = self.noise_scheduler.add_noise(traj_embed, noise, timesteps)

        with autocast():
            # add static feature info to all condition keys to avoid noise
            input_tokens = input_tokens + static_features

            pred_noise = self.denoiser(traj_embed_noise, mask, input_tokens)

        noise_loss = F.mse_loss(pred_noise, noise, reduction="none")
        noise_loss = noise_loss[mask]
        losses["diffusion"] = noise_loss.mean()

        losses["ae/with_noise"] = (
            self.xy_embedding.loss(traj_embed_noise, positions)[mask].mean() * 0.1
        )
        losses["ae/ae"] = self.xy_embedding.loss(traj_embed, positions)[mask].mean()

        losses_backward(losses)

        if writer and log_img:
            # calculate cross_attn_weights
            with torch.no_grad():
                fig = plt.figure()

                # generate prediction
                self.train()

                pred_len = mask[0].sum()

                pred_traj = torch.randn_like(noise[:1])
                self.eval_noise_scheduler.set_timesteps(self.num_inference_timesteps)
                for timestep in self.eval_noise_scheduler.timesteps:
                    with autocast():
                        pred_traj = self.eval_noise_scheduler.scale_model_input(
                            pred_traj, timestep
                        )
                        noise = self.denoiser(pred_traj, mask[:1], input_tokens[:1])
                    pred_traj = self.eval_noise_scheduler.step(
                        noise,
                        timestep,
                        pred_traj,
                        generator=torch.Generator(device=device).manual_seed(0),
                    ).prev_sample

                pred_positions = self.xy_embedding.decode(pred_traj)[0, :pred_len].cpu()
                plt.plot(pred_positions[..., 0], pred_positions[..., 1], label="pred")

                noise_positions = self.xy_embedding.decode(traj_embed_noise[:1])[
                    0,
                    :pred_len,
                ].cpu()
                plt.plot(
                    noise_positions[..., 0], noise_positions[..., 1], label="with_noise"
                )

                pos_positions = self.xy_embedding.decode(traj_embed[:1])[
                    0, :pred_len
                ].cpu()
                plt.plot(pos_positions[..., 0], noise_positions[..., 1], label="ae")

                target = positions[0, :pred_len].detach().cpu()
                plt.plot(target[..., 0], target[..., 1], label="target")

                writer.add_scalar(
                    "paths/pred_mae",
                    F.l1_loss(pred_positions, target).item(),
                    global_step=global_step,
                )
                writer.add_scalar(
                    "paths/pred_len",
                    pred_len,
                    global_step=global_step,
                )

                self.eval()

            fig.legend()
            plt.gca().set_aspect("equal")
            writer.add_figure(
                "paths/target",
                fig,
                global_step=global_step,
            )

        return losses
