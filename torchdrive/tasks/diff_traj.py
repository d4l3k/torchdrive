from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from diffusers import DDPMScheduler
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.tensorboard import SummaryWriter

from torchdrive.amp import autocast
from torchdrive.autograd import autograd_context, register_log_grad_norm
from torchdrive.data import Batch
from torchdrive.losses import losses_backward
from torchdrive.tasks.van import Van
from torchtune.modules import RotaryPositionalEmbeddings
from torchworld.models.vit import MaskViT
from torchworld.transforms.img import normalize_img, normalize_mask, render_color
from torchworld.transforms.mask import random_block_mask, true_mask
from torchworld.transforms.pca import structured_pca


class Decoder(nn.Module):
    """Transformer decoder model for 1d sequences"""

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
        self.cam_shape = cam_shape
        self.dim = dim
        self.num_heads = num_heads
        self.positional_embedding = RotaryPositionalEmbeddings(
            dim // num_heads, max_seq_len=max_seq_len
        )

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
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
        x = self.positional_embedding(
            x.unflatten(-1, (self.num_heads, self.dim // self.num_heads))
        ).flatten(-1)

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

    def forward(self, pos: torch.Tensor):
        """
        Args:
            pos: the list of positions (...)

        Returns:
            the embedding of the position (..., dim)
        """

        dx = (self.shape - 1) // 2
        x = (pos * dx / self.scale + dx).long()

        x = x.clamp(min=0, max=self.shape - 1)

        return self.embedding[x]

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Convert the embedding back to the position using a cosine similarity distance function.

        Args:
            input: input embedding to decode (bs, seq_len, dim)

        Returns:
            the position (bs, seq_len)
        """

        # (bs, seq_len, dim) @ (x, dim) -> (bs, seq_len, x)
        similarity = torch.einsum("bsd,xd->bsx", input, self.embedding)

        x = similarity.argmax(dim=-1)

        dx = (self.shape - 1) // 2
        x = (x.float() - dx) * self.scale / dx
        return x


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
        #
        self.xy_embedding = XYEmbedding(
            shape=(128, 128),
            scale=100,  # 100 meters
            dim=dim,
        )

        self.decoder = Decoder(
            max_seq_len=20,
            num_layers=num_layers,
            num_heads=num_heads,
            dim=dim,
            mlp_dim=dim_feedforward,
            attention_dropout=dropout,
        )

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        return [
            {
                "name": "encoders",
                "params": list(self.encoders.parameters()),
                "lr": lr / len(self.encoders),
            },
            {
                "name": "decoder",
                "params": list(self.backbone.parameters()),
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
        self.ema_encoders.update_parameters(self.encoders)

        BS = len(batch.distances)
        log_img, log_text = self.should_log(global_step, BS)

        # for size, device only
        empty_mask = torch.empty(
            self.feat_shape, device=self.encode_time_embedding.device
        )

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
                    f"mask/{cam}/count",
                    mask.long().sum(),
                    global_step=global_step,
                )
            if writer is not None and log_img:
                writer.add_image(
                    f"mask/{cam}/mask",
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
            cam_feats += (
                self.encode_time_embedding
            )  # broadcast across batch and seq len

            # flatten time
            # (bs, num_encode_frames, seq_len, hidden_dim) -> (bs, num_encode_frames * seq_len, hidden_dim)
            cam_feats = cam_feats.flatten(1, 2)

            all_feats.append(cam_feats)

        input_tokens = torch.cat(all_feats, dim=1)

        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        noise = torch.randn(sample_image.shape)
        timesteps = torch.LongTensor([50])
        noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

        return losses
