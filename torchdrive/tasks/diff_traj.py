import math
import os.path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchmetrics
from diffusers import EulerDiscreteScheduler
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_euler_angles
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
from torchdrive.models.transformer import transformer_init
from torchdrive.models.vista import VistaSampler
from torchdrive.tasks.context import Context
from torchdrive.tasks.van import Van
from torchdrive.transforms.batch import Compose, ImageTransform, NormalizeCarPosition
from torchtune.modules import RotaryPositionalEmbeddings
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from torchworld.models.vit import MaskViT
from torchworld.transforms.img import (
    normalize_img,
    normalize_mask,
    render_color,
    render_pca,
)
from torchworld.transforms.mask import random_block_mask, true_mask
from torchworld.transforms.transform3d import Transform3d


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
        transformer_init(self.layers)

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
        # x = x.unflatten(-1, (self.num_heads, self.dim // self.num_heads))
        # x = self.positional_embedding(x)
        # x = x.flatten(-2, -1)

        if False:
            attn_mask = square_mask(input_mask, num_heads=self.num_heads)
            # True values are ignored so need to flip the mask
            attn_mask = torch.logical_not(attn_mask)
        else:
            attn_mask = None

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


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """
    Args:
        pos_tensor: [bs, n, 2] -> [bs, n, dim], input range 0-1

    Converts a position into a sine encoded tensor.
    From: https://github.com/sshaoshuai/MTR/blob/master/mtr/models/motion_decoder/mtr_decoder.py#L134
    # Copyright (c) 2022 Shaoshuai Shi. All Rights Reserved.
    # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
    # ------------------------------------------------------------------------
    # DAB-DETR
    # Copyright (c) 2022 IDEA. All Rights Reserved.
    # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
    # ------------------------------------------------------------------------
    # Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
    # Copyright (c) 2021 Microsoft. All Rights Reserved.
    # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
    # ------------------------------------------------------------------------
    # Modified from DETR (https://github.com/facebookresearch/detr)
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
    # ------------------------------------------------------------------------
    """
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def nll_loss_gmm_direct(
    pred_trajs,
    gt_trajs,
    gt_valid_mask,
    pre_nearest_mode_idxs=None,
    timestamp_loss_weight=None,
    use_square_gmm=False,
    log_std_range=(-1.609, 5.0),
    rho_limit=0.5,
):
    """
    Gausian Mixture Model loss for trajectories.

    Adapted from https://github.com/sshaoshuai/MTR/blob/master/mtr/utils/loss_utils.py

    GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
    Written by Shaoshuai Shi

    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
        gt_trajs (batch_size, num_timestamps, 2):
        gt_valid_mask (batch_size, num_timestamps):
        timestamp_loss_weight (num_timestamps):
    """
    if use_square_gmm:
        assert pred_trajs.shape[-1] == 3
    else:
        assert pred_trajs.shape[-1] == 5

    batch_size = pred_trajs.size(0)

    if pre_nearest_mode_idxs is not None:
        nearest_mode_idxs = pre_nearest_mode_idxs
    else:
        distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1)
        distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)

        nearest_mode_idxs = distance.argmin(dim=-1)
    nearest_mode_bs_idxs = torch.arange(batch_size).type_as(
        nearest_mode_idxs
    )  # (batch_size, 2)

    nearest_trajs = pred_trajs[
        nearest_mode_bs_idxs, nearest_mode_idxs
    ]  # (batch_size, num_timestamps, 5)
    res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
    dx = res_trajs[:, :, 0]
    dy = res_trajs[:, :, 1]

    if use_square_gmm:
        log_std1 = log_std2 = torch.clip(
            nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1]
        )
        std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
        rho = torch.zeros_like(log_std1)
    else:
        log_std1 = torch.clip(
            nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1]
        )
        log_std2 = torch.clip(
            nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1]
        )
        std1 = torch.exp(log_std1)  # (0.2m to 150m)
        std2 = torch.exp(log_std2)  # (0.2m to 150m)
        rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

    gt_valid_mask = gt_valid_mask.type_as(pred_trajs)
    if timestamp_loss_weight is not None:
        gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

    # -log(a^-1 * e^b) = log(a) - b
    reg_gmm_log_coefficient = (
        log_std1 + log_std2 + 0.5 * torch.log(1 - rho**2)
    )  # (batch_size, num_timestamps)
    reg_gmm_exp = (0.5 * 1 / (1 - rho**2)) * (
        (dx**2) / (std1**2)
        + (dy**2) / (std2**2)
        - 2 * rho * dx * dy / (std1 * std2)
    )  # (batch_size, num_timestamps)

    reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

    return reg_loss, nearest_mode_idxs, nearest_trajs


class XYGMMEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        max_dist: float,
        dropout: float = 0.1,
        num_traj: int = 5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_dist = max_dist
        self.num_traj = num_traj
        # [x, y, log_std1, log_std2, rho]
        self.traj_size = 5

        self.encoder = MLP(dim + 2, dim, dim, num_layers=3, dropout=dropout)
        self.decoder = MLP(
            dim, dim, self.num_traj * self.traj_size, num_layers=3, dropout=dropout
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xy: the list of positions (..., 2)

        Returns:
            the embedding of the position (..., dim)
        """
        normalized_xy = (xy / (2 * self.max_dist)) + 0.5
        emb = gen_sineembed_for_position(xy, hidden_dim=self.dim)
        combined = torch.cat((emb, normalized_xy), dim=-1)
        return self.encoder(combined.permute(0, 2, 1)).permute(0, 2, 1)

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the position embedding (bs, n, dim)
        Returns:
            the position (bs, num_traj, n, 2)
        """
        input = input.float()
        # [bs, dim * num_traj, n]
        out = self.decoder(input.permute(0, 2, 1))
        # [bs, num_traj, dim, n]
        out = out.unflatten(1, (self.num_traj, self.traj_size))
        # [bs, num_traj, n, dim]
        out = out.permute(0, 1, 3, 2)
        return out

    def loss(
        self, emb: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            emb: the embedding of the position (bs, n, dim)
            target: the position (bs, n, 2)
            mask: the mask of the position (bs, n)

        Returns:
            the loss (bs, n)
            best trajectories: (bs, n, 2)
        """
        pred_traj = self.decode(emb)
        # l2 distance norm
        loss, nearest_mode, nearest_trajs = nll_loss_gmm_direct(pred_traj, target, mask)
        return loss, nearest_trajs[..., :2], pred_traj


class XYSineMLPEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        max_dist: float,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_dist = max_dist

        self.decoder = MLP(dim, dim, 2, num_layers=3, dropout=dropout)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xy: the list of positions (..., 2)

        Returns:
            the embedding of the position (..., dim)
        """
        xy = (xy / (2 * self.max_dist)) + 0.5
        return gen_sineembed_for_position(xy, hidden_dim=self.dim)

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the position embedding (bs, n, dim)
        Returns:
            the position (bs, n, 2)
        """
        output = self.decoder(input.permute(0, 2, 1)).permute(0, 2, 1)
        return (output.float().sigmoid() - 0.5) * (2 * self.max_dist)

    def loss(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos = self.decode(emb)
        # l2 distance norm
        # return torch.linalg.vector_norm(pos-target, dim=-1)
        return F.mse_loss(pos, target, reduction="none").mean(dim=-1)


class ConvNextPathPred(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        max_seq_len: int = 18,
        pool_size: int = 4,
        num_traj: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights

        self.dim = dim
        self.max_seq_len = max_seq_len
        # [x, y, log_std1, log_std2, rho]
        self.traj_size = 5
        self.num_traj = num_traj

        self.encoder = convnext_base(
            weights=ConvNeXt_Base_Weights.IMAGENET1K_V1,
        ).features
        enc_dim = 1024
        self.avgpool = nn.AdaptiveAvgPool2d(pool_size)

        self.static_features_encoder = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dim, dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(enc_dim * pool_size * pool_size + dim, max_seq_len * dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(max_seq_len * dim, max_seq_len * dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(
                max_seq_len * dim,
                max_seq_len * self.num_traj * self.traj_size + self.num_traj,
            ),
        )

    def forward(
        self,
        static_features: torch.Tensor,
        color: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # take first frame
        with autocast():
            x = color[:, 0]
            x = self.encoder(x)
            x = self.avgpool(x)

        x = x.flatten(1, 3).float()

        static_features_emb = self.static_features_encoder(static_features)

        combined = torch.cat((x, static_features_emb), dim=-1)

        embed = self.decoder(combined)

        traj_classes = embed[:, : self.num_traj]

        pred_traj = embed[:, self.num_traj :]
        pred_traj = pred_traj.unflatten(
            1, (self.num_traj, self.max_seq_len, self.traj_size)
        )

        length = min(self.max_seq_len, target.shape[1])

        pred_traj = pred_traj[:, :, :length]
        target = target[:, :length]
        mask = mask[:, :length]

        traj_loss, nearest_mode, nearest_trajs = nll_loss_gmm_direct(
            pred_traj, target, mask
        )

        class_loss = F.cross_entropy(traj_classes, nearest_mode, reduction="none")

        assert traj_loss.numel() > 1
        assert class_loss.numel() > 1

        losses = {
            "paths/best": traj_loss,
            "paths/class": class_loss,
        }

        return losses, nearest_trajs[..., :2], pred_traj


def random_traj(
    BS: int, seq_len: int, device: object, vel: torch.Tensor
) -> torch.Tensor:
    """Generates a random trajectory at the specified velocity."""

    # scale from 0.5 to 1.5
    speed = (torch.rand(BS, device=device) + 0.5) * vel

    angle = torch.rand(BS, device=device) * math.pi
    x = torch.sin(angle) * torch.arange(seq_len, device=device) / 2 * speed
    y = torch.cos(angle) * torch.arange(seq_len, device=device) / 2 * speed

    traj = torch.stack([x, y], dim=-1)
    return traj


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
        num_layers: int = 12,
        num_heads: int = 16,
        num_encode_frames: int = 1,
        num_frames: int = 1,
        num_inference_timesteps: int = 50,
        num_train_timesteps: int = 1000,
        max_seq_len: int = 256,
        test: bool = False,
    ):
        super().__init__()

        self.cameras = cameras
        self.num_frames = num_frames
        self.num_encode_frames = num_encode_frames
        self.cam_shape = cam_shape
        self.feat_shape = (cam_shape[0] // 16, cam_shape[1] // 16)
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.noise_scale = 25.0

        self.model = ConvNextPathPred()

        # dream parameters
        self.dream_steps = 1
        self.vista_fps = 10
        self.steps_per_second = 2
        if not test:
            vista_frames = (
                1 + self.vista_fps * self.dream_steps // self.steps_per_second
            )
            self.vista = VistaSampler(steps=25, num_frames=vista_frames)

        self.batch_transform = Compose(
            NormalizeCarPosition(start_frame=0),
            # ImageTransform(
            #    v2.RandomRotation(15, InterpolationMode.BILINEAR),
            #    v2.RandomErasing(),
            # ),
        )

        self.test_mae = torchmetrics.MeanAbsoluteError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_losses = defaultdict(lambda: torchmetrics.aggregation.MeanMetric())

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        return [
            {
                "name": "model",
                "params": list(self.model.parameters()),
                "lr": lr,
                "weight_decay": 1e-4,
            },
        ]

    def should_log(self, global_step: int, BS: int) -> Tuple[bool, bool]:
        log_text_interval = 10 // BS
        # log_text_interval = 1
        # It's important to scale the less frequent interval off the more
        # frequent one to avoid divisor issues.
        log_img_interval = log_text_interval * 10
        log_img = (global_step % log_img_interval) == 0
        log_text = (global_step % log_text_interval) == 0

        return log_img, log_text

    def prepare_inputs(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: a batch of data

        Returns:
            positions: (bs, num_encode_frames, 2)
            mask: (bs, num_encode_frames)
            velocity: (bs,)
        """
        world_to_car, mask, lengths = batch.long_cam_T
        positions = batch.positions()
        positions = positions[..., :2]

        # calculate velocity between first two frames to allow model to understand current speed
        # TODO: convert this to a categorical embedding

        # approximately 0.5 fps since video is 12hz
        positions = positions[:, ::6]
        mask = mask[:, ::6]

        # at 2 hz multiply by 2 to get true velocity
        velocity = positions[:, 1] - positions[:, 0]
        assert positions.size(-1) == 2
        velocity = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True) * 2

        return positions, mask, velocity

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
        ctx = Context(
            log_img=log_img,
            log_text=log_text,
            global_step=global_step,
            writer=writer,
            output=output,
            start_frame=0,
            weights=batch.weight,
            scaler=None,
        )

        for cam in self.cameras:
            feats = batch.color[cam][:, : self.num_encode_frames]
            if log_img:
                ctx.add_image(
                    f"{cam}/color",
                    normalize_img(feats[0, 0]),
                )

        positions, mask, velocity = self.prepare_inputs(batch)

        lengths = mask.sum(dim=-1)
        min_len = lengths.amin()
        assert min_len > 0, "got example with zero sequence length"

        # truncate to shortest sequence
        pos_len = lengths.amin()

        num_elements = mask.float().sum()

        if log_text:
            ctx.add_scalar(
                "paths/pos_len",
                pos_len,
            )
            ctx.add_scalar(
                "paths/num_elements",
                num_elements,
            )

        posmax = positions.abs().amax()
        assert posmax < 100000, positions

        cam = self.cameras[0]
        pred_losses, pred_traj, all_pred_traj = self.model(
            velocity, batch.color[cam], positions, mask
        )
        losses.update(pred_losses)

        pred_len = min(pred_traj.size(1), mask[0].sum().item())
        pred_traj_len = min(positions.size(1), pred_traj.size(1))

        rand_traj = random_traj(BS, pred_traj_len, device=device, vel=velocity)

        dreamed_imgs = []
        for i in range(BS):
            cond_img = batch.color[cam][i : i + 1, 0]
            cond_traj = rand_traj[i : i + 1]

            dreamed_img = self.vista.generate(cond_img, cond_traj)
            # add last img (frame 10 == 1s)
            dreamed_imgs.append(dreamed_img[-1])

        # [BS, 1, 3, H, W]
        dream_img = torch.stack(dreamed_imgs, dim=0).unsqueeze(1)

        if log_img:
            ctx.add_image(
                f"{cam}/dream",
                normalize_img(dream_img[0, 0]),
            )

        dream_target, dream_mask, dream_positions, dream_pred = compute_dream_pos(
            positions[:, :pred_traj_len],
            mask[:, :pred_traj_len],
            rand_traj[:, :pred_traj_len],
            step=self.dream_steps,
        )

        dream_losses, dream_traj, all_dream_traj = self.model(
            velocity, dream_img, dream_target, dream_mask
        )
        for k, v in dream_losses.items():
            losses[f"dream-{k}"] = v

        if writer and log_text:
            size = min(pred_traj.size(1), positions.size(1))

            ctx.add_scalar(
                "paths/pred_mae",
                F.l1_loss(pred_traj[:, :size], positions[:, :size], reduction="none")[
                    mask[:, :size]
                ]
                .mean()
                .cpu()
                .item(),
            )

        if self.training:
            ctx.backward(losses)

        if writer and log_img:
            # calculate cross_attn_weights
            with torch.no_grad():
                fig = plt.figure()

                target = positions[0, :pred_len].detach().cpu()
                plt.plot(target[..., 0], target[..., 1], label="target")

                for i in range(self.model.num_traj):
                    pred_positions = all_pred_traj[0, i, :pred_len].cpu()
                    plt.plot(
                        pred_positions[..., 0], pred_positions[..., 1], label=f"pred{i}"
                    )

                ctx.add_scalar(
                    "paths/pred_len",
                    pred_len,
                )

            fig.legend()
            plt.gca().set_aspect("equal")
            ctx.add_figure(
                "paths/target",
                fig,
            )

            with torch.no_grad():
                fig = plt.figure()

                pred_len = min(pred_len, dream_mask[0].sum().item())

                og_target = dream_positions[0, :pred_len].detach().cpu()
                plt.plot(og_target[..., 0], og_target[..., 1], label="positions")

                target = dream_target[0, :pred_len].detach().cpu()
                plt.plot(target[..., 0], target[..., 1], label="dream_target")

                pred_positions = dream_pred[0, :pred_len].cpu()
                plt.plot(
                    pred_positions[..., 0], pred_positions[..., 1], label="rand_traj"
                )

                pred_positions = dream_traj[0, :pred_len].cpu()
                plt.plot(pred_positions[..., 0], pred_positions[..., 1], label="pred")

            fig.legend()
            plt.gca().set_aspect("equal")

            ctx.add_figure(
                "dream-paths/target",
                fig,
            )

        return losses

    def test(
        self, batch: Batch, global_step: int, writer: Optional[SummaryWriter] = None
    ) -> Dict[str, torch.Tensor]:
        batch = self.batch_transform(batch)

        losses = {}

        BS = len(batch.distances)
        device = batch.device()

        log_img, log_text = self.should_log(global_step, BS)
        ctx = Context(
            log_img=log_img,
            log_text=log_text,
            global_step=global_step,
            writer=writer,
            output=None,
            start_frame=0,
            weights=batch.weight,
            scaler=None,
            name="test/",
        )

        positions, mask, velocity = self.prepare_inputs(batch)
        cam = self.cameras[0]

        pred_losses, pred_traj, all_pred_traj = self.model(
            velocity, batch.color[cam], positions, mask
        )
        for name, loss in pred_losses.items():
            metric = self.test_losses[name]
            metric.to(device)
            metric.update(loss.mean())

        size = min(pred_traj.size(1), positions.size(1))

        pred_traj = pred_traj[:, :size].flatten()
        positions = positions[:, :size].flatten()

        self.test_mae.update(pred_traj, positions)
        self.test_mse.update(pred_traj, positions)

        if log_text:
            for name, loss in self.test_losses.items():
                ctx.add_scalar(
                    f"loss/{name}",
                    loss.compute(),
                )

            ctx.add_scalar("mae", self.test_mae.compute())
            ctx.add_scalar("mse", self.test_mse.compute())


def compute_dream_pos(
    positions: torch.Tensor, mask: torch.Tensor, pred_traj: torch.Tensor, step: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a new ground truth trajectory for the dreamer to use as a loss.
    Outputted directory is centered at 0,0 and uses the new direction.

    Args:
        positions: (B, T, 2) the ground truth trajectory
        mask: (B, T) mask for the ground truth trajectory
        pred_traj: (B, T, 2) the trajectory the dreamer is following
        step: pred_traj[:, step] is the new root position
    Returns:
        dream_target: (B, T-step, 2) the new ground truth trajectory in step coordinate frame
        dream_mask: (B, T-step) the new mask for the ground truth trajectory
        dream_positions:  (B, T-step, 2) positions in step coordinate frame
        dream_pred: (B, T-step, 2)  pred_traj in step coordinate frame
    """
    direction = pred_traj[:, step] - pred_traj[:, step - 1]

    angle = torch.atan2(direction[:, 1], direction[:, 0])
    rot = torch.stack(
        [
            torch.stack(
                [
                    torch.cos(angle),
                    -torch.sin(angle),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    torch.sin(angle),
                    torch.cos(angle),
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )
    rot = rot.pinverse()

    # drop old points
    positions = positions[:, step:]
    mask = mask[:, step:]
    pred_traj = pred_traj[:, step:]

    # use linear interpolation between pred_traj and positions
    # factor = torch.arange(0, positions.size(1), device=positions.device) / (positions.size(1) - 1)
    # factor = factor.unsqueeze(0).unsqueeze(-1)
    # dream_pos = pred_traj * (1-factor) + positions * factor

    # use ema interpolation between pred_traj and positions
    factor = torch.full((positions.size(1),), 0.5, device=positions.device)
    factor[0] = 1.0
    factor = torch.cumprod(factor, dim=0)
    factor = factor.unsqueeze(0).unsqueeze(-1)

    dream_pos = pred_traj * factor + positions * (1 - factor)

    origin = dream_pos[:, 0:1]

    # center dream_pos according to direction
    dream_pos = dream_pos - origin
    pred_traj = pred_traj - origin
    positions = positions - origin

    # reorientate
    dream_pos = dream_pos.matmul(rot)
    pred_traj = pred_traj.matmul(rot)
    positions = positions.matmul(rot)

    return dream_pos, mask, positions, pred_traj
