from typing import List, Optional, Dict, Tuple
from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel

from torchdrive.amp import autocast
from torchdrive.autograd import (
    autograd_context,
    register_log_grad_norm,
)
from torchdrive.data import Batch
from torchdrive.losses import losses_backward
from torchdrive.tasks.van import Van
from torchworld.transforms.img import normalize_img, normalize_mask, render_color
from torchworld.transforms.mask import random_block_mask, true_mask
from torchworld.transforms.pca import structured_pca

class MaskViT(nn.Module):
    def __init__(
        self,
        attention_dropout: float,
        cam_shape: Tuple[int, int],
        dim: int,
    ) -> None:
        super().__init__()

        self.cam_shape = cam_shape
        self.encoder = vit_b_16(
            weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1,
            progress=True,
            attention_dropout=attention_dropout,
        )
        self.encoder.encoder.pos_embedding = nn.Parameter(
            torch.empty(
                1,
                self.encoder.hidden_dim,
                cam_shape[0] // self.encoder.patch_size,
                cam_shape[1] // self.encoder.patch_size,
            ).normal_(std=0.02)
        )
        self.project = nn.Linear(self.encoder.hidden_dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Reshape and permute the input tensor
        n, c, h, w = x.shape
        p = self.encoder.patch_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.encoder.conv_proj(x)

        x = x + self.encoder.encoder.pos_embedding

        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, mask.sum())
        x = x[:, :, mask]

        # (n, hidden_dim, mask_elems) -> (n, mask_elems, hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        x = self.encoder.encoder.ln(self.encoder.encoder.layers(self.encoder.encoder.dropout(x)))
        return self.project(x)


class Decoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        cam_shape: int,
        num_frames: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        attention_dropout: float,
    ):
        super().__init__()
        self.cam_shape = cam_shape
        self.queries = nn.Parameter(
            torch.empty(1, num_frames * cam_shape[0] * cam_shape[1], hidden_dim).normal_(std=0.02)
        )  # from BERT
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

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.queries.repeat(input.shape[0], 1, 1)
        for layer in self.layers:
            x = layer(x, input)

        return x


class ViTJEPA(nn.Module, Van):
    def __init__(
        self,
        cameras: List[str],
        num_frames: int,
        num_encode_frames: int,
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
        self.encoders = nn.ModuleDict({
            cam: MaskViT(
                cam_shape=cam_shape,
                dim=dim,
                attention_dropout=0.1,
            )
            for cam in cameras
        })
        self.ema_encoders = AveragedModel(
            self.encoders,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.998),
        )
        self.backbone = Decoder(
            cam_shape=(32, 32),
            hidden_dim=dim,
            mlp_dim=dim_feedforward,
            num_heads=num_heads,
            num_layers=num_layers,
            attention_dropout=dropout,
            num_frames=1,
        )
        self.decoders = nn.ModuleDict({
            cam: Decoder(
                num_frames=num_frames,
                cam_shape=self.feat_shape,
                hidden_dim=dim,
                mlp_dim=dim_feedforward,
                num_heads=num_heads,
                num_layers=12,  # num_layers,
                attention_dropout=dropout,
            )
            for cam in cameras
        })
        self.encode_time_embedding = nn.Parameter(
            torch.empty(1, num_encode_frames, 1, dim).normal_(std=0.02)
        )

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        return [
            {
                "name": "encoders",
                "params": list(self.encoders.parameters()),
                "lr": lr / len(self.encoders),
            },
            {
                "name": "decoders",
                "params": list(self.decoders.parameters()),
                "lr": lr / len(self.decoders),
            },
            {
                "name": "backbone",
                "params": list(self.backbone.parameters()),
                "lr": lr,
            },
            {
                "name": "time_embedding",
                "params": [self.encode_time_embedding],
                "lr": lr,
            },
            {
                "name": "ema",
                "params": list(self.ema_encoders.parameters()),
                "lr": 0.0,
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
            self.feat_shape,
            device=self.encode_time_embedding.device
        )

        all_feats = []

        for cam in self.cameras:
            feats = batch.color[cam][:, :self.num_encode_frames]
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
                    encoder, feats.flatten(0, 1), mask, use_reentrant=False,
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
            cam_feats += self.encode_time_embedding  # broadcast across batch and seq len

            # flatten time
            # (bs, num_encode_frames, seq_len, hidden_dim) -> (bs, num_encode_frames * seq_len, hidden_dim)
            cam_feats = cam_feats.flatten(1, 2)

            all_feats.append(cam_feats)

        input_tokens = torch.cat(all_feats, dim=1)

        # run backbone
        with autocast():
            input_tokens = torch.utils.checkpoint.checkpoint(
                self.backbone, input_tokens, use_reentrant=False,
            )

        # pause gradient on the input tokens so we can run backprop on each decoder camera separately
        with autograd_context(input_tokens) as input_tokens:

            losses = {}

            for cam in self.cameras:
                all_color = batch.color[cam]
                mask = true_mask(empty_mask)

                with torch.no_grad(), autocast():
                    target_feats = self.ema_encoders.module[cam](all_color.flatten(0, 1), mask).detach()
                    target_feats = target_feats.unflatten(0, all_color.shape[:2])
                    target_feats = target_feats.unflatten(2, self.feat_shape)
                    all_target_feats_raw = target_feats
                    # normalize on hidden dim
                    all_target_feats = F.layer_norm(
                        target_feats,
                        (target_feats.size(-1),),
                    )
                assert not all_target_feats.requires_grad

                with autocast():
                    pred_feats = self.decoders[cam](input_tokens)
                    all_pred_feats = pred_feats.unflatten(1, target_feats.shape[1:4])

                if writer is not None and log_text:
                    register_log_grad_norm(
                        t=pred_feats,
                        writer=writer,
                        key="gradnorm/cam-decoder",
                        tag=cam,
                        global_step=global_step,
                    )

                for frame in range(self.num_frames):
                    target_feats = all_target_feats[:, frame]
                    target_feats_raw = all_target_feats_raw[:, frame]
                    pred_feats = all_pred_feats[:, frame]
                    color = all_color[:, frame]

                    losses[f"cam_features/{cam}/{frame}"] = F.mse_loss(
                        pred_feats, target_feats
                    )

                    if writer is not None and log_text:
                        writer.add_scalars(
                            f"{cam}/{frame}/predicted-minmax",
                            {
                                "max": pred_feats.max(),
                                "min": pred_feats.min(),
                                "mean": pred_feats.mean(),
                            },
                            global_step=global_step,
                        )
                        writer.add_scalars(
                            f"{cam}/{frame}/target-minmax",
                            {
                                "max": target_feats.max(),
                                "min": target_feats.min(),
                                "mean": target_feats.mean(),
                            },
                            global_step=global_step,
                        )

                    if writer is not None and log_img:
                        writer.add_image(
                            f"{cam}/{frame}/target",
                            render_color(target_feats[0].sum(dim=-1)),
                            global_step=global_step,
                        )
                        writer.add_image(
                            f"{cam}/{frame}/target_raw",
                            render_color(target_feats_raw[0].sum(dim=-1)),
                            global_step=global_step,
                        )
                        writer.add_image(
                            f"{cam}/{frame}/predicted",
                            render_color(pred_feats[0].sum(dim=-1)),
                            global_step=global_step,
                        )
                        writer.add_image(
                            f"{cam}/{frame}/color",
                            normalize_img(color[0]),
                            global_step=global_step,
                        )
                        pca = structured_pca(
                            torch.cat((pred_feats[0], target_feats[0]), dim=1),
                            dim=3,
                        ).permute(2, 0, 1)
                        writer.add_image(
                            f"{cam}/{frame}/pca",
                            normalize_img(
                                pca
                            ),
                            global_step=global_step,
                        )

                # run camera specific backwards pass
                losses_backward(losses)

        return losses
