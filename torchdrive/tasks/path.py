from contextlib import nullcontext
from typing import Callable, Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt

import torch
import torchmetrics
from torch import nn
from torchworld.models.transformer import collect_cross_attention_weights
from torchworld.transforms.img import render_color

from torchdrive.data import Batch
from torchdrive.models.path import (
    PathAutoRegressiveTransformer,
    PathOneShotTransformer,
    XYEncoder,
)
from torchdrive.tasks.bev import BEVTask, Context


def unflatten_strided(x: torch.Tensor, stride: int) -> torch.Tensor:
    """
    Returns the tensor with the last channel unflattened into `stride` number of
    channels with each channel being a strided view of i::stride.

    Arguments
    ---------
    x: [..., stride*n]
    Returns:
    [..., stride, n]
    """
    dim = x.size(-1)
    n = dim // stride
    assert (
        n * stride == dim
    ), f"last channel {dim} must be a multiple of stride {stride}"

    out = []
    for i in range(stride):
        out.append(x[..., i::stride])
    return torch.stack(out, dim=-2)


class PathTask(BEVTask):
    def __init__(
        self,
        bev_shape: Tuple[int, int],
        bev_dim: int,
        dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 6 * 2,
        downsample: int = 6,
        num_ar_iters: int = 1,
        max_dist: float = 128,
        num_buckets: int = 512,
        one_shot: bool = False,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.num_ar_iters = num_ar_iters
        self.downsample = downsample
        self.num_buckets = num_buckets
        self.bev_shape = bev_shape

        self.xy_encoder = XYEncoder(num_buckets=num_buckets, max_dist=max_dist)

        if one_shot:
            self.transformer: nn.Module = PathOneShotTransformer(
                bev_shape=bev_shape,
                bev_dim=bev_dim,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                compile_fn=compile_fn,
                max_seq_len=max_seq_len,
                pos_dim=num_buckets * 2,
                static_features=2 * 3 + 1,  # start, final, velocity, time
            )
        else:
            self.transformer: nn.Module = PathAutoRegressiveTransformer(
                bev_shape=bev_shape,
                bev_dim=bev_dim,
                dim=dim,
                num_heads=num_heads,
                num_layers=num_layers,
                compile_fn=compile_fn,
                max_seq_len=max_seq_len,
                pos_dim=num_buckets * 2,
                static_features=2 * 3 + 1,  # start, final, velocity, time
            )

        self.ae_mae = torchmetrics.MeanAbsoluteError()
        self.perplexity = torchmetrics.Perplexity()
        self.position_mae = torchmetrics.MeanAbsoluteError()

    def forward(
        self,
        ctx: Context,
        batch: Batch,
        grids: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        bev = grids[-1].float()

        BS = len(batch.distances)
        device = bev.device

        world_to_car, mask, lengths = batch.long_cam_T
        car_to_world = torch.zeros_like(world_to_car)
        car_to_world[mask] = world_to_car[mask].inverse()

        assert mask.int().sum() == lengths.sum(), (mask, lengths)

        zero_coord = torch.zeros(1, 4, device=device, dtype=torch.float)
        zero_coord[:, -1] = 1

        positions = torch.matmul(car_to_world, zero_coord.T).squeeze(-1)
        positions /= positions[..., -1:] + 1e-8  # perspective warp
        positions = positions[..., :2].permute(0, 2, 1)

        # downsample to 1/6 the frame rate (i.e. 12fps to 2fpss)
        downsample = self.downsample

        # used for direction bucket
        final_pos = positions[torch.arange(BS), :, (lengths - 1)]
        # expand to one per downsample path
        final_pos = final_pos.unsqueeze(1).expand(-1, downsample, -1).flatten(0, 1)

        assert batch.frame_time.size(1) >= downsample
        start_time = batch.frame_time[:, :downsample].flatten(0, 1)

        # crop positions and mask to be multiple of downsample so we can run
        # multiple downsampled paths in parallel
        pos_len = positions.size(-1)
        pos_len = min(pos_len // downsample, self.max_seq_len + 1)

        positions = positions[..., : pos_len * downsample]
        mask = mask[..., : pos_len * downsample]

        positions = (
            unflatten_strided(positions, downsample).permute(0, 2, 1, 3).flatten(0, 1)
        )
        mask = unflatten_strided(mask, downsample).flatten(0, 1)
        assert positions.shape, (BS * downsample, 2, pos_len)
        assert mask.shape, (BS * downsample, pos_len)

        bev = bev.unsqueeze(1).expand(-1, downsample, -1, -1, -1).flatten(0, 1)

        start_pos = positions[:, :, 0]
        velocity = positions[:, :, 1] - positions[:, :, 0]

        static_features = torch.cat(
            (start_pos, final_pos, velocity, start_time.unsqueeze(1)), dim=1
        )

        lengths = mask.sum(dim=-1)

        # if we need to be aligned to size 8
        # pos_len = pos_len - (pos_len % 8) + 1
        num_elements = mask.float().sum()

        assert mask.int().sum() == lengths.sum(), (mask, lengths)

        assert pos_len > 1, "pos length too short"

        if ctx.log_text:
            ctx.add_scalar("paths/seq_len", pos_len)
            ctx.add_scalar("paths/num_elements", num_elements)

        posmax = positions.abs().amax()
        assert posmax < 1000, positions

        # for i, length in enumerate(lengths):
        #    if length == 0:
        #        continue
        #    pos = positions[i, :, :length]
        #    if torch.logical_and(pos[0] == 0, pos[1] == 0).any():
        #        breakpoint()

        prev = positions[..., :-1]
        target = positions[..., 1:]
        target_mask = mask[..., 1:]

        all_predicted = []
        losses = {}

        # loop multiple times to allow model to learn from it's mistakes
        for i in range(3):
            inp_one_hot = self.xy_encoder.encode_one_hot(prev)  # TODO

            if ctx.log_img:
                collect_ctx = collect_cross_attention_weights(self.transformer)
            else:
                collect_ctx = nullcontext()
            with collect_ctx as cross_attn_weights:
                predicted_one_hot, ae_prev = self.transformer(
                    bev, inp_one_hot, static_features
                )

            per_token_loss = self.xy_encoder.loss(predicted_one_hot, target)
            per_token_loss = per_token_loss[target_mask]
            loss = per_token_loss.sum() / (per_token_loss.numel() + 1)

            # normalize by number of elements in sequence
            losses[f"position/{i}"] = loss

            # calculate perplexity for x and y separately
            x_labels, y_labels = self.xy_encoder.encode_labels(target)
            x_pred, y_pred = self.xy_encoder.split_xy_one_hot(predicted_one_hot)
            self.perplexity.update(x_pred.permute(0, 2, 1), x_labels)
            self.perplexity.update(y_pred.permute(0, 2, 1), y_labels)

            predicted = self.xy_encoder.decode(predicted_one_hot)
            all_predicted.append(predicted)

            # ensure encoder and decoder are in sync
            # losses[f"ae/{i}"] = F.l1_loss(ae_prev, prev)
            # self.ae_mae.update(ae_prev, prev)

            l2_diff = torch.linalg.vector_norm(predicted - target, dim=1)[target_mask]
            self.position_mae.update(l2_diff, torch.zeros_like(l2_diff))

            # shift predicted value by 1 to become the new input
            # detach the gradient to ensure model doesn't learn degenerate
            # patterns
            prev = torch.cat((positions[..., :1], predicted[..., :-1].detach()), dim=-1)
            assert prev.shape == target.shape, (prev.shape, target.shape)

        if ctx.log_text:
            ctx.add_scalar("ae/mae", self.ae_mae.compute())
            self.ae_mae.reset()
            ctx.add_scalar("position/mae", self.position_mae.compute())
            self.position_mae.reset()
            ctx.add_scalar("position/perplexity", self.perplexity.compute())
            self.perplexity.reset()

        if ctx.log_img:
            ctx.add_text("debug/target", str(target[0]))
            ctx.add_text("debug/prev", str(prev[0]))
            ctx.add_text("debug/mask", str(mask[0]))
            ctx.add_text("debug/length", str(lengths[0]))
            ctx.add_text("debug/predicted_one_hot", str(predicted_one_hot[0]))

            ctx.add_image("paths/one_hot", render_color(predicted_one_hot[0]))

            # calculate cross_attn_weights
            weights = torch.cat(cross_attn_weights, dim=1)
            weights = weights.mean(dim=1)
            weights = weights.unflatten(-1, self.bev_shape)
            ctx.add_image("paths/cross_attn", render_color(weights[0]))

            with torch.no_grad():
                fig = plt.figure()
                # pyre-fixme[9]: int
                length: int = min(lengths[0].item(), self.max_seq_len)
                plt.plot(*target[0, 0:2, : length - 1].detach().cpu(), label="target")

                reencoded_target = self.xy_encoder.decode(
                    self.xy_encoder.encode_one_hot(target[0:, 0:2, : length - 1])
                )
                plt.plot(
                    *reencoded_target[0, 0:2, : length - 1].detach().cpu(),
                    label="reencoded_target",
                )

                plt.plot(*target[0, 0:2, 0].detach().cpu(), "go", label="origin")
                plt.plot(*final_pos[0, 0:2].detach().cpu(), "ro", label="final")

                for i, predicted in enumerate(all_predicted):
                    if i % max(1, self.num_ar_iters // 4) != 0:
                        continue
                    plt.plot(
                        *predicted[0, 0:2, : length - 1].detach().cpu(),
                        label=f"predicted {i}",
                    )

                fig.legend()
                plt.gca().set_aspect("equal")
                ctx.add_figure("paths/predicted", fig)

                fig = plt.figure()
                # autoregressive
                self.eval()
                autoregressive = self.transformer.infer(
                    self.xy_encoder,
                    self.transformer,
                    bev,
                    positions[..., :1],
                    static_features,
                    n=length - 1,
                )
                assert autoregressive.shape == (BS * downsample, 2, length), (
                    BS,
                    autoregressive.shape,
                    positions.shape,
                    length,
                )

                ag_target = positions[..., :length]
                ag_mask = mask[..., :length]
                l2_diff = torch.linalg.vector_norm(autoregressive - ag_target, dim=1)[
                    ag_mask
                ]
                ctx.add_scalar("position/mae_infer", l2_diff.mean())

                plt.plot(*target[0, 0:2, : length - 1].detach().cpu(), label="target")
                plt.plot(
                    *autoregressive[0, 0:2, :length].detach().cpu(),
                    label="autoregressive",
                )
                plt.plot(*target[0, 0:2, 0].detach().cpu(), "go", label="origin")
                plt.plot(*final_pos[0, 0:2].detach().cpu(), "ro", label="final")
                self.train()

                fig.legend()
                plt.gca().set_aspect("equal")
                ctx.add_figure("paths/autoregressive", fig)

        losses = {k: v * 100 for k, v in losses.items()}
        return losses

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        decoder_params = list(self.transformer.pos_decoder.parameters())

        other_params = _params_difference(self.parameters(), decoder_params)
        return [
            {"name": "default", "params": other_params, "lr": lr, "weight_decay": 1e-4},
            {
                "name": "decoder",
                "params": decoder_params,
                "lr": lr / 10,
                "weight_decay": 1e-4,
            },
        ]


def _params_difference(
    params: Iterator[nn.Parameter], to_remove: List[nn.Parameter]
) -> List[nn.Parameter]:
    out = []
    to_remove_set = set(to_remove)
    for p in params:
        if p in to_remove_set:
            continue
        out.append(p)
    return out
