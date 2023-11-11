from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt

import torch
import torchmetrics
from torch import nn
from torchworld.transforms.img import render_color

from torchdrive.data import Batch
from torchdrive.models.path import PathOneShotTransformer, XYEncoder
from torchdrive.tasks.bev import BEVTask, Context


class PathTask(BEVTask):
    def __init__(
        self,
        bev_shape: Tuple[int, int],
        bev_dim: int,
        dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 6 * 2,
        num_ar_iters: int = 1,
        max_dist: float = 128,
        num_buckets: int = 100,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.num_ar_iters = num_ar_iters

        self.xy_encoder = XYEncoder(num_buckets=num_buckets, max_dist=max_dist)

        self.transformer: nn.Module = PathOneShotTransformer(
            bev_shape=bev_shape,
            bev_dim=bev_dim,
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            compile_fn=compile_fn,
            max_seq_len=max_seq_len,
            pos_dim=num_buckets * 2,
        )

        self.ae_mae = torchmetrics.MeanAbsoluteError()
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

        # used for direction bucket
        final_pos = positions[torch.arange(BS), :, (lengths - 1)]

        # downsample to 1/6 the frame rate (i.e. 12fps to 2fpss)
        downsample = 6
        positions = positions[..., ctx.start_frame :: downsample]
        mask = mask[..., ctx.start_frame :: downsample]
        lengths = mask.sum(dim=1)

        pos_len = positions.size(-1)
        # if we need to be aligned to size 8
        # pos_len = pos_len - (pos_len % 8) + 1
        pos_len = min(pos_len, self.max_seq_len + 1)
        positions = positions[..., :pos_len]
        mask = mask[..., :pos_len]
        lengths = lengths.clamp(max=pos_len)
        num_elements = mask.float().sum()

        assert mask.int().sum() == lengths.sum(), (mask, lengths)

        assert pos_len > 1, "pos length too short"

        if ctx.log_text:
            ctx.add_scalar("paths/seq_len", pos_len)
            ctx.add_scalar("paths/num_elements", num_elements)

        posmax = positions.abs().amax()
        assert posmax < 1000, positions

        for i, length in enumerate(lengths):
            if length == 0:
                continue
            pos = positions[i, :, :length]
            if torch.logical_and(pos[0] == 0, pos[1] == 0).any():
                breakpoint()

        prev = positions[..., :-1]
        target = positions[..., 1:]
        target_mask = mask[..., 1:]

        # print(prev)
        # print(target)

        # print(target, num_elements)
        # print(target_mask)
        all_predicted = []
        losses = {}
        i = 0

        inp_one_hot = self.xy_encoder.encode_one_hot(target)  # TODO
        final_pos_one_hot = self.xy_encoder.encode_one_hot(
            final_pos.unsqueeze(2)
        ).squeeze(2)
        predicted_one_hot, ae_prev = self.transformer(
            bev, inp_one_hot, final_pos_one_hot
        )

        per_token_loss = self.xy_encoder.loss(predicted_one_hot, target)
        per_token_loss = per_token_loss[target_mask]
        loss = per_token_loss.sum() / (per_token_loss.numel() + 1)

        # normalize by number of elements in sequence
        losses[f"position/{i}"] = loss

        predicted = self.xy_encoder.decode(predicted_one_hot)
        all_predicted.append(predicted)

        # ensure encoder and decoder are in sync
        # losses[f"ae/{i}"] = F.l1_loss(ae_prev, prev)
        # self.ae_mae.update(ae_prev, prev)

        l2_diff = torch.linalg.vector_norm(predicted - target, dim=1)[target_mask]
        self.position_mae.update(l2_diff, torch.zeros_like(l2_diff))

        if ctx.log_text:
            ctx.add_scalar("ae/mae", self.ae_mae.compute())
            ctx.add_scalar("position/mae", self.position_mae.compute())
            self.ae_mae.reset()
            self.position_mae.reset()

        if ctx.log_img:
            ctx.add_text("debug/target", str(target[0]))
            ctx.add_text("debug/prev", str(prev[0]))
            ctx.add_text("debug/mask", str(mask[0]))
            ctx.add_text("debug/length", str(lengths[0]))
            ctx.add_text("debug/predicted_one_hot", str(predicted_one_hot[0]))

            ctx.add_image("paths/one_hot", render_color(predicted_one_hot[0]))

            with torch.no_grad():
                fig = plt.figure()
                # pyre-fixme[9]: int
                length: int = min(lengths[0].item(), self.max_seq_len + 1)
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
                autoregressive = PathOneShotTransformer.infer(
                    self.xy_encoder,
                    self.transformer,
                    bev[:1],
                    positions[:1, ..., :1],
                    final_pos[:1],
                    n=length - 1,
                )
                assert autoregressive.shape == (1, 2, length), (
                    autoregressive.shape,
                    length,
                )
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

        losses = {k: v * 1000 for k, v in losses.items()}
        return losses

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        return [
            {"name": "default", "params": self.parameters(), "lr": lr},
        ]
