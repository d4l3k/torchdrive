from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from torchdrive.data import Batch
from torchdrive.models.path import PathTransformer, rel_dists
from torchdrive.tasks.bev import BEVTask, Context


class PathTask(BEVTask):
    def __init__(
        self,
        bev_shape: Tuple[int, int],
        bev_dim: int,
        dim: int = 768,
        num_heads: int = 16,
        num_layers: int = 6,
        max_seq_len: int = 6 * 2,
        num_ar_iters: int = 6,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.num_ar_iters = num_ar_iters

        self.transformer: nn.Module = PathTransformer(
            bev_shape=bev_shape,
            bev_dim=bev_dim,
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            compile_fn=compile_fn,
            max_seq_len=max_seq_len * 2,
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

        zero_coord = torch.zeros(1, 4, device=device, dtype=torch.float)
        zero_coord[:, -1] = 1

        positions = torch.matmul(car_to_world, zero_coord.T)[..., :3, 0].permute(
            0, 2, 1
        )

        # used for direction bucket
        final_pos = positions[torch.arange(BS), :, lengths - 1]

        # downsample to 1/6 the frame rate (i.e. 12fps to 2fpss)
        downsample = 6
        positions = positions[..., ctx.start_frame :: downsample]
        mask = mask[..., ::downsample]
        lengths = (lengths - ctx.start_frame) // downsample

        pos_len = positions.size(-1)
        # if we need to be aligned to size 8
        # pos_len = pos_len - (pos_len % 8) + 1
        pos_len = min(pos_len, self.max_seq_len + 1)
        positions = positions[..., :pos_len]
        mask = mask[..., 0:pos_len]
        num_elements = mask.float().sum()

        assert pos_len > 1, "pos length too short"

        if ctx.log_text:
            ctx.add_scalar("paths/seq_len", pos_len)
            ctx.add_scalar("paths/num_elements", num_elements)

        posmax = positions.abs().amax()
        assert posmax < 1000

        # target = positions[..., 1:]
        prev = positions[..., :-1]
        target = positions[..., 1:]

        all_predicted = []
        losses = {}

        for i in range(self.num_ar_iters):
            predicted, ae_prev = self.transformer(bev, prev, final_pos)
            all_predicted.append(predicted)

            # ensure encoder and decoder are in sync
            losses[f"ae/{i}"] = F.l1_loss(ae_prev, prev)
            self.ae_mae.update(ae_prev, prev)

            predicted_mask = mask[..., 1:]

            l2_diff = torch.linalg.vector_norm(predicted - target, dim=1)[
                predicted_mask
            ]
            self.position_mae.update(l2_diff, torch.zeros_like(l2_diff))

            per_token_loss = F.l1_loss(predicted, target, reduction="none")
            per_token_loss *= predicted_mask.unsqueeze(1).expand(-1, 3, -1).float()

            # normalize by number of elements in sequence
            losses[f"position/{i}"] = (
                per_token_loss.sum(dim=(1, 2)) * 5 / (num_elements + 1)
            )

            pred_dists = rel_dists(predicted)
            target_dists = rel_dists(target)
            rel_dist_loss = F.l1_loss(
                pred_dists,
                target_dists,
                reduction="none",
            )
            rel_dist_loss *= predicted_mask.float()
            losses[f"rel_dists/{i}"] = rel_dist_loss.sum(dim=1) / (num_elements + 1)

            # keep first value the same and shift predicted over by 1
            prev = torch.cat((prev[..., :1], predicted[..., :-1]), dim=-1)

        if ctx.log_text:
            ctx.add_scalar("ae/mae", self.ae_mae.compute())
            ctx.add_scalar("position/mae", self.position_mae.compute())

        if ctx.log_img:
            with torch.no_grad():
                fig = plt.figure()
                length = min(lengths[0], self.max_seq_len + 1)
                plt.plot(*target[0, 0:2, :length].detach().cpu(), label="target")
                plt.plot(*target[0, 0:2, 0].detach().cpu(), "go", label="origin")
                plt.plot(*final_pos[0, 0:2].detach().cpu(), "ro", label="final")

                for i, predicted in enumerate(all_predicted):
                    if i % max(1, self.num_ar_iters // 4) != 0:
                        continue
                    plt.plot(
                        *predicted[0, 0:2, :length].detach().cpu(),
                        label=f"predicted {i}",
                    )

                fig.legend()
                plt.gca().set_aspect("equal")
                ctx.add_figure("paths/predicted", fig)

                fig = plt.figure()
                # autoregressive
                self.eval()
                autoregressive = PathTransformer.infer(
                    self.transformer,
                    bev[:1],
                    positions[:1, ..., :1],
                    final_pos[:1],
                    n=length - 1,
                )
                assert autoregressive.shape == (1, 3, length), (
                    autoregressive.shape,
                    length,
                )
                plt.plot(*target[0, 0:2, :length].detach().cpu(), label="target")
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

        losses = {k: v * 10 * 5 for k, v in losses.items()}
        return losses

    def param_opts(self, lr: float) -> List[Dict[str, object]]:
        return [
            {"name": "default", "params": self.parameters(), "lr": lr},
        ]
