from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from torchdrive.data import Batch
from torchdrive.models.path import PathTransformer
from torchdrive.tasks.bev import BEVTask, Context


class PathTask(BEVTask):
    def __init__(
        self,
        bev_shape: Tuple[int, int],
        bev_dim: int,
        dim: int = 768,
        num_heads: int = 16,
        num_layers: int = 12,
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
        )

        self.ae_mae = torchmetrics.MeanAbsoluteError()

    def forward(
        self, ctx: Context, batch: Batch, bev: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
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
        lengths //= downsample

        pos_len = positions.size(-1)
        # if we need to be aligned to size 8
        # pos_len = pos_len - (pos_len % 8) + 1
        pos_len = min(pos_len, self.max_seq_len + 1)
        positions = positions[..., :pos_len]
        mask = mask[..., 1:pos_len].float()
        num_elements = mask.sum()

        assert pos_len > 1, "pos length too short"

        if ctx.log_text:
            ctx.add_scalar("paths/seq_len", pos_len)
            ctx.add_scalar("paths/num_elements", num_elements)

        posmax = positions.abs().amax()
        assert posmax < 1000

        target = positions[..., 1:]
        prev = positions[..., :-1]

        all_predicted = []
        losses = {}

        for i in range(self.num_ar_iters):
            predicted, ae_prev = self.transformer(bev, prev, final_pos)

            # ensure encoder and decoder are in sync
            losses[f"ae/{i}"] = F.huber_loss(ae_prev, prev)
            self.ae_mae.update(ae_prev, prev)

            all_predicted.append(predicted)

            per_token_loss = F.huber_loss(predicted, target, reduction="none", delta=20.0)
            per_token_loss *= mask.unsqueeze(1).expand(-1, 3, -1)

            # normalize by number of elements in sequence
            losses[f"position/{i}"] = (
                per_token_loss.sum(dim=(1, 2)) * 5 / (num_elements + 1)
            )

            # keep first values the same and shift predicted over by 1
            prev = torch.cat((prev[..., :1], predicted[..., :-1]), dim=-1)

        if ctx.log_text:
            ctx.add_scalar("ae/mae", self.ae_mae.compute())

        if ctx.log_img:
            with torch.no_grad():
                fig = plt.figure()
                length = lengths[0] - 1
                plt.plot(*target[0, 0:2, :length].detach().cpu(), label="target")

                for i, predicted in enumerate(all_predicted):
                    if i % max(1, self.num_ar_iters // 4) != 0:
                        continue
                    plt.plot(
                        *predicted[0, 0:2, :length].detach().cpu(),
                        label=f"predicted {i}",
                    )

                # autoregressive
                self.eval()
                autoregressive = PathTransformer.infer(
                    self.transformer,
                    bev[:1],
                    positions[:1, ..., :2],
                    final_pos[:1],
                    n=length - 2,
                )
                plt.plot(*autoregressive[0, 0:2].detach().cpu(), label="autoregressive")
                self.train()

                fig.legend()
                plt.gca().set_aspect("equal")
                ctx.add_figure("paths", fig)

        return losses
