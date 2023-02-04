from typing import Dict, Tuple

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from torchdrive.amp import autocast
from torchdrive.data import Batch
from torchdrive.models.path import PathTransformer
from torchdrive.tasks.bev import BEVTask, Context


class PathTask(BEVTask):
    def __init__(
        self,
        bev_shape: Tuple[int, int],
        bev_dim: int,
        dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        self.transformer = PathTransformer(
            bev_shape=bev_shape,
            bev_dim=bev_dim,
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def forward(
        self, ctx: Context, batch: Batch, bev: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        BS = len(batch.distances)
        device = bev.device

        start_T = batch.cam_T[:, ctx.start_frame]
        cam_T = start_T.unsqueeze(1).pinverse().matmul(batch.long_cam_T)

        zero_coord = torch.zeros(1, 4, device=device, dtype=torch.float)
        zero_coord[:, -1] = 1

        positions = torch.matmul(cam_T, zero_coord.T)[..., :3, 0].permute(0, 2, 1)
        # downsample to 1/3 the frame rate
        positions = positions[..., ::3]

        # TODO: try adding noise to positions to help recover
        prev = positions[..., :-1]
        target = positions[..., 1:]

        with autocast():
            predicted = self.transformer(bev, prev).float()

        if ctx.log_text:
            ctx.add_scalar("paths/seq_len", positions.size(-1))

        if ctx.log_img:
            fig = plt.figure()
            plt.plot(*target[0, 0:2].detach().cpu(), label="target")
            plt.plot(*predicted[0, 0:2].detach().cpu(), label="predicted")
            fig.legend()
            plt.gca().set_aspect("equal")
            ctx.add_figure("paths", fig)
        position_loss = (
            F.mse_loss(predicted, target, reduction="none").mean(dim=(1, 2)) * 5
        )
        return {"position": position_loss}
