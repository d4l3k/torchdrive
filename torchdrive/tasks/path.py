from typing import Dict, Tuple

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

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
        max_seq_len: int = 40,
    ) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len

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
        long_cam_T, mask, lengths = batch.long_cam_T
        cam_T = start_T.unsqueeze(1).pinverse().matmul(long_cam_T)

        zero_coord = torch.zeros(1, 4, device=device, dtype=torch.float)
        zero_coord[:, -1] = 1

        positions = torch.matmul(cam_T, zero_coord.T)[..., :3, 0].permute(0, 2, 1)

        # used for direction bucket
        final_pos = positions[torch.arange(BS), :, lengths - 1]

        # downsample to 1/3 the frame rate
        positions = positions[..., ::3]
        mask = mask[..., ::3]
        lengths //= 3

        pos_len = positions.size(-1)
        pos_len = pos_len - (pos_len % 8) + 1
        pos_len = min(pos_len, self.max_seq_len + 1)
        positions = positions[..., :pos_len]
        mask = mask[..., 1:pos_len]

        assert pos_len > 1, "pos length too short"

        # TODO: try adding noise to positions to help recover
        prev = positions[..., :-1]
        target = positions[..., 1:]

        predicted = self.transformer(bev, prev, final_pos)

        predicted = predicted.sigmoid()
        predicted = (predicted * 200) - 100

        if ctx.log_text:
            ctx.add_scalar("paths/seq_len", pos_len)

        if ctx.log_img:
            fig = plt.figure()
            length = lengths[0] - 1
            plt.plot(*target[0, 0:2, :length].detach().cpu(), label="target")
            plt.plot(*predicted[0, 0:2, :length].detach().cpu(), label="predicted")
            fig.legend()
            plt.gca().set_aspect("equal")
            ctx.add_figure("paths", fig)

        per_token_loss = F.huber_loss(predicted, target, reduction="none")
        per_token_loss *= mask.unsqueeze(1).float()
        position_loss = per_token_loss.mean(dim=(1, 2)) * 200

        return {
            "position": position_loss,
        }
