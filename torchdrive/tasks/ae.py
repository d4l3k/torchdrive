from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from torchdrive.amp import autocast
from torchdrive.data import Batch
from torchdrive.losses import projection_loss
from torchdrive.models.bev import GridTransformer
from torchdrive.models.regnet import UpsamplePEBlock
from torchdrive.tasks.bev import BEVTask, Context
from torchdrive.transforms.img import normalize_img, render_color


class AETask(BEVTask):
    """
    Per camera autoencoder task. Outputs a smaller version of the original
    image.
    """

    def __init__(
        self,
        cameras: List[str],
        cam_shape: Tuple[int, int],
        bev_shape: Tuple[int, int],
        dim: int,
    ) -> None:
        super().__init__()

        self.cam_shape = cam_shape
        self.cameras = cameras

        transformer_shape = (cam_shape[0] // 16, cam_shape[1] // 16)
        self.out_shape: Tuple[int, int] = (cam_shape[0] // 8, cam_shape[1] // 8)

        self.decoders: nn.ModuleDict = nn.ModuleDict(
            {
                cam: nn.Sequential(
                    GridTransformer(
                        input_shape=bev_shape,
                        output_shape=transformer_shape,
                        dim=dim,
                        num_inputs=1,
                    ),
                    UpsamplePEBlock(
                        in_ch=dim, out_ch=dim // 2, input_shape=transformer_shape
                    ),
                    nn.Conv2d(dim // 2, 3, kernel_size=1),
                )
                for cam in cameras
            }
        )

    def forward(
        self, ctx: Context, batch: Batch, bev: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        BS = len(batch.distances)
        device = bev.device
        num_frames = batch.distances.shape[1]
        losses = {}
        for cam in self.cameras:
            target = batch.color[cam][:, ctx.start_frame]
            with autocast():
                target = F.interpolate(
                    target, size=self.out_shape, mode="bilinear", align_corners=False
                )
                mask = F.interpolate(
                    batch.mask[cam],
                    size=self.out_shape,
                    mode="bilinear",
                    align_corners=False,
                )
                out = self.decoders[cam]([bev])

            loss = projection_loss(out, target, mask) * 20
            losses[f"ssim/{cam}"] = loss

            if ctx.log_img:
                ctx.add_image(
                    f"render/{cam}",
                    normalize_img(
                        torch.cat(
                            (
                                out[0],
                                target[0],
                            ),
                            dim=2,
                        )
                    ),
                )
                ctx.add_image(
                    f"proj_loss/{cam}",
                    render_color(loss[0][0]),
                )
        return losses
