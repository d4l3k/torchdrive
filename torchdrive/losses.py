from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchvision.ops.boxes import box_area


def tvl1_loss(voxel: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Computes the voxel opacity total variation loss from
    https://research.facebook.com/publications/neural-volumes-learning-dynamic-renderable-volumes-from-images/
    https://github.com/facebookresearch/neuralvolumes/blame/main/models/decoders/voxel1.py#L225

    Arguments:
        voxel: 4d dense voxel grid: [BS, X, Y, Z], values 0-1

    Returns:
        losses: [BS]
    """
    assert voxel.ndim == 4, voxel.shape
    logalpha = torch.log(eps + voxel)

    return torch.mean(
        torch.sqrt(
            # pyre-fixme[6]: expected Tensor but got float
            eps
            +
            # pyre-fixme[58]: ** not supported for Tensor and int
            (logalpha[:, :-1, :-1, 1:] - logalpha[:, :-1, :-1, :-1]) ** 2
            +
            # pyre-fixme[58]: ** not supported for Tensor and int
            (logalpha[:, :-1, 1:, :-1] - logalpha[:, :-1, :-1, :-1]) ** 2
            +
            # pyre-fixme[58]: ** not supported for Tensor and int
            (logalpha[:, 1:, :-1, :-1] - logalpha[:, :-1, :-1, :-1]) ** 2
        ),
        dim=[1, 2, 3],
    )


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images

    From:
    https://github.com/nianticlabs/monodepth2/blob/master/layers.py#L218
    """

    def __init__(self) -> None:
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1: float = 0.01**2
        self.C2: float = 0.03**2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def smooth_loss(disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness

    From:
    https://github.com/nianticlabs/monodepth2/blob/master/layers.py#L202
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(
        torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True
    )
    grad_img_y = torch.mean(
        torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True
    )

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def losses_backward(
    losses: Dict[str, torch.Tensor],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    weights: Optional[torch.Tensor] = None,
) -> None:
    """
    Computes the backward function weighted to the weights in the batch and
    updates the value in the provided dictionary to a non-gradient one.

    Args:
        losses: dictionary of str to tensor losses
    """
    weighted_losses = {}
    for k, v in losses.items():
        if not v.requires_grad:
            continue
        v = v.float()
        if weights is not None:
            assert v.shape == weights.shape, f"{k} {v.shape}"
            v *= weights
        weighted_losses[k] = v.mean()

    if len(weighted_losses) == 0:
        return

    losses.update({k: v.detach() for k, v in weighted_losses.items()})
    loss = sum(weighted_losses.values())
    if scaler:
        loss = scaler.scale(loss)
    loss.backward()


# modified from torchvision to also return the union
def box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    https://github.com/facebookresearch/detr/blob/main/LICENSE
    Apache 2.0 License
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)

    https://github.com/facebookresearch/detr/blob/main/LICENSE
    Apache 2.0 License
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
