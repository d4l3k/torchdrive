import json
import os.path
from typing import Callable, cast, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import draw_bounding_boxes

from torchdrive.amp import autocast
from torchdrive.data import Batch
from torchdrive.losses import generalized_box_iou
from torchdrive.matcher import HungarianMatcher
from torchdrive.models.det import DetBEVTransformerDecoder
from torchdrive.tasks.bev import BEVTask, Context
from torchdrive.transforms.bboxes import (
    bboxes3d_to_points,
    decode_bboxes3d,
    points_to_bboxes2d,
)
from torchdrive.transforms.img import normalize_img

# TARGET_SIZES - class: (l, w, h)
TARGET_SIZES: Dict[int, Tuple[float, float, float]] = {
    # pedestrian
    0: (0.2, 0.4, 1.68),
    # car - Model Y
    2: (4.751, 2.129, 1.624),
    # truck, 26ft, box
    3: (10.39, 2.34, 3.99),
    # bus, 40ft, New Flyer Xcelsior XE40
    4: (12.19, 2.59, 3.2),
    # motorcycle
    6: (2.159, 1.0, 1.27),
    # bicycle
    7: (1.75, 0.4, 1.05),
    # traffic light
    8: (0.5, 0.5, 1.1),
}


class DetTask(BEVTask):
    def __init__(
        self,
        cameras: List[str],
        cam_shape: Tuple[int, int],
        bev_shape: Tuple[int, int],
        dim: int,
        device: torch.device,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        super().__init__()

        self.cam_shape = cam_shape
        self.cameras = cameras

        decoder = DetBEVTransformerDecoder(
            bev_shape=bev_shape,
            dim=dim,
        )
        self.num_classes: int = decoder.num_classes
        self.decoder: nn.Module = compile_fn(decoder)

        # not a module -- not saved
        self.matcher = HungarianMatcher()
        self.decode_bboxes3d = compile_fn(decode_bboxes3d)
        self.bboxes3d_to_points = compile_fn(bboxes3d_to_points)
        self.points_to_bboxes2d = compile_fn(points_to_bboxes2d)

    def forward(
        self, ctx: Context, batch: Batch, bev: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        BS = len(batch.distances)
        device = bev.device
        num_frames = batch.distances.shape[1]

        with torch.autograd.profiler.record_function("decoding"):
            classes_logits, bboxes3d = self.decoder(bev)
            classes_softmax = F.softmax(classes_logits.float(), dim=-1)

            with autocast():
                xyz, vel, sizes = self.decode_bboxes3d(bboxes3d)

        if ctx.log_img:
            json_path = os.path.join(ctx.output, f"det_{ctx.global_step}.json")
            with open(json_path, "w") as f:
                json.dump(
                    {
                        "classes": classes_softmax[0].tolist(),
                        "xyz": xyz[0].tolist(),
                        "vel": vel[0].tolist(),
                        "sizes": sizes[0].tolist(),
                    },
                    f,
                )

        num_queries = classes_logits.shape[1]
        assert num_queries == 100

        if ctx.log_text:
            ctx.add_scalars(
                "classes/minmax",
                {
                    "max": classes_softmax.max(),
                    "min": classes_softmax.min(),
                    "mean": classes_softmax.mean(),
                },
            )

        h, w = self.cam_shape
        normalize_coords = torch.tensor(
            (w, h, w, h), dtype=torch.float32, device=device
        )

        view_frames = range(0, min(5, num_frames))

        losses = {}
        unmatched_queries = torch.full((BS, num_queries), True, dtype=torch.bool)

        with torch.autograd.profiler.record_function("matching"):
            for frame in view_frames:
                dur = batch.frame_time[:, frame] - batch.frame_time[:, ctx.start_frame]
                points3d = self.bboxes3d_to_points(bboxes3d, time=dur)

                for cam in self.cameras:
                    K = batch.K[cam]
                    # always predict from same position so predictions are relative
                    # velocities
                    T = batch.T[cam]

                    pix_coords, bboxes2d, invalid_mask = self.points_to_bboxes2d(
                        points3d, K, T, w, h
                    )

                    primary_color = batch.color[cam][:, frame]
                    target_preds = [batch.det[cam][i][frame] for i in range(BS)]

                    targets: List[Dict[str, torch.Tensor]] = []
                    num_targets = 0

                    mask = batch.mask[cam]
                    for i, bpreds in enumerate(target_preds):
                        num_target_boxes = len(bpreds)
                        labels = []
                        boxes = []
                        for kls, class_boxes in enumerate(bpreds):
                            tkls = torch.tensor(kls, device=device)
                            for box in class_boxes:
                                # don't use low confidence predictions
                                p = box[4]
                                if p < 0.5:
                                    continue

                                # filter out boxes in mask
                                x = int((box[0] + box[2]) // 2)
                                y = int((box[1] + box[3]) // 2)
                                mask_p = mask[i, 0, y, x]
                                if mask_p < 0.5:
                                    continue

                                labels.append(tkls)
                                boxes.append(box[:4].to(device, non_blocking=True))

                        labels = (
                            torch.stack(labels)
                            if len(labels) > 0
                            else torch.zeros(0, dtype=torch.int64, device=device)
                        )
                        boxes = (
                            torch.stack(boxes)
                            if len(boxes) > 0
                            else torch.zeros(0, 4, device=device)
                        )
                        targets.append(
                            {
                                "labels": labels,
                                "boxes": boxes / normalize_coords,
                            }
                        )
                        num_targets += len(boxes)

                    if num_targets == 0:
                        continue
                    outputs = {
                        "pred_logits": classes_logits,
                        "pred_boxes": bboxes2d / normalize_coords,
                    }

                    pairs = self.matcher(
                        outputs=outputs,
                        targets=targets,
                        invalid_mask=invalid_mask,
                    )

                    num_boxes = sum(len(t["labels"]) for t in targets)
                    num_boxes = torch.as_tensor(
                        [num_boxes], dtype=torch.float, device=device
                    )
                    # TODO: for distributed training all reduce?
                    num_boxes = cast(int, torch.clamp(num_boxes, min=1).item())

                    if ctx.log_text:
                        ctx.add_scalar(
                            f"num_boxes/{cam}/{frame}",
                            num_boxes,
                        )

                    for batchi, (pred_idxs, target_idxs) in enumerate(pairs):
                        # mark which queries were used so we can apply penalty
                        unmatched_queries[batchi, pred_idxs] = False

                    losses[f"loss_labels/{cam}/{frame}"] = self.loss_labels(
                        outputs, targets, pairs, num_boxes
                    )

                    # clamp outputs to image space -- we do this after bbox matching
                    # to avoid matching very far away boxes
                    outputs["pred_boxes"][..., (0, 1)].clamp_(min=0, max=w)
                    outputs["pred_boxes"][..., (2, 3)].clamp_(min=0, max=h)

                    boxlosses = self.loss_boxes(outputs, targets, pairs, num_boxes)
                    for k, v in boxlosses.items():
                        losses[f"loss_{k}/{cam}/{frame}"] = v

                    if ctx.log_img:
                        color = (
                            normalize_img(primary_color.contiguous()).clamp(
                                min=0, max=1
                            )
                            * 255
                        ).byte()
                        color_target = draw_bounding_boxes(
                            color[0],
                            boxes=targets[0]["boxes"] * normalize_coords,
                            labels=[str(i.item()) for i in targets[0]["labels"]],
                        )
                        idxs = pairs[0][0]
                        color_pred = draw_bounding_boxes(
                            color[0],
                            boxes=bboxes2d[0, idxs],
                            labels=[
                                str(i.item())
                                for i in classes_logits[0, idxs].argmax(dim=-1)
                            ],
                        )

                        ctx.add_image(
                            f"{cam}/{frame}/target",
                            color_target,
                        )
                        ctx.add_image(
                            f"{cam}/{frame}/pred",
                            color_pred,
                        )

        unmatched_classes = classes_logits[unmatched_queries]
        # set target to num_classes for unmatched ones
        target_classes = torch.full(
            unmatched_classes.shape[:1],
            self.num_classes,
            dtype=torch.int64,
            device=device,
        )

        classes = classes_logits.argmax(dim=-1)
        if False:
            with torch.autograd.profiler.record_function("dim"):
                LOSS_DIM_WEIGHT = 0.1
                for k in range(self.num_classes + 1):
                    idxs = classes == k
                    class_sizes = sizes[idxs]
                    if class_sizes.numel() == 0:
                        continue
                    length_widths = class_sizes[..., :2]
                    heights = class_sizes[..., 2]

                    if k in TARGET_SIZES:
                        numel = heights.shape[0]
                        target_l, target_w, target_h = TARGET_SIZES[k]
                        losses[f"lossdim/{k}/height"] = (
                            F.l1_loss(
                                heights, torch.tensor(target_h, device=device).expand(numel)
                            )
                            * LOSS_DIM_WEIGHT
                        )
                        losses[f"lossdim/{k}/length_width"] = (
                            F.l1_loss(
                                length_widths.prod(dim=-1),
                                torch.tensor(target_l * target_w, device=device).expand(
                                    numel
                                ),
                            )
                            * LOSS_DIM_WEIGHT
                        )

        if ctx.log_text:
            log_sizes = {}
            log_heights = {}
            for k in range(self.num_classes + 1):
                idxs = classes == k
                class_sizes = sizes[idxs]
                if class_sizes.numel() == 0:
                    continue
                length_widths = class_sizes[..., :2]
                heights = class_sizes[..., 2]

                log_sizes[str(k)] = length_widths.mean()
                log_heights[str(k)] = heights.mean()
            ctx.add_scalars(
                "classes/sizes",
                log_sizes,
            )
            ctx.add_scalars(
                "classes/heights",
                log_heights,
            )

            num_matched = unmatched_queries.logical_not().count_nonzero()
            num_predicted = (classes != self.num_classes).count_nonzero()

            ctx.add_scalars(
                "classes/num_boxes3d",
                {"predicted": num_predicted, "matched": num_matched},
            )

        if len(unmatched_classes) > 0:
            losses["unmatched"] = (
                F.cross_entropy(unmatched_classes, target_classes) * 10
            )

        losses = {k: v * 100 for k, v in losses.items()}

        return losses

    # These are modified from
    # https://github.com/facebookresearch/detr/blob/main/models/detr.py#L108
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
    # Apache-2.0 license

    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> torch.Tensor:
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]

        only does matched cases and no nonmatched penalty
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        src_logits_o = src_logits[idx]
        assert len(target_classes_o) > 0, "can't compute loss_label with no items"

        loss_ce = F.cross_entropy(src_logits_o.float(), target_classes_o)
        return loss_ce

    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() * 5 / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        losses["loss_giou"] = loss_giou.sum() * 2 / num_boxes
        return losses

    def _get_src_permutation_idx(
        self, indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
