import os.path
from typing import Dict, List, Tuple, Union

import mmcv

import torch
from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.models import build_detector
from mmdet.models.detectors import TwoStageDetector
from torchvision import transforms

from torchdrive.transforms.img import normalize_img_cuda


class BDD100KDet:
    """
    This is a helper class intended for doing semantic object detections during
    training. This is explicitly not an nn.Module and isn't trainable. If you
    want to fine tune a model please use the underlying mmdet model directly.
    """

    LABELS = [
        "pedestrian",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "traffic light",
        "traffic sign",
        "<no match>",
    ]

    def __init__(
        self,
        device: torch.device,
        half: bool = True,
        config: str = "cascade_rcnn_convnext-t_fpn_fp16_3x_det_bdd100k.py",
    ) -> None:
        cfg_file = os.path.join(
            os.path.dirname(__file__),
            "../../third-party/bdd100k-models/det/configs/det/",
            config,
        )

        self.half = half

        cfg = mmcv.Config.fromfile(cfg_file)
        cfg.model.pretrained = None

        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None and half:
            wrap_fp16_model(model)

        # pyre-fixme[6]: map_location device
        checkpoint = load_checkpoint(model, cfg.load_from, map_location=device)
        model = revert_sync_batchnorm(model)
        model = model.eval()
        if half:
            model = model.half()

        # pyre-fixme[8]
        self.model: TwoStageDetector = model.to(device)

        # anchors are stupidly on CPU device -- force them over early
        anchors = self.model.rpn_head.anchor_generator.base_anchors
        for i, a in enumerate(anchors):
            anchors[i] = a.to(device)

        self._transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def transform(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Dict[str, Union[Tuple[int, int, int], int]]]]:
        img = normalize_img_cuda(img)
        if self.half:
            img = img.half()
        img: torch.Tensor = self._transform(img)
        bs, ch, h, w = img.shape
        shape = (h, w, ch)
        img_metas = [{"img_shape": shape, "ori_shape": shape, "scale_factor": 1}] * bs
        return img, img_metas

    def __call__(self, img: torch.Tensor) -> List[object]:
        """
        Normalizes the image and then executes it.

        Args:
           img: torch tensor [BS, 3, H, W]
        Returns:
           a list following:
           (BS, classes=10, # examples per class, 4)
        """
        with torch.cuda.amp.autocast():
            img, img_metas = self.transform(img)
            if self.half:
                img = img.half()
            return self.model.simple_test(img, img_metas=img_metas)
