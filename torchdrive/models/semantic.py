import os.path
from typing import Callable

import mmcv
import torch
from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from torch import nn
from torchvision import transforms

from torchdrive.transforms.img import normalize_img_cuda


class BDD100KSemSeg:
    """
    This is a helper class for doing semantic segmentation of an input image
    during training. This is explicitly not an nn.Module and isn't trainable. If
    you want to fine tune a model please use the underlying mmseg model
    directly.
    """

    # all semantic labels
    LABELS = {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic light",
        7: "traffic sign",
        8: "vegetation",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle",
    }

    # dynamic objects
    DYNAMIC = (
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    )

    # interesting is non-road, non-sky objects
    INTERESTING = (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    )

    def __init__(
        self,
        device: torch.device,
        half: bool = True,
        config: str = "upernet_convnext-t_fp16_512x1024_80k_sem_seg_bdd100k.py",
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        cfg_file = os.path.join(
            os.path.dirname(__file__),
            "../../third-party/bdd100k-models/sem_seg/configs/sem_seg/",
            config,
        )

        if device == torch.device("cpu"):
            # half not supported on CPU
            half = False
        self.half = half

        cfg = mmcv.Config.fromfile(cfg_file)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))

        # pyre-fixme[6]: map_location device
        checkpoint = load_checkpoint(model, cfg.load_from, map_location=device)
        model = revert_sync_batchnorm(model)
        model = model.eval()
        if half:
            model = model.half()

        model = model.to(device)
        # pyre-fixme[6]: nn.Module
        self.model: nn.Module = compile_fn(model.encode_decode)
        self.transform: nn.Module = compile_fn(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        # pyre-fixme[6]: nn.Module
        self.normalize: nn.Module = compile_fn(normalize_img_cuda)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img = self.normalize(img)
            if self.half:
                img = img.half()
            img = self.transform(img)
            return self.model(img, img_metas=[])
