import os.path

import mmcv
import torch
from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.models.segmentors import BaseSegmentor
from torchvision import transforms

from torchdrive.transforms.img import normalize_img_cuda


class BDD100KSemSeg:
    """
    This is a helper class for doing semantic segmentation of an input image
    during training. This is explicitly not an nn.Module and isn't trainable. If
    you want to fine tune a model please use the underlying mmseg model
    directly.
    """

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

    def __init__(
        self,
        device: torch.device,
        half: bool = True,
        config: str = "upernet_convnext-t_fp16_512x1024_80k_sem_seg_bdd100k.py",
    ) -> None:
        cfg_file = os.path.join(
            os.path.dirname(__file__),
            "../../third-party/bdd100k-models/sem_seg/configs/sem_seg/",
            config,
        )
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

        # pyre-fixme[8]
        self.model: BaseSegmentor = model.to(device)
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            img = normalize_img_cuda(img)
            if self.half:
                img = img.half()
            img = self.transform(img)
            return self.model.encode_decode(img, img_metas=[])
