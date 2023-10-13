import os.path
from typing import Callable

import torch
from torch import nn
from torchvision import transforms

from torchdrive.transforms.img import normalize_img_cuda

TS_MODELS = {
    "upernet_convnext-t_fp16_512x1024_80k_sem_seg_bdd100k.py": (
        "https://drive.google.com/uc?export=download&id=1iXRlXZNc1B3OmI9wbyMrVA1y0IC-qFTO&confirm=yes"
    ),
}


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

    # dynamic object indexes when filtered by NON_SKY
    DYNAMIC_NON_SKY = tuple(
        idx-1 for idx in DYNAMIC
    )

    NON_SKY = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        # 10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    )

    # The observed relative pixel wise frequency of each of the different
    # classes.
    CLASS_FREQUENCY = [
        1.8476e-01,
        4.9409e-02,
        2.5560e-01,
        9.4501e-03,
        5.8030e-02,
        9.4640e-03,
        6.0450e-04,
        2.3885e-03,
        2.0300e-01,
        4.2501e-02,
        1.0847e-01,
        2.4595e-03,
        7.1098e-05,
        5.0693e-02,
        1.7129e-02,
        5.1975e-03,
        1.0447e-04,
        1.8954e-04,
        4.6450e-04,
    ]

    def __init__(
        self,
        device: torch.device,
        half: bool = True,
        config: str = "upernet_convnext-t_fp16_512x1024_80k_sem_seg_bdd100k.py",
        mmlab: bool = True,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        if device == torch.device("cpu"):
            # half not supported on CPU
            half = False
        self.half = half

        if not mmlab:
            path = TS_MODELS[config]
            model: nn.Module = torch.hub.load_state_dict_from_url(  # pyre-ignore[9]
                path, map_location=device, file_name=config
            )
        else:
            import mmcv
            from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm
            from mmcv.runner import load_checkpoint
            from mmseg.models import build_segmentor

            cfg_file = os.path.join(
                os.path.dirname(__file__),
                "../../third-party/bdd100k-models/sem_seg/configs/sem_seg/",
                config,
            )

            cfg = mmcv.Config.fromfile(cfg_file)
            cfg.model.pretrained = None
            cfg.model.train_cfg = None
            model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))

            # pyre-fixme[6]: map_location device
            checkpoint = load_checkpoint(model, cfg.load_from, map_location=device)
            model = revert_sync_batchnorm(model)
            # pyre-fixme[8]: attribute used as type
            model.forward = model.forward_dummy

        model = model.eval()
        if half:
            model = model.half()
        model = model.to(device)
        self.orig_model = model
        model = compile_fn(model)
        self.model: nn.Module = model
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
            return self.model(img)


if __name__ == "__main__":
    dtype = torch.half
    device = torch.device("cuda")
    m = BDD100KSemSeg(device=device, mmlab=False, compile_fn=torch.compile)
    model = m.model
    inp = torch.rand(2, 3, 120, 240, device=device)
    m(inp)
