import os.path
from typing import Callable, Dict, List, Tuple, Union

import torch
from torch import nn
from torchvision import transforms

from torchdrive.amp import autocast
from torchdrive.attention import attention
from torchdrive.models.mlp import MLP
from torchdrive.models.regnet import ConvPEBlock
from torchdrive.positional_encoding import apply_sin_cos_enc2d
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
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        import mmcv
        from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm
        from mmcv.runner import load_checkpoint, wrap_fp16_model
        from mmdet.models import build_detector
        from mmdet.models.detectors import TwoStageDetector

        cfg_file = os.path.join(
            os.path.dirname(__file__),
            "../../third-party/bdd100k-models/det/configs/det/",
            config,
        )

        if device == torch.device("cpu"):
            # half not supported on CPU
            half = False
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

        model = model.to(device)
        # pyre-fixme[6]: invalid parameter
        self.model: nn.Module = compile_fn(model.simple_test)

        if isinstance(model, TwoStageDetector):
            # anchors are stupidly on CPU device -- force them over early
            anchors = model.rpn_head.anchor_generator.base_anchors
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

    def __call__(self, img: torch.Tensor) -> List[List[torch.Tensor]]:
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
            return self.model(img, img_metas=img_metas)


class DetBEVDecoder(nn.Module):
    """
    BEV based detection decoder. Consumes a BEV grid and generates detections
    using a cross attention decoder.
    """

    def __init__(
        self,
        bev_shape: Tuple[int, int],
        dim: int,
        num_queries: int = 100,
        num_heads: int = 12,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.bev_shape = bev_shape
        self.dim = dim
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.num_queries = num_queries

        self.bev_project = ConvPEBlock(dim, dim, bev_shape, depth=1)

        self.query_embed = nn.Embedding(num_queries, dim)
        self.kv_encoder = nn.Sequential(
            nn.Conv1d(dim, 2 * dim, 1),
        )

        self.bbox_decoder = MLP(dim, 128, 9, num_layers=3)
        self.class_decoder = nn.Conv1d(dim, num_classes + 1, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            feats: (BS, dim, x, y)
        Returns:
            classes: logits (BS, num_queries, 11)
            bboxes: 0-1 (BS, num_queries, 9)
        """
        BS = len(x)
        with autocast():
            x = self.bev_project(x)

            query = self.query_embed.weight.to(x.dtype).expand(BS, -1, -1)
            q_seqlen = self.num_queries

            x = apply_sin_cos_enc2d(x)
            x = x.reshape(*x.shape[:2], -1)
            kv = self.kv_encoder(x).permute(0, 2, 1)

            bev = attention(query, kv, dim=self.dim, num_heads=self.num_heads)
            bev = bev.reshape(BS, self.num_queries, self.dim)
            # (BS, ch, num_queries)
            bev = bev.permute(0, 2, 1)

            # (BS, num_queries, ch)
            bboxes = self.bbox_decoder(bev).permute(0, 2, 1)
            classes = self.class_decoder(bev).permute(0, 2, 1)  # logits
        bboxes = bboxes.float().sigmoid()  # normalized 0 to 1

        return classes, bboxes


class DetBEVTransformerDecoder(nn.Module):
    """
    BEV based detection decoder. Consumes a BEV grid and generates detections
    using a detr style transformer.
    """

    def __init__(
        self,
        bev_shape: Tuple[int, int],
        dim: int,
        num_queries: int = 100,
        num_heads: int = 8,
        num_classes: int = 10,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()

        self.bev_shape = bev_shape
        self.dim = dim
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.num_queries = num_queries

        self.query_embed = nn.Embedding(num_queries, dim)
        self.bev_project = nn.Conv2d(dim, dim, 1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            dim_feedforward=dim_feedforward,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.bbox_decoder = MLP(dim, dim, 9, num_layers=4)
        self.class_decoder = nn.Conv1d(dim, num_classes + 1, 1)

    def forward(self, bev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            feats: (BS, dim, x, y)
        Returns:
            classes: logits (BS, num_queries, 11)
            bboxes: 0-1 (BS, num_queries, 9)
        """
        BS = len(bev)
        with autocast():
            bev = self.bev_project(bev)
            bev = apply_sin_cos_enc2d(bev)

            target = self.query_embed.weight.expand(BS, -1, -1)
            bev = bev.flatten(2, 3).permute(0, 2, 1)
            out = self.transformer_decoder(
                tgt=target,
                memory=bev,
            )

            out = out.permute(0, 2, 1)  # (BS, ch, num_queries)

            bboxes = self.bbox_decoder(out).permute(0, 2, 1)
            classes = self.class_decoder(out).permute(0, 2, 1)  # logits
        bboxes = bboxes.float().sigmoid()  # normalized 0 to 1

        return classes, bboxes


if __name__ == "__main__":
    m = BDD100KDet(device=torch.device("cpu"))
    model = m.model.__self__
    img, img_meta = m.transform(torch.rand(2, 3, 120, 240))
    img_meta_list = [img_meta]
    original_forward = model.forward

    def forward(imgs):
        return original_forward(
            [imgs], img_metas=img_meta_list, return_loss=False, rescale=False
        )

    model.forward = forward
    # model.forward = partial(
    #        model.forward,
    #        img_metas=img_meta_list,
    #        return_loss=False,
    #        rescale=False)
    # model.forward.__globals__ = original_forward.__globals__
    # model.forward.__code__ = original_forward.__code__
    # model.forward.__closure__ = original_forward.__closure__

    model(img)

    # scripted = torch.jit.script(model, example_inputs=[[[img]]])
    scripted = torch.jit.trace(model, img)

    breakpoint()
