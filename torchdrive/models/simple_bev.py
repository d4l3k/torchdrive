"""
This code is copied and adapted from the Simple BEV reference implementation.

See https://github.com/aharley/simple_bev/blob/main/LICENSE for the LICENSE.

MIT License

Copyright (c) 2022 aharley

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

import torch
import torchvision
from pytorch3d.transforms.transform3d import Transform3d
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
from torchvision.models.resnet import resnet18

from torchdrive.amp import autocast

from torchdrive.data import Batch
from torchdrive.models.bev import BEVUpsampler
from torchdrive.models.bev_backbone import BEVBackbone
from torchdrive.transforms.simple_bev import lift_cam_to_voxel

EPS = 1e-4


def set_bn_momentum(model: nn.Module, momentum: float = 0.1) -> None:
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum


class UpsamplingConcat(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, scale_factor: int = 2
    ) -> None:
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=False
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # upsample doesn't support bfloat16 -- explicit cast
        x_to_upsample = x_to_upsample.float()
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)


class UpsamplingAdd(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, scale_factor: int = 2
    ) -> None:
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=False
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        # upsample doesn't support bfloat16 -- explicit cast
        x = x.float()
        x = self.upsample_layer(x)
        return x + x_skip


class FPN(nn.Module):
    """
    Implements the FPN used in the Simple BEV segnet decoder.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1: nn.Module = backbone.bn1
        self.relu: nn.Module = backbone.relu

        self.layer1: nn.Module = backbone.layer1
        self.layer2: nn.Module = backbone.layer2
        self.layer3: nn.Module = backbone.layer3

        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, in_channels, scale_factor=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [BS, in_channels, H, W]
        Returns:
            * [BS, in_channels, H, W]
            * [BS, 256, H/8, W/8]
        """
        # (H, W)
        skip_x = {"1": x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x["2"] = x
        x = self.layer2(x)
        skip_x["3"] = x

        # (H/8, W/8)
        x4 = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x4, skip_x["3"])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x["2"])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x["1"])

        return x, x4


# extends FPN to preserve state_dict keys
class Decoder(FPN):
    """
    Decoder implements the decoder for Simple BEV Segnet. This consumes the
    compressed BEV grid.
    """

    def __init__(
        self, in_channels: int, n_classes: int, predict_future_flow: bool
    ) -> None:
        super().__init__(in_channels=in_channels)
        self.predict_future_flow = predict_future_flow

        shared_out_channels = in_channels

        self.feat_head = nn.Sequential(
            nn.Conv2d(
                shared_out_channels,
                shared_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                shared_out_channels, shared_out_channels, kernel_size=1, padding=0
            ),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(
                shared_out_channels,
                shared_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(
                shared_out_channels,
                shared_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(
                shared_out_channels,
                shared_out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head: nn.Sequential = nn.Sequential(
                nn.Conv2d(
                    shared_out_channels,
                    shared_out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

    # pyre-ignore[15]: inconsistent override
    def forward(
        self,
        x: torch.Tensor,
        bev_flip_indices: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        b, c, h, w = x.shape

        x, _ = super().forward(x)

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(
                x[bev_flip2_index], [-2]
            )  # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        feat_output = self.feat_head(x)
        segmentation_output = self.segmentation_head(x)
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_future_output = (
            self.instance_future_head(x) if self.predict_future_flow else None
        )

        return {
            "raw_feat": x,
            "feat": feat_output.view(b, *feat_output.shape[1:]),
            "segmentation": segmentation_output.view(b, *segmentation_output.shape[1:]),
            "instance_center": instance_center_output.view(
                b, *instance_center_output.shape[1:]
            ),
            "instance_offset": instance_offset_output.view(
                b, *instance_offset_output.shape[1:]
            ),
            "instance_flow": instance_future_output.view(
                b, *instance_future_output.shape[1:]
            )
            if instance_future_output is not None
            else None,
        }


class ResNetEncoder(nn.Module):
    def __init__(self, C: int, resnet: torchvision.models.ResNet) -> None:
        super().__init__()
        self.C = C
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3: nn.Module = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x


class RegNetEncoder(nn.Module):
    def __init__(self, C: int, regnet: torchvision.models.RegNet) -> None:
        super().__init__()
        self.C = C
        self.backbone = nn.Sequential(regnet.stem, regnet.trunk_output[:-2])
        self.layer3: nn.Module = regnet.trunk_output[-2]

        ch3 = regnet.trunk_output[-2][0].proj[0].in_channels
        ch4 = regnet.trunk_output[-1][0].proj[0].in_channels

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(ch3 + ch4, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x


class Segnet(nn.Module):
    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        use_radar: bool = False,
        use_lidar: bool = False,
        use_metaradar: bool = False,
        do_rgbcompress: bool = True,
        rand_flip: bool = False,
        latent_dim: int = 128,
        encoder_type: str = "res101",
    ) -> None:
        """
        Args:
            grid_shape: tuple of (X, Y, Z) where Z is the vertical axis.
        """
        super(Segnet, self).__init__()
        assert encoder_type in ["res101", "res50", "effb0", "effb4"]

        X, Y, Z = grid_shape
        self.grid_shape = grid_shape
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.use_metaradar = use_metaradar
        self.do_rgbcompress = do_rgbcompress
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder: nn.Module = ResNetEncoder(
                feat2d_dim, torchvision.models.resnet101(pretrained=True)
            )
        elif encoder_type == "res50":
            self.encoder = ResNetEncoder(
                feat2d_dim, torchvision.models.resnet50(pretrained=True)
            )
        else:
            raise ValueError(f"invalid encoder type {encoder_type}")

        # BEV compressor
        if self.use_radar:
            if self.use_metaradar:
                self.bev_compressor: nn.Sequential = nn.Sequential(
                    nn.Conv2d(
                        feat2d_dim * Z + 16 * Z,
                        feat2d_dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(
                        feat2d_dim * Z + 1,
                        feat2d_dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
        elif self.use_lidar:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(
                    feat2d_dim * Z + Z,
                    feat2d_dim,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
        else:
            if self.do_rgbcompress:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(
                        feat2d_dim * Z,
                        feat2d_dim,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                        bias=False,
                    ),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                # use simple sum
                pass

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim, n_classes=1, predict_future_flow=False
        )

        # Weights
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(
        self,
        rgb_camXs: torch.Tensor,
        pix_T_cams: torch.Tensor,
        cam0_T_camXs: torch.Tensor,
        rad_occ_mem0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            B = batch size, S = number of cameras, C = 3, H = img height, W = img width
            rgb_camXs: (B,S,C,H,W)
                camera RGB data, should be between 0-1, will be normalized
            pix_T_cams: (B,S,4,4)
                camera intrinsics, normalized to height/width 1
            cam0_T_camXs: (B,S,4,4)
                camera extrinsics
            rad_occ_mem0:
                - None when use_radar = False, use_lidar = False
                - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
                - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
                - (B, 1, Z, Y, X) when use_lidar = True
        Returns:
            raw_feat
            feat
            segmentation
            instance_center
            instance_offset
        """
        B, S, C, H, W = rgb_camXs.shape
        assert C == 3
        X, Y, Z = self.grid_shape

        # rgb encoder
        device = rgb_camXs.device
        rgb_camXs_ = self.transform(rgb_camXs).flatten(0, 1)
        rand_flip = self.rand_flip
        if rand_flip:
            B0, _, _, _ = rgb_camXs_.shape
            rgb_flip_index = np.random.choice([0, 1], B0).astype(bool)
            rgb_camXs_[rgb_flip_index] = torch.flip(rgb_camXs_[rgb_flip_index], [-1])
        else:
            rgb_flip_index = None
        feat_camXs_ = self.encoder(rgb_camXs_).unflatten(0, (B, S))
        if rand_flip:
            feat_camXs_[rgb_flip_index] = torch.flip(feat_camXs_[rgb_flip_index], [-1])

        # unproject image feature to 3d grid
        feat_voxels, feat_valids = lift_cam_to_voxel(
            features=feat_camXs_.flatten(0, 1),
            K=pix_T_cams.flatten(0, 1),
            T=cam0_T_camXs.pinverse().flatten(0, 1),
            grid_shape=(X, Y, Z),
        )
        feat_voxels = feat_voxels.unflatten(0, (B, S))
        feat_valids = feat_valids.unflatten(0, (B, S))

        feat_mem = feat_voxels.sum(dim=1) / feat_valids.sum(dim=1).clamp(min=1)

        if self.rand_flip:
            bev_flip1_index = np.random.choice([0, 1], B).astype(bool)
            bev_flip2_index = np.random.choice([0, 1], B).astype(bool)
            feat_mem[bev_flip1_index] = torch.flip(feat_mem[bev_flip1_index], [-1])
            feat_mem[bev_flip2_index] = torch.flip(feat_mem[bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[bev_flip1_index] = torch.flip(
                    rad_occ_mem0[bev_flip1_index], [-1]
                )
                rad_occ_mem0[bev_flip2_index] = torch.flip(
                    rad_occ_mem0[bev_flip2_index], [-3]
                )
        else:
            bev_flip1_index = None
            bev_flip2_index = None

        # bev compressing
        if self.use_radar:
            assert rad_occ_mem0 is not None
            if not self.use_metaradar:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(
                    B, self.feat2d_dim * Y, Z, X
                )
                rad_bev = torch.sum(rad_occ_mem0, 3).clamp(
                    0, 1
                )  # squish the vertical dim
                feat_bev_ = torch.cat([feat_bev_, rad_bev], dim=1)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(
                    B, self.feat2d_dim * Y, Z, X
                )
                rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16 * Y, Z, X)
                feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
                feat_bev = self.bev_compressor(feat_bev_)
        elif self.use_lidar:
            assert rad_occ_mem0 is not None
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(
                B, self.feat2d_dim * Y, Z, X
            )
            rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
            feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
            feat_bev = self.bev_compressor(feat_bev_)
        else:  # rgb only
            if self.do_rgbcompress:
                feat_bev_ = feat_mem.permute(0, 1, 4, 2, 3).flatten(1, 2)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev = torch.sum(feat_mem, dim=3)

        # bev decoder
        out_dict = self.decoder(
            feat_bev, (bev_flip1_index, bev_flip2_index) if self.rand_flip else None
        )

        raw_e = out_dict["raw_feat"]
        feat_e = out_dict["feat"]
        seg_e = out_dict["segmentation"]
        center_e = out_dict["instance_center"]
        offset_e = out_dict["instance_offset"]

        return raw_e, feat_e, seg_e, center_e, offset_e

    def forward_batch(
        self,
        batch: Batch,
        frame: int,
        center: Tuple[float, float, float] = (-0.5, -0.5, 0),
        scale: float = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the model on a torchdrive Batch. See forward for information on the
        return values.

        Positions the vehicle in the center of the grid.

        Args:
            batch: the batch of data
            frame: the frame index in the batch to use for cameras
        """

        cameras = list(batch.T.keys())
        device = batch.weight.device

        rgb_camXs = torch.stack([batch.color[cam][:, frame] for cam in cameras], dim=1)
        pix_T_cams = torch.stack([batch.K[cam] for cam in cameras], dim=1)
        cam0_T_camXs = torch.stack([batch.T[cam] for cam in cameras], dim=1)

        center = tuple(a * b for a, b in zip(self.grid_shape, center))
        voxel_to_world = (
            Transform3d(device=device)
            .translate(*center)
            .scale(1 / scale)
            .get_matrix()
            .permute(0, 2, 1)
        )

        cam0_T_camXs = cam0_T_camXs.pinverse().matmul(voxel_to_world).pinverse()

        return self.forward(
            rgb_camXs=rgb_camXs,
            pix_T_cams=pix_T_cams,
            cam0_T_camXs=cam0_T_camXs,
        )


def segnet_rgb(grid_shape: Tuple[int, int, int], pretrained: bool = False) -> Segnet:
    """
    Instantiates a standard Segnet model with just RGB cameras.

    Args:
        grid_shape: tuple of (X, Y, Z) coordinates
        pretrained: whether to load a pretrained version of the model
    """
    m = Segnet(grid_shape=grid_shape)
    if pretrained:
        assert grid_shape[2] == 8, "pretrained model requires height 8"
        state_dict = torch.hub.load_state_dict_from_url(
            "https://drive.google.com/uc?export=download&id=18N3NDoeT3Z6T5x6Y_Qqs__KOAoh4pZY7&confirm=yes",
            map_location=torch.device("cpu"),
        )
        m.load_state_dict(state_dict["model_state_dict"])
    return m


class SegnetBackbone(BEVBackbone):
    """
    A BEV backbone using the Segnet projections and decoder.

    This extends Simple BEV to support multiple time frames.
    """

    def __init__(
        self,
        cam_dim: int,
        dim: int,
        hr_dim: int,
        grid_shape: Tuple[int, int, int],
        num_frames: int,
        scale: float,
        center: Tuple[float, float, float] = (-0.5, -0.5, 0),
        num_upsamples: int = 0,
        compile_fn: Callable[[nn.Module], nn.Module] = lambda m: m,
    ) -> None:
        super().__init__()

        assert dim == 256, "dim must equal intermediate"
        self.dim = dim
        self.grid_shape = grid_shape
        self.num_frames = num_frames
        self.scale = scale
        self.center: Tuple[int, int, int] = tuple(
            a * b for a, b in zip(center, self.grid_shape)
        )

        self.project = nn.ModuleList(
            [compile_fn(nn.Conv2d(cam_dim, cam_dim, 1)) for i in range(num_frames)]
        )
        self.fpn: nn.Module = compile_fn(FPN(dim))

        self.bev_compressor: nn.Module = compile_fn(
            nn.Sequential(
                nn.Conv2d(
                    cam_dim * grid_shape[-1],
                    dim,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(dim),
                nn.GELU(),
            )
        )
        self.upsample: nn.Module = compile_fn(
            BEVUpsampler(
                num_upsamples=num_upsamples,
                bev_shape=grid_shape[:2],
                dim=dim,
                output_dim=hr_dim,
            )
        )

    def forward(
        self, camera_features: Mapping[str, List[torch.Tensor]], batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        BS = batch.batch_size()
        S = len(camera_features) * self.num_frames
        device = batch.device()

        voxel_to_world = (
            Transform3d(device=device)
            .translate(*self.center)
            .scale(1 / self.scale)
            .get_matrix()
            .permute(0, 2, 1)
        )

        # compute relative positions to the last encode frame
        start_T = batch.cam_T[:, self.num_frames].unsqueeze(1)
        cam_T = start_T.pinverse().matmul(batch.cam_T)

        features = []
        Ks = []
        Ts = []
        # camera order doesn't matter for segnet
        for cam, time_feats in camera_features.items():
            for i, feats in enumerate(time_feats):
                with autocast():
                    features.append(self.project[i](feats))
                Ks.append(batch.K[cam])
                T = batch.T[cam].pinverse()
                T = T.matmul(cam_T[:, i])
                T = T.matmul(voxel_to_world)
                Ts.append(T)

        feature = torch.stack(features, dim=1)
        K = torch.stack(Ks, dim=1)
        T = torch.stack(Ts, dim=1)
        feat_mem = checkpoint(lift_cam_to_voxel_mean, feature, K, T, self.grid_shape)

        with autocast():
            # flatten voxel grid to bev
            feat_mem = feat_mem.flatten(1, 2)
            feat_mem = self.bev_compressor(feat_mem)

            # run through FPN
            x, x4 = self.fpn(feat_mem)
            x = self.upsample(x)
            return x, x4


def lift_cam_to_voxel_mean(
    features: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Runs lift_cam_to_voxel and takes the weighted mean of the features along the
    second batch dimension.

    Args:
       features: [batch_size, S, channels, height, width]
       K: [batch_size, S, 4, 4]
       T: [batch_size, S, 4, 4]
    Returns: the merged grids [batch_size, channels, z, x, y]
    """

    BS, S, _, _ = K.shape

    feat_voxels, feat_valids = lift_cam_to_voxel(
        features=features.flatten(0, 1),
        K=K.flatten(0, 1),
        T=T.flatten(0, 1),
        grid_shape=grid_shape,
    )

    with autocast():
        feat_voxels = feat_voxels.unflatten(0, (BS, S))
        feat_valids = feat_valids.unflatten(0, (BS, S))

        # merge voxel grids
        feat_mem = feat_voxels.sum(dim=1) / feat_valids.sum(dim=1).clamp(min=1)
        feat_mem = feat_mem.permute(0, 1, 4, 2, 3)
    return feat_mem
