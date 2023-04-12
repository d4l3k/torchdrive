"""
Implementation of upernet

This is adapted from
https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py

MIT License

Copyright (c) 2020 Yassine

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(
        self, in_channels: int, bin_sizes: Tuple[int, int, int, int] = (1, 2, 4, 6)
    ) -> None:
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList(
            [self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + (out_channels * len(bin_sizes)),
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def _make_stages(
        self, in_channels: int, out_channels: int, bin_sz: int
    ) -> nn.Module:
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend(
            [
                F.interpolate(
                    stage(features), size=(h, w), mode="bilinear", align_corners=True
                )
                for stage in self.stages
            ]
        )
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        output_stride: int = 16,
        backbone: str = "resnet101",
        pretrained: bool = True,
    ) -> None:
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.initial: nn.Sequential = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            # initialize_weights(self.initial)
        else:
            self.initial = nn.Sequential(*list(model.children())[:4])

        self.layer1: nn.Module = model.layer1
        self.layer2: nn.Module = model.layer2
        self.layer3: nn.Module = model.layer3
        self.layer4: nn.Module = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)
        else:
            raise RuntimeError("invalid stride")

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if "conv1" in n and (backbone == "resnet34" or backbone == "resnet18"):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif "conv2" in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif "downsample.0" in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if "conv1" in n and (backbone == "resnet34" or backbone == "resnet18"):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif "conv2" in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif "downsample.0" in n:
                m.stride = (s4, s4)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.initial(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


def up_and_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        F.interpolate(
            x, size=(y.size(2), y.size(3)), mode="bilinear", align_corners=True
        )
        + y
    )


class FPN_fuse(nn.Module):
    def __init__(
        self,
        feature_channels: Tuple[int, int, int, int] = (256, 512, 1024, 2048),
        fpn_out: int = 256,
    ) -> None:
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList(
            [
                nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                for ft_size in feature_channels[1:]
            ]
        )
        self.smooth_conv = nn.ModuleList(
            [nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
            * (len(feature_channels) - 1)
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(
                len(feature_channels) * fpn_out,
                fpn_out,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:

        features[1:] = [
            conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)
        ]
        P = [
            up_and_add(features[i], features[i - 1])
            for i in reversed(range(1, len(features)))
        ]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [
            F.interpolate(feature, size=(H, W), mode="bilinear", align_corners=True)
            for feature in P[1:]
        ]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


def upernet_resnet(
    num_classes: int, backbone: str = "resnet101", pretrained: bool = True
) -> nn.Module:
    backbone_module = ResNet(in_channels=3, backbone=backbone, pretrained=pretrained)
    if backbone == "resnet34" or backbone == "resnet18":
        feature_channels = (64, 128, 256, 512)
    else:
        feature_channels = (256, 512, 1024, 2048)
    return UperNet(
        num_classes, backbone_module, feature_channels, fpn_out=feature_channels[0]
    )


class UperNet(nn.Module):
    # Implementing only the object path
    def __init__(
        self,
        num_classes: int,
        backbone: nn.Module,
        feature_channels: Tuple[int, int, int, int],
        fpn_out: int = 256,
    ) -> None:
        super(UperNet, self).__init__()

        self.backbone = backbone
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = (x.size()[2], x.size()[3])

        features = self.backbone(x)
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=input_size, mode="bilinear")
        return x
