from torchvision.models import resnet18, ResNet18_Weights
from torch import nn
import torch


class posenet18(nn.Module):
    def __init__(self, num_images: int, output_channels: int) -> nn.Module:
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.net.conv1 = nn.Conv2d(
            num_images * 3,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.net.fc = nn.Linear(512, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1, 2).float()
        return self.net(x)
