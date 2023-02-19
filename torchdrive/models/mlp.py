import torch
from torch import nn


class ConvMLP(nn.Module):
    """
    ConvMLP is a multilayer perceptron implemented as a set of 1d filter size 1
    convolutions so you can process many of them at once.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [BS, input_dim, queries]
        Returns:
            [BS, output_dim, queries]
        """
        return self.decoder(x)
