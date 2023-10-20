import torch
from torch import nn


class MLP(nn.Module):
    """
    MLP is a multilayer perceptron.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [BS, input_dim, queries]
        Returns:
            [BS, output_dim, queries]
        """
        return self.decoder(x.permute(0, 2, 1)).permute(0, 2, 1)
