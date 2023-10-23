import torch
from torch import nn


class MLP(nn.Module):
    """
    MLP is a multilayer perceptron.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> None:
        super().__init__()

        assert num_layers >= 2

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]

        for i in range(num_layers - 2):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ]

        layers.append(
            nn.Linear(hidden_dim, output_dim),
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [BS, input_dim, queries]
        Returns:
            [BS, output_dim, queries]
        """
        return self.decoder(x.permute(0, 2, 1)).permute(0, 2, 1)
