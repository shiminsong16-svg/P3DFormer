import torch
import torch.nn as nn
from typing import Sequence

class BoundaryHead(nn.Module):
    """Predict an instance-boundary logit for each point feature."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 128,
        dropout: Sequence[float] = (0.3, 0.3),
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout[0]),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout[1])
        )

        self.boundary_head = nn.Linear(hidden_dim, out_channels)

    def forward(self, point_feats: torch.Tensor):
        """Return unnormalized logits with shape ``(num_points, out_channels)``."""
        x = self.mlp(point_feats)
        return self.boundary_head(x)
