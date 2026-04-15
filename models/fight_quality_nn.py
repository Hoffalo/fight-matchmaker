"""
models/fight_quality_nn.py
PyTorch Neural Network for predicting Fight Quality Score.

Architecture: Deep MLP with residual connections, batch norm, and dropout.
Input:  48-dim matchup feature vector (24 per fighter)
Output: 1 scalar — predicted Fight Quality Score (0–100)
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and dropout."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class FightQualityNN(nn.Module):
    """
    Deep MLP for fight quality prediction.

    Input:   48-dimensional matchup feature vector
    Output:  1 scalar (0-100 quality score)

    Architecture:
      - Input projection to hidden_dim
      - N residual blocks
      - Projection head → scalar output
    """

    def __init__(
        self,
        input_dim: int = 48,
        hidden_layers: list[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128, 64, 32]

        layers = []
        prev_dim = input_dim

        # Build encoder
        for i, dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.GELU())
            if i < len(hidden_layers) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Output in [0, 1], scaled to [0, 100] post-hoc
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 48) float32 tensor
        Returns:
            (batch, 1) float32 tensor in [0, 1] (multiply by 100 for score)
        """
        h = self.encoder(x)
        return self.head(h)

    def predict_score(self, x: np.ndarray) -> float:
        """Convenience: predict a single matchup vector, return 0-100 score."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            raw = self.forward(t)
            return float(raw.cpu().item() * 100.0)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict a batch, returns array of 0-100 scores."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(device)
            raw = self.forward(t)
            return (raw.squeeze(-1).cpu().numpy() * 100.0)
