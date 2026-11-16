"""Two-layer fully-connected classifier with Dropout."""
from __future__ import annotations

import torch.nn as nn


def build_model(input_dim: int, hidden_dim: int, dropout_p: float, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_p),
        nn.Linear(hidden_dim, num_classes),
    )
