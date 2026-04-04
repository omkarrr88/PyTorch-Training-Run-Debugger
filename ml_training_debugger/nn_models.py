"""PyTorch model definitions for the training debugger.

SimpleCNN (~50K params) and SimpleMLP (~20K params).
Spec reference: Section 9 — PyTorch Model Pool.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """3-layer CNN for CIFAR-10 style classification. ~50K params."""

    def __init__(self, num_layers: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 4 * 4, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleMLP(nn.Module):
    """3-layer MLP for CIFAR-10 style classification. ~20K params."""

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim: int = 128,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


def create_model(model_type: str) -> nn.Module:
    """Create a model by type string."""
    if model_type == "mlp":
        return SimpleMLP()
    return SimpleCNN()
