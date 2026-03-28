"""PyTorch-native fault injection engine.

Real torch.nn.Module models, real torch.autograd gradients,
real state_dict() weight snapshots. Zero numpy.
Spec reference: Sections 6, 9.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ml_training_debugger.models import GradientStats, ModelWeightStats
from ml_training_debugger.scenarios import ScenarioParams


class SimpleCNN(nn.Module):
    """3-layer CNN for CIFAR-10 style classification. ~50K params.

    Spec Section 9 — PyTorch Model Pool.
    """

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
    """3-layer MLP for CIFAR-10 style classification. ~20K params.

    Second architecture — randomly selected alongside SimpleCNN at reset().
    """

    def __init__(self, input_dim: int = 3072, hidden_dim: int = 128, num_classes: int = 10) -> None:
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


def _create_model(model_type: str) -> nn.Module:
    """Create a model by type string."""
    if model_type == "mlp":
        return SimpleMLP()
    return SimpleCNN()


def create_model_and_inject_fault(
    scenario: ScenarioParams,
) -> tuple[nn.Module, dict]:
    """Instantiate a real PyTorch model and inject the specified fault.

    Returns:
        (model, info_dict) where info_dict contains computed artifacts.
    """
    torch.manual_seed(scenario.seed)

    model = _create_model(scenario.model_type)
    criterion = nn.CrossEntropyLoss()
    info: dict = {}

    # Generate random batch (CIFAR-10 style: 3x32x32)
    batch_x = torch.randn(8, 3, 32, 32)
    batch_y = torch.randint(0, 10, (8,))

    if scenario.root_cause.value == "lr_too_high":
        # Exploding gradients: high LR with SGD → gradients explode on all layers
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=scenario.learning_rate * 10.0
        )
        for _ in range(3):
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        # Run one final backward to capture extreme gradients
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()

    elif scenario.root_cause.value == "vanishing_gradients":
        # Simulate vanishing gradients: run forward/backward then scale grads
        # to simulate gradient decay through deep layers
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=scenario.learning_rate)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        # Scale gradients to simulate vanishing: deeper layers get smaller grads
        depth_mult = scenario.depth_multiplier
        layer_idx = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                decay = torch.tensor(1e-7) * torch.exp(
                    torch.tensor(-depth_mult * layer_idx)
                )
                param.grad.data = param.grad.data * decay
                layer_idx += 1

    elif scenario.root_cause.value == "data_leakage":
        # Normal model — no gradient anomaly
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

    elif scenario.root_cause.value == "overfitting":
        # Normal model with zero weight decay
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=scenario.weight_decay,
        )
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

    elif scenario.root_cause.value == "batchnorm_eval_mode":
        # model.eval() before training — the real bug
        model.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Still run forward/backward to get gradient data
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

    elif scenario.root_cause.value == "code_bug":
        # Normal training with the model bug injected in code only
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

    elif scenario.root_cause.value == "scheduler_misconfigured":
        # Normal model, but with an aggressively decaying LR scheduler
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scenario.scheduler_step_size,
            gamma=scenario.scheduler_gamma,
        )
        for _ in range(3):
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        info["final_lr"] = optimizer.param_groups[0]["lr"]

    return model, info


def extract_gradient_stats(
    model: nn.Module,
    scenario: Optional[ScenarioParams] = None,
) -> list[GradientStats]:
    """Extract gradient statistics from real param.grad tensors.

    For Task 5 (batchnorm_eval_mode), injects red-herring spike on
    the configured layer.
    """
    stats: list[GradientStats] = []

    if isinstance(model, SimpleMLP):
        named_layers = [
            ("fc1", model.fc1),
            ("fc2", model.fc2),
            ("fc3", model.fc3),
        ]
    else:
        named_layers = [
            ("conv1", model.conv1),
            ("conv2", model.conv2),
            ("conv3", model.conv3),
            ("fc", model.fc),
        ]

    for layer_name, layer in named_layers:
        norms: list[float] = []
        for param in layer.parameters():
            if param.grad is not None:
                norm_val = torch.norm(param.grad).item()
                norms.append(norm_val)

        if not norms:
            norms = [0.0]

        mean_norm = sum(norms) / len(norms)
        max_norm = max(norms)

        # Build norm_history (simulated last 5 values, based on current)
        norm_history = [mean_norm * (0.9 + 0.2 * i / 4) for i in range(5)]

        # Task 5 red herring: spike on configured layer
        if scenario and scenario.root_cause.value == "batchnorm_eval_mode":
            if layer_name == scenario.red_herring_spike_layer:
                spike = scenario.red_herring_intensity
                norm_history = [
                    mean_norm,
                    mean_norm,
                    mean_norm * spike,
                    mean_norm * spike * 1.2,
                    mean_norm,
                ]
                mean_norm = sum(norm_history) / len(norm_history)
                max_norm = max(norm_history)

            # Conv1 near-vanishing red herring
            if layer_name == "conv1" and scenario.red_herring_spike_layer != "conv1":
                near_vanish = 0.0003
                norm_history = [near_vanish * (0.95 + 0.1 * i / 4) for i in range(5)]
                mean_norm = near_vanish
                max_norm = max(norm_history)

        is_exploding = mean_norm > 10.0
        is_vanishing = mean_norm < 1e-6

        stats.append(
            GradientStats(
                layer_name=layer_name,
                norm_history=norm_history,
                mean_norm=mean_norm,
                max_norm=max_norm,
                is_exploding=is_exploding,
                is_vanishing=is_vanishing,
            )
        )

    return stats


def extract_weight_stats(model: nn.Module) -> list[ModelWeightStats]:
    """Extract weight statistics from real model.state_dict()."""
    stats: list[ModelWeightStats] = []
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        stats.append(
            ModelWeightStats(
                layer_name=name,
                weight_norm=torch.norm(param).item(),
                weight_mean=param.mean().item(),
                weight_std=param.std().item(),
                weight_min=param.min().item(),
                weight_max=param.max().item(),
                dead_neuron_pct=0.0,
                has_nan=bool(torch.isnan(param).any().item()),
                has_inf=bool(torch.isinf(param).any().item()),
            )
        )
    return stats


def extract_model_modes(model: nn.Module) -> dict[str, str]:
    """Extract training/eval mode for each named module."""
    modes: dict[str, str] = {}
    for name, module in model.named_modules():
        if name == "":
            continue
        modes[name] = "train" if module.training else "eval"
    return modes
