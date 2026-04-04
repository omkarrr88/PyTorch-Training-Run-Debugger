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
from ml_training_debugger.nn_models import SimpleCNN, SimpleMLP, create_model
from ml_training_debugger.scenarios import ScenarioParams

# Re-export for backwards compatibility (tests import from here)
__all__ = ["SimpleCNN", "SimpleMLP", "create_model"]

_create_model = create_model


# Cache for real training curves — keyed by (task_id, seed, model_type)
_TRAINING_CACHE: dict[tuple[str, int, str], dict[str, list[float]]] = {}

TRAINING_EPOCHS = 20
TRAINING_BATCH_SIZE = 16


def run_real_training(scenario: ScenarioParams) -> dict[str, list[float]]:
    """Run real 20-epoch mini-training and return loss/accuracy curves.

    Caches results per (task_id, seed, model_type) for instant subsequent resets.
    Each call takes ~0.5-2s on CPU; cached calls are instant.
    """
    cache_key = (scenario.task_id, scenario.seed, scenario.model_type)
    if cache_key in _TRAINING_CACHE:
        return _TRAINING_CACHE[cache_key]

    torch.manual_seed(scenario.seed)
    model = _create_model(scenario.model_type)
    criterion = nn.CrossEntropyLoss()
    root = scenario.root_cause.value

    # Configure optimizer based on fault type
    if root == "lr_too_high":
        lr = scenario.learning_rate
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        model.train()
    elif root == "vanishing_gradients":
        optimizer = torch.optim.SGD(model.parameters(), lr=scenario.learning_rate)
        model.train()
    elif root == "batchnorm_eval_mode":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.eval()  # The bug
    elif root == "scheduler_misconfigured":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scenario.scheduler_step_size,
            gamma=scenario.scheduler_gamma,
        )
        model.train()
    elif root == "overfitting":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, weight_decay=scenario.weight_decay
        )
        model.train()
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()

    loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_acc_history: list[float] = []

    # Generate fixed training and validation data
    torch.manual_seed(scenario.seed + 100)
    train_x = torch.randn(TRAINING_BATCH_SIZE * 4, 3, 32, 32)
    train_y = torch.randint(0, 10, (TRAINING_BATCH_SIZE * 4,))
    val_x = torch.randn(TRAINING_BATCH_SIZE, 3, 32, 32)
    val_y = torch.randint(0, 10, (TRAINING_BATCH_SIZE,))

    # For data leakage: copy some training samples into validation
    if root == "data_leakage":
        leak_count = max(1, int(TRAINING_BATCH_SIZE * scenario.leakage_pct))
        val_x[:leak_count] = train_x[:leak_count]
        val_y[:leak_count] = train_y[:leak_count]

    for epoch in range(TRAINING_EPOCHS):
        # Training step
        batch_idx = (epoch % 4) * TRAINING_BATCH_SIZE
        bx = train_x[batch_idx : batch_idx + TRAINING_BATCH_SIZE]
        by = train_y[batch_idx : batch_idx + TRAINING_BATCH_SIZE]

        optimizer.zero_grad()
        output = model(bx)
        loss = criterion(output, by)

        loss_val = loss.item()
        if loss_val != loss_val:  # NaN check
            loss_history.append(float("inf"))
        else:
            loss_history.append(loss_val)

        try:
            loss.backward()
            optimizer.step()
            if root == "scheduler_misconfigured":
                scheduler.step()
        except RuntimeError:
            loss_history[-1] = float("inf")

        # Validation step (no grad)
        with torch.no_grad():
            val_out = model(val_x)
            v_loss = criterion(val_out, val_y)
            v_loss_val = v_loss.item()
            val_loss_history.append(v_loss_val if v_loss_val == v_loss_val else float("inf"))
            preds = val_out.argmax(dim=1)
            acc = (preds == val_y).float().mean().item()
            val_acc_history.append(acc)

    result = {
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "val_acc_history": val_acc_history,
    }
    _TRAINING_CACHE[cache_key] = result
    return result


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
