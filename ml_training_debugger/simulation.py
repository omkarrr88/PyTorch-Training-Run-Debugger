"""Parametric curve generation using torch.Tensor operations.

All loss/accuracy histories are generated via parametric equations.
Zero numpy. Spec reference: Section 6.
"""

from __future__ import annotations

import torch

from ml_training_debugger.scenarios import ScenarioParams

EPOCHS = 20


def gen_loss_history(scenario: ScenarioParams) -> list[float]:
    """Generate training loss history (20 epochs) using torch ops."""
    torch.manual_seed(scenario.seed)
    t = torch.arange(EPOCHS, dtype=torch.float32)

    root = scenario.root_cause.value

    if root == "lr_too_high":
        # Exponentially growing loss
        lr_tensor = torch.tensor(scenario.learning_rate, dtype=torch.float32)
        base = torch.exp(lr_tensor * t * 0.5)
        loss = 2.3 * base
        # Add NaN marker after epoch 12
        loss_list = loss.tolist()
        for i in range(12, EPOCHS):
            loss_list[i] = float("inf")
        return loss_list

    if root == "vanishing_gradients":
        # Flat loss — barely decreases
        noise = torch.randn(EPOCHS) * 0.02
        loss = 2.3 - t * 0.002 + noise
        return loss.clamp(min=0.01).tolist()

    if root == "data_leakage":
        # Normal-looking training loss
        loss = 2.3 * torch.exp(-0.15 * t) + 0.05
        noise = torch.randn(EPOCHS) * 0.02
        return (loss + noise).clamp(min=0.01).tolist()

    if root == "overfitting":
        # Steadily decreasing to near-zero
        loss = 2.3 * torch.exp(-0.25 * t) + 0.01
        noise = torch.randn(EPOCHS) * 0.01
        return (loss + noise).clamp(min=0.001).tolist()

    if root == "batchnorm_eval_mode":
        # Roughly normal with higher variance
        base = 2.3 * torch.exp(-0.1 * t) + 0.3
        noise = torch.randn(EPOCHS) * 0.15
        return (base + noise).clamp(min=0.1).tolist()

    if root == "code_bug":
        loss = 2.3 * torch.exp(-0.05 * t) + 0.5
        noise = torch.randn(EPOCHS) * 0.1
        return (loss + noise).clamp(min=0.1).tolist()

    if root == "scheduler_misconfigured":
        # Training starts well, then LR drops too aggressively causing stagnation
        step_size = scenario.scheduler_step_size
        gamma = scenario.scheduler_gamma
        loss_list: list[float] = []
        for i in range(EPOCHS):
            if i < step_size:
                val = 2.3 * (1.0 - 0.15 * i)  # normal decrease
            else:
                steps_decayed = (i - step_size) // step_size + 1
                effective_lr_ratio = gamma ** steps_decayed
                val = 2.3 * (1.0 - 0.15 * step_size) + 0.05 * (i - step_size) * (1 - effective_lr_ratio)
            loss_list.append(max(0.3, val + torch.randn(1).item() * 0.05))
        return loss_list

    # Fallback
    return (2.3 * torch.exp(-0.1 * t)).tolist()


def gen_val_accuracy_history(scenario: ScenarioParams) -> list[float]:
    """Generate validation accuracy history (20 epochs) using torch ops."""
    torch.manual_seed(scenario.seed + 1)
    t = torch.arange(EPOCHS, dtype=torch.float32)

    root = scenario.root_cause.value

    if root == "lr_too_high":
        # Collapses along with training loss
        acc = torch.sigmoid(torch.linspace(0, -3, EPOCHS)) * 0.5
        return acc.clamp(0.0, 1.0).tolist()

    if root == "vanishing_gradients":
        # Near random chance
        noise = torch.randn(EPOCHS) * 0.02
        acc = 0.10 + t * 0.001 + noise
        return acc.clamp(0.0, 1.0).tolist()

    if root == "data_leakage":
        # Suspiciously high from epoch 1
        leakage = torch.tensor(scenario.leakage_pct, dtype=torch.float32)
        base = torch.sigmoid(torch.linspace(-3, 3, EPOCHS))
        acc = base * (1.0 - leakage) + leakage * 0.95
        # Inflate early epochs
        acc = acc.clamp(0.0, 1.0)
        # Ensure suspiciously high from epoch 1
        acc_list = acc.tolist()
        for i in range(EPOCHS):
            acc_list[i] = max(acc_list[i], 0.82 * (1.0 + scenario.leakage_pct))
        return [min(v, 0.99) for v in acc_list]

    if root == "overfitting":
        # Rises then falls — classic divergence
        div = scenario.divergence_epoch
        acc_list: list[float] = []
        for i in range(EPOCHS):
            if i < div:
                val = 0.10 + (0.75 - 0.10) * (i / max(div, 1))
            else:
                decline = (i - div) * 0.02
                val = 0.75 - decline
            acc_list.append(max(0.0, min(1.0, val)))
        return acc_list

    if root == "batchnorm_eval_mode":
        # Slow degradation ~1-2% per epoch
        start = 0.76
        noise = torch.randn(EPOCHS) * 0.01
        acc = torch.tensor(
            [start - 0.015 * i for i in range(EPOCHS)], dtype=torch.float32
        )
        acc = acc + noise
        return acc.clamp(0.0, 1.0).tolist()

    if root == "code_bug":
        noise = torch.randn(EPOCHS) * 0.03
        acc = 0.10 + t * 0.005 + noise
        return acc.clamp(0.0, 1.0).tolist()

    if root == "scheduler_misconfigured":
        # Accuracy improves initially, then stagnates/degrades when scheduler kills LR
        step_size = scenario.scheduler_step_size
        acc_list: list[float] = []
        for i in range(EPOCHS):
            if i < step_size:
                val = 0.10 + 0.08 * i
            else:
                val = 0.10 + 0.08 * step_size - 0.01 * (i - step_size)
            acc_list.append(max(0.05, min(0.95, val + torch.randn(1).item() * 0.02)))
        return acc_list

    # Fallback
    return (torch.sigmoid(torch.linspace(-3, 3, EPOCHS)) * 0.9).tolist()


def gen_val_loss_history(scenario: ScenarioParams) -> list[float]:
    """Generate validation loss history (20 epochs) using torch ops."""
    torch.manual_seed(scenario.seed + 2)
    t = torch.arange(EPOCHS, dtype=torch.float32)

    root = scenario.root_cause.value

    if root == "lr_too_high":
        # Mirrors training loss divergence
        lr_tensor = torch.tensor(scenario.learning_rate, dtype=torch.float32)
        loss = 2.3 * torch.exp(lr_tensor * t * 0.5)
        loss_list = loss.tolist()
        for i in range(12, EPOCHS):
            loss_list[i] = float("inf")
        return loss_list

    if root == "vanishing_gradients":
        noise = torch.randn(EPOCHS) * 0.02
        loss = 2.3 - t * 0.001 + noise
        return loss.clamp(min=0.01).tolist()

    if root == "data_leakage":
        # Low val loss (because leaking train data into val)
        base = 2.3 * torch.exp(-0.2 * t) + 0.03
        noise = torch.randn(EPOCHS) * 0.02
        return (base + noise).clamp(min=0.01).tolist()

    if root == "overfitting":
        # Initially decreases, then diverges upward
        div = scenario.divergence_epoch
        loss_list: list[float] = []
        for i in range(EPOCHS):
            if i < div:
                val = 2.3 * (1.0 - 0.8 * i / max(div, 1))
            else:
                val = 0.46 + 0.1 * (i - div)
            loss_list.append(max(0.01, val))
        return loss_list

    if root == "batchnorm_eval_mode":
        # Slightly increasing
        base = 1.5 + t * 0.03
        noise = torch.randn(EPOCHS) * 0.1
        return (base + noise).clamp(min=0.1).tolist()

    if root == "code_bug":
        loss = 2.3 * torch.exp(-0.03 * t) + 0.8
        noise = torch.randn(EPOCHS) * 0.1
        return (loss + noise).clamp(min=0.1).tolist()

    if root == "scheduler_misconfigured":
        step_size = scenario.scheduler_step_size
        loss_list: list[float] = []
        for i in range(EPOCHS):
            if i < step_size:
                val = 2.3 * (1.0 - 0.12 * i)
            else:
                val = 2.3 * (1.0 - 0.12 * step_size) + 0.03 * (i - step_size)
            loss_list.append(max(0.1, val + torch.randn(1).item() * 0.05))
        return loss_list

    # Fallback
    return (2.3 * torch.exp(-0.1 * t) + 0.1).tolist()


def _gen_confusion_matrix(scenario: ScenarioParams) -> list[list[float]]:
    """Generate a 10x10 confusion matrix based on the fault type."""
    torch.manual_seed(scenario.seed + 10)
    root = scenario.root_cause.value
    n = 10

    if root == "data_leakage":
        # High diagonal but with leakage-induced off-diagonal noise
        base = torch.eye(n) * 0.8
        noise = torch.rand(n, n) * scenario.leakage_pct * 0.3
        cm = base + noise
    elif root == "overfitting":
        # Near-perfect diagonal (memorized)
        cm = torch.eye(n) * 0.95 + torch.rand(n, n) * 0.02
    else:
        # Normal confusion with moderate accuracy
        cm = torch.eye(n) * 0.6 + torch.rand(n, n) * 0.08

    # Normalize rows to sum to ~1.0
    row_sums = cm.sum(dim=1, keepdim=True)
    cm = cm / row_sums
    return cm.tolist()


def gen_data_batch_stats(scenario: ScenarioParams) -> dict:
    """Generate data batch statistics for the scenario."""
    torch.manual_seed(scenario.seed + 3)

    root = scenario.root_cause.value

    cm = _gen_confusion_matrix(scenario)

    if root == "data_leakage":
        overlap = 0.5 + scenario.leakage_pct * 1.5
        overlap = min(overlap, 0.92)
        return {
            "label_distribution": {i: 0.1 for i in range(10)},
            "feature_mean": 0.45 + torch.randn(1).item() * 0.05,
            "feature_std": 0.22 + torch.randn(1).item() * 0.02,
            "null_count": 0,
            "class_overlap_score": overlap,
            "batch_size": 64,
            "duplicate_ratio": scenario.leakage_pct,
            "confusion_matrix": cm,
        }

    if root == "overfitting":
        return {
            "label_distribution": {i: 0.1 for i in range(10)},
            "feature_mean": 0.48 + torch.randn(1).item() * 0.03,
            "feature_std": 0.25 + torch.randn(1).item() * 0.02,
            "null_count": 0,
            "class_overlap_score": 0.0,
            "batch_size": 64,
            "duplicate_ratio": 0.0,
            "confusion_matrix": cm,
        }

    # Default: normal data
    return {
        "label_distribution": {i: 0.1 for i in range(10)},
        "feature_mean": 0.47 + torch.randn(1).item() * 0.03,
        "feature_std": 0.24 + torch.randn(1).item() * 0.02,
        "null_count": 0,
        "class_overlap_score": 0.0 + torch.randn(1).abs().item() * 0.05,
        "batch_size": 64,
        "duplicate_ratio": 0.0,
        "confusion_matrix": cm,
    }
