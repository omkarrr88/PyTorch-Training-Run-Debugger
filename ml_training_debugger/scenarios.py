"""ScenarioParams and scenario sampling.

Internal scenario configuration — not exposed to the agent.
Spec reference: Sections 6, 10, 11.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from ml_training_debugger.models import RootCauseDiagnosis


@dataclasses.dataclass(frozen=True)
class ScenarioParams:
    """Internal scenario parameters created at reset() time."""

    task_id: str
    root_cause: RootCauseDiagnosis
    seed: int
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    leakage_pct: float = 0.0
    depth_multiplier: float = 1.0
    divergence_epoch: int = 5
    red_herring_intensity: float = 1.0
    red_herring_spike_layer: str = "fc"
    bug_type: Optional[str] = None
    notes: Optional[str] = None
    error_log: Optional[str] = None
    gpu_memory_used_gb: float = 6.2
    max_steps: int = 20
    model_type: str = "cnn"
    difficulty_level: int = 3
    scheduler_gamma: float = 0.1
    scheduler_step_size: int = 10


def _task_seed(task_id: str, seed: int) -> int:
    """Derive a deterministic seed from task_id and provided seed."""
    task_num = int(task_id.split("_")[1])
    return seed * 1000 + task_num


def _choose(options: list, rng: torch.Generator) -> object:
    """Choose a random element from a list using torch RNG."""
    idx = int(torch.randint(0, len(options), (1,), generator=rng).item())
    return options[idx]


def _pick_model_type(rng: torch.Generator) -> str:
    """Randomly pick CNN or MLP architecture."""
    return str(_choose(["cnn", "mlp"], rng))


def sample_scenario(
    task_id: str, seed: int = 42, difficulty_level: int = 3
) -> ScenarioParams:
    """Sample a ScenarioParams for the given task.

    Args:
        task_id: One of task_001 through task_007.
        seed: Base seed for reproducibility.
        difficulty_level: 1 (easy signals) to 5 (max ambiguity). Default 3.

    Returns:
        ScenarioParams with randomized fault parameters.

    Raises:
        ValueError: If task_id is unknown.
    """
    effective_seed = _task_seed(task_id, seed)
    rng = torch.Generator()
    rng.manual_seed(effective_seed)

    if task_id == "task_001":
        lr = _choose([0.05, 0.08, 0.10, 0.15, 0.30], rng)
        return ScenarioParams(
            task_id=task_id,
            root_cause=RootCauseDiagnosis.LR_TOO_HIGH,
            seed=effective_seed,
            learning_rate=float(lr),
            error_log=f"RuntimeError: Loss is NaN at epoch 12 (lr={lr})",
            max_steps=20,
            model_type=_pick_model_type(rng),
            difficulty_level=difficulty_level,
        )

    if task_id == "task_002":
        lr = _choose([1e-6, 5e-6, 1e-5], rng)
        depth_mult = _choose([1.0, 1.5, 2.0], rng)
        return ScenarioParams(
            task_id=task_id,
            root_cause=RootCauseDiagnosis.VANISHING_GRADIENTS,
            seed=effective_seed,
            learning_rate=float(lr),
            depth_multiplier=float(depth_mult),
            notes=(
                "Training resumed from a checkpoint saved at epoch 0 — "
                "early learning rate warmup may still be in effect."
            ),
            max_steps=20,
            model_type=_pick_model_type(rng),
            difficulty_level=difficulty_level,
        )

    if task_id == "task_003":
        leakage = _choose([0.12, 0.18, 0.22, 0.28], rng)
        return ScenarioParams(
            task_id=task_id,
            root_cause=RootCauseDiagnosis.DATA_LEAKAGE,
            seed=effective_seed,
            leakage_pct=float(leakage),
            notes=(
                "Model architecture upgraded from 2-layer to 4-layer CNN "
                "at epoch 2. Performance improvement may reflect increased "
                "model capacity."
            ),
            max_steps=25,
            model_type=_pick_model_type(rng),
            difficulty_level=difficulty_level,
        )

    if task_id == "task_004":
        wd = _choose([0.0, 0.0001, 0.001], rng)
        div_epoch = _choose([5, 8, 12], rng)
        return ScenarioParams(
            task_id=task_id,
            root_cause=RootCauseDiagnosis.OVERFITTING,
            seed=effective_seed,
            weight_decay=float(wd),
            divergence_epoch=int(div_epoch),
            notes=(
                "Dataset augmentation was disabled for this run to speed "
                "up training. Re-enabling may improve generalization."
            ),
            max_steps=25,
            model_type=_pick_model_type(rng),
            difficulty_level=difficulty_level,
        )

    if task_id == "task_005":
        intensity = torch.empty(1).uniform_(0.8, 2.5, generator=rng).item()
        spike_layer = _choose(["fc", "conv1"], rng)
        return ScenarioParams(
            task_id=task_id,
            root_cause=RootCauseDiagnosis.BATCHNORM_EVAL_MODE,
            seed=effective_seed,
            red_herring_intensity=float(intensity),
            red_herring_spike_layer=str(spike_layer),
            gpu_memory_used_gb=14.56,
            error_log=(
                "Warning: GPU memory pressure detected, consider reducing "
                "batch size or enabling gradient checkpointing"
            ),
            max_steps=30,
            model_type="cnn",  # CNN always for BatchNorm eval — MLP BatchNorm1d behaves differently
            difficulty_level=difficulty_level,
        )

    if task_id == "task_006":
        bug = _choose(
            ["eval_mode", "detach_loss", "zero_grad_missing", "inplace_relu"], rng
        )
        return ScenarioParams(
            task_id=task_id,
            root_cause=RootCauseDiagnosis.CODE_BUG,
            seed=effective_seed,
            bug_type=str(bug),
            notes="Try adjusting the learning rate schedule.",
            max_steps=30,
            model_type="cnn",  # Code templates reference CNN training — keep CNN for consistency
            difficulty_level=difficulty_level,
        )

    if task_id == "task_007":
        gamma = _choose([0.01, 0.001, 0.0001], rng)
        step_size = _choose([2, 3, 5], rng)
        return ScenarioParams(
            task_id=task_id,
            root_cause=RootCauseDiagnosis.SCHEDULER_MISCONFIGURED,
            seed=effective_seed,
            scheduler_gamma=float(gamma),
            scheduler_step_size=int(step_size),
            notes="LR scheduler was recently added to improve convergence.",
            max_steps=25,
            model_type=_pick_model_type(rng),
            difficulty_level=difficulty_level,
        )

    raise ValueError(f"Unknown task_id: {task_id}")
