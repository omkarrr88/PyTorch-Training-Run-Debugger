"""All Pydantic models, enums, and typed data structures.

No business logic. Pure data definitions.
Spec reference: Section 10 — Data Models.
"""

from __future__ import annotations

import enum
from typing import Optional, Union

import torch  # noqa: F401 — PyTorch-native project, required import
from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class RootCauseDiagnosis(str, enum.Enum):
    """Closed enumeration of ML failure root causes. Spec Section 10."""

    LR_TOO_HIGH = "lr_too_high"
    VANISHING_GRADIENTS = "vanishing_gradients"
    DATA_LEAKAGE = "data_leakage"
    OVERFITTING = "overfitting"
    BATCHNORM_EVAL_MODE = "batchnorm_eval_mode"
    CODE_BUG = "code_bug"


VALID_DIAGNOSES: set[str] = {d.value for d in RootCauseDiagnosis}


class TrainingConfig(BaseModel):
    """Typed hyperparameter configuration. Spec Section 10."""

    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 64
    hidden_dim: int = 64
    num_layers: int = 3
    optimizer: str = "adam"
    dropout_rate: float = 0.0
    gradient_clip_norm: Optional[float] = None


VALID_CONFIG_KEYS: set[str] = set(TrainingConfig.model_fields.keys())


class GradientStats(BaseModel):
    """Per-layer gradient information from real torch.autograd. Spec Section 10."""

    layer_name: str
    norm_history: list[float]
    mean_norm: float
    max_norm: float
    is_exploding: bool  # True when mean_norm > 10.0
    is_vanishing: bool  # True when mean_norm < 1e-6


class ModelWeightStats(BaseModel):
    """Per-layer weight statistics from real state_dict(). Spec Section 10."""

    layer_name: str
    weight_norm: float
    weight_mean: float
    weight_std: float
    weight_min: float
    weight_max: float
    dead_neuron_pct: float = 0.0
    has_nan: bool = False
    has_inf: bool = False


class DataBatchStats(BaseModel):
    """Data batch inspection results. Spec Section 10."""

    label_distribution: dict[int, float]
    feature_mean: float
    feature_std: float
    null_count: int = 0
    class_overlap_score: float
    batch_size: int
    duplicate_ratio: float = 0.0


class CodeSnippet(BaseModel):
    """PyTorch code for Task 6 inspection. Spec Section 10."""

    code: str
    filename: str = "train.py"
    line_count: int
    imports: list[str]
    hint: Optional[str] = None


class EpisodeState(BaseModel):
    """Tracks agent history within an episode. Spec Section 10."""

    step_count: int = 0
    gradients_inspected: bool = False
    gradients_were_normal: bool = False
    data_inspected: bool = False
    model_modes_inspected: bool = False
    model_weights_inspected: bool = False
    code_inspected: bool = False
    fix_action_taken: bool = False
    restart_after_fix: bool = False
    diagnosis_submitted: bool = False
    actions_taken: list[str] = Field(default_factory=list)

    def compute_available_actions(self) -> list[str]:
        """Dynamically compute available actions based on current state.

        Rules from spec Section 10 — Dynamic available_actions:
        - restart_run: only after fix_action_taken
        - rollback_checkpoint: only after restart_after_fix
        - fix_code: only after code_inspected
        - mark_diagnosed: disappears after diagnosis_submitted
        """
        actions: list[str] = [
            "inspect_gradients",
            "inspect_data_batch",
            "inspect_model_modes",
            "inspect_model_weights",
            "inspect_code",
            "modify_config",
            "add_callback",
            "replace_optimizer",
            "patch_data_loader",
            "fix_model_mode",
        ]
        if self.code_inspected:
            actions.append("fix_code")
        if self.fix_action_taken:
            actions.append("restart_run")
        if self.restart_after_fix:
            actions.append("rollback_checkpoint")
        if not self.diagnosis_submitted:
            actions.append("mark_diagnosed")
        return actions


ALL_ACTION_TYPES: set[str] = {
    "inspect_gradients",
    "inspect_data_batch",
    "inspect_model_modes",
    "inspect_model_weights",
    "inspect_code",
    "modify_config",
    "add_callback",
    "replace_optimizer",
    "patch_data_loader",
    "fix_model_mode",
    "fix_code",
    "restart_run",
    "mark_diagnosed",
    "rollback_checkpoint",
}


class MLTrainingAction(Action):
    """What the agent can do — extends openenv Action. Spec Section 10."""

    action_type: str
    target: Optional[str] = None
    value: Optional[Union[float, int, str]] = None
    diagnosis: Optional[str] = None
    line: Optional[int] = None
    replacement: Optional[str] = None


class MLTrainingObservation(Observation):
    """Full observation — extends openenv Observation.

    Observation base has built-in: done (bool), reward (float|None), metadata (dict).
    Spec Section 10.
    """

    run_id: str = ""
    framework: str = "pytorch"
    epoch: int = 20
    training_loss_history: list[float] = Field(default_factory=list)
    val_loss_history: list[float] = Field(default_factory=list)
    val_accuracy_history: list[float] = Field(default_factory=list)
    gradient_stats: list[GradientStats] = Field(default_factory=list)
    model_weight_stats: Optional[list[ModelWeightStats]] = None
    gpu_memory_used_gb: float = 6.2
    gpu_memory_total_gb: float = 16.0
    learning_rate: float = 0.001
    current_config: TrainingConfig = Field(default_factory=TrainingConfig)
    error_log: Optional[str] = None
    data_batch_stats: Optional[DataBatchStats] = None
    model_mode_info: Optional[dict[str, str]] = None
    code_snippet: Optional[CodeSnippet] = None
    available_actions: list[str] = Field(default_factory=list)
    episode_state: EpisodeState = Field(default_factory=EpisodeState)
    notes: Optional[str] = None
