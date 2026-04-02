"""Per-task grader functions — returns normalized 0.0-1.0 score at episode end.

Separate from reward_engine.py. Evaluates EpisodeState holistically.
NOT a sum of step rewards. Spec reference: Section 11 grader breakdowns.
"""

from __future__ import annotations

import torch  # noqa: F401 — PyTorch-native project

from ml_training_debugger.models import EpisodeState
from ml_training_debugger.scenarios import ScenarioParams

FIX_ACTIONS = frozenset(
    {
        "modify_config",
        "add_callback",
        "replace_optimizer",
        "patch_data_loader",
        "fix_model_mode",
        "fix_code",
    }
)


def _has_action(state: EpisodeState, action_type: str) -> bool:
    return action_type in state.actions_taken


def _correct_diagnosis(state: EpisodeState, scenario: ScenarioParams) -> bool:
    if not state.diagnosis_submitted:
        return False
    # Find the diagnosis from actions_taken metadata
    # We store "mark_diagnosed:<diagnosis>" in actions_taken
    for action_str in reversed(state.actions_taken):
        if action_str.startswith("mark_diagnosed:"):
            submitted = action_str.split(":", 1)[1]
            return submitted == scenario.root_cause.value
    return False


def _submitted_diagnosis(state: EpisodeState) -> str | None:
    for action_str in reversed(state.actions_taken):
        if action_str.startswith("mark_diagnosed:"):
            return action_str.split(":", 1)[1]
    return None


def grade_task_001(state: EpisodeState, scenario: ScenarioParams) -> float:
    """Grade Task 1 — Exploding Gradients (easy). Spec Section 11."""
    score = 0.0

    # +0.05 for inspect_gradients
    if state.gradients_inspected:
        score += 0.05

    # +0.20 for correct fix (modify_config with LR reduction)
    if _has_action(state, "modify_config"):
        score += 0.20

    # +0.35 for restart with convergence
    if state.restart_after_fix:
        score += 0.35

    # +0.40 for correct diagnosis
    if _correct_diagnosis(state, scenario):
        score += 0.40

    return min(1.0, max(0.0, score))


def grade_task_002(state: EpisodeState, scenario: ScenarioParams) -> float:
    """Grade Task 2 — Vanishing Gradients (easy). Spec Section 11."""
    score = 0.0

    if state.gradients_inspected:
        score += 0.05
    if _has_action(state, "modify_config"):
        score += 0.20
    if state.restart_after_fix:
        score += 0.35
    if _correct_diagnosis(state, scenario):
        score += 0.40

    return min(1.0, max(0.0, score))


def grade_task_003(state: EpisodeState, scenario: ScenarioParams) -> float:
    """Grade Task 3 — Silent Data Leakage (medium). Spec Section 11."""
    score = 0.0

    # +0.05 for inspect_data_batch
    if state.data_inspected:
        score += 0.05

    # +0.30 for patch_data_loader
    if _has_action(state, "patch_data_loader"):
        score += 0.30

    # +0.30 for restart with convergence (val accuracy normalizes)
    if state.restart_after_fix:
        score += 0.30

    # +0.35 for correct diagnosis
    if _correct_diagnosis(state, scenario):
        score += 0.35

    return min(1.0, max(0.0, score))


def grade_task_004(state: EpisodeState, scenario: ScenarioParams) -> float:
    """Grade Task 4 — Overfitting (medium). Spec Section 11."""
    score = 0.0

    if state.data_inspected:
        score += 0.05
    if _has_action(state, "modify_config") or _has_action(state, "add_callback"):
        score += 0.25
    if state.restart_after_fix:
        score += 0.30
    if _correct_diagnosis(state, scenario):
        score += 0.40

    return min(1.0, max(0.0, score))


def grade_task_005(state: EpisodeState, scenario: ScenarioParams) -> float:
    """Grade Task 5 — BatchNorm Eval Mode (hard). Spec Section 11.

    Hard task requires thorough investigation for full credit.
    Context-gated penalty: -0.20 if add_callback after gradients_were_normal.
    Wrong-fix penalty: -0.10 if modify_config used (LR isn't the problem).
    Full credit requires inspecting gradients, weights, data, AND modes.
    """
    score = 0.0

    # Investigation credits (+0.05 each, max +0.15 for 3 types)
    if state.gradients_inspected:
        score += 0.05
    if state.model_modes_inspected:
        score += 0.05
    if state.data_inspected:
        score += 0.05

    # Red herring penalty: -0.20 for add_callback after normal gradients
    if (
        _has_action(state, "add_callback")
        and state.gradients_inspected
        and state.gradients_were_normal
    ):
        score -= 0.20

    # Wrong-fix penalty: -0.10 for modify_config (LR isn't the issue here)
    if _has_action(state, "modify_config"):
        score -= 0.10

    # Fix credit scaled by investigation thoroughness
    # Full credit requires weight inspection (rules out weight-related causes)
    if _has_action(state, "fix_model_mode"):
        if state.model_weights_inspected and state.data_inspected:
            score += 0.25  # Thorough: checked weights AND data
        elif state.model_weights_inspected or state.data_inspected:
            score += 0.15  # Partial thoroughness
        else:
            score += 0.08  # Quick fix without ruling out alternatives

    # Restart credit scaled similarly
    if state.restart_after_fix:
        if state.model_weights_inspected:
            score += 0.20  # Full restart credit
        else:
            score += 0.10  # Partial credit

    # +0.40 for correct diagnosis
    if _correct_diagnosis(state, scenario):
        score += 0.40

    return min(1.0, max(0.0, score))


def grade_task_006(state: EpisodeState, scenario: ScenarioParams) -> float:
    """Grade Task 6 — PyTorch Code Bug (hard). Spec Section 11.

    Diagnosis must ALWAYS be 'code_bug' regardless of bug variant.
    Hard task rewards thorough investigation before fixing.
    Full credit requires ruling out non-code causes via weight inspection.
    """
    score = 0.0

    # Investigation credits (+0.05 each, up to +0.25 for all 5 types)
    if state.code_inspected:
        score += 0.05
    if state.gradients_inspected:
        score += 0.05
    if state.model_modes_inspected:
        score += 0.05
    if state.model_weights_inspected:
        score += 0.05
    if state.data_inspected:
        score += 0.05

    # Code fix credit scaled by investigation thoroughness
    if _has_action(state, "fix_code") and state.fix_action_taken:
        if state.model_weights_inspected:
            score += 0.15  # Thorough: ruled out weight-related causes
        else:
            score += 0.08  # Quick fix without full investigation

    # Restart credit scaled by thoroughness
    if state.restart_after_fix:
        if state.model_weights_inspected:
            score += 0.15  # Full restart credit
        else:
            score += 0.08  # Partial credit

    # +0.45 for correct diagnosis (must be code_bug)
    if _correct_diagnosis(state, scenario):
        score += 0.45

    return min(1.0, max(0.0, score))


def grade_task_007(state: EpisodeState, scenario: ScenarioParams) -> float:
    """Grade Task 7 — LR Scheduler Misconfigured (hard). Spec extension.

    Requires thorough investigation: agents must inspect weights to rule out
    weight-related issues before concluding scheduler is the root cause.
    Penalizes wrong fixes (e.g. patch_data_loader when data is fine).
    """
    score = 0.0

    # Investigation credits (+0.05 each, up to +0.20 for all 4 types)
    if state.gradients_inspected:
        score += 0.05
    if state.data_inspected:
        score += 0.05
    if state.model_weights_inspected:
        score += 0.05
    if state.model_modes_inspected:
        score += 0.05

    # Fix credit scaled by investigation thoroughness
    if _has_action(state, "modify_config"):
        if state.model_weights_inspected:
            score += 0.20  # Thorough: ruled out weight issues
        else:
            score += 0.12  # Partial: didn't check weights

    # Restart credit scaled by thoroughness
    if state.restart_after_fix:
        if state.model_weights_inspected:
            score += 0.20  # Full restart credit
        else:
            score += 0.12  # Partial credit

    # Diagnosis
    if _correct_diagnosis(state, scenario):
        score += 0.40

    # Wrong-fix penalty: patch_data_loader when data is clean
    if _has_action(state, "patch_data_loader"):
        score -= 0.10

    return min(1.0, max(0.0, score))


# Registry mapping task IDs to grader functions
GRADERS = {
    "task_001": grade_task_001,
    "task_002": grade_task_002,
    "task_003": grade_task_003,
    "task_004": grade_task_004,
    "task_005": grade_task_005,
    "task_006": grade_task_006,
    "task_007": grade_task_007,
}


def grade_episode(task_id: str, state: EpisodeState, scenario: ScenarioParams) -> float:
    """Grade a completed episode. Returns 0.0-1.0."""
    grader = GRADERS.get(task_id)
    if grader is None:
        return 0.0
    return grader(state, scenario)
