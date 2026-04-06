"""Reward function — 7 components, separate from graders.

Returns a float per step for RL training signal. Hard cap at [-1.0, 1.0].
"""

from __future__ import annotations

import torch  # noqa: F401

from ml_training_debugger.models import EpisodeState, MLTrainingAction
from ml_training_debugger.scenarios import ScenarioParams

# Reward constants
STEP_PENALTY = -0.01
INVESTIGATION_BONUS = 0.05
CONTEXT_GATED_PENALTY = -0.20
INVALID_ACTION_PENALTY = -0.05
WRONG_CODE_FIX_PENALTY = -0.10
CORRECT_DIAGNOSIS_REWARD = 0.50
WRONG_DIAGNOSIS_PENALTY = -0.30
TERMINAL_CONVERGENCE_REWARD = 0.40

INVESTIGATION_ACTIONS = frozenset(
    {
        "inspect_gradients",
        "inspect_data_batch",
        "inspect_model_modes",
        "inspect_model_weights",
        "inspect_code",
    }
)

_INSPECTION_STATE_MAP = {
    "inspect_gradients": "gradients_inspected",
    "inspect_data_batch": "data_inspected",
    "inspect_model_modes": "model_modes_inspected",
    "inspect_model_weights": "model_weights_inspected",
    "inspect_code": "code_inspected",
}


def compute_reward(
    action: MLTrainingAction,
    state: EpisodeState,
    scenario: ScenarioParams,
    is_valid_action: bool = True,
    is_correct_fix: bool | None = None,
    convergence_confirmed: bool = False,
) -> float:
    """Compute reward for a single step.

    Args:
        action: The action taken.
        state: Episode state BEFORE the action is applied.
        scenario: Current scenario params.
        is_valid_action: Whether the action is in available_actions.
        is_correct_fix: For fix_code — True/False/None.
        convergence_confirmed: Whether restart showed convergence.

    Returns:
        Reward float, capped at [-1.0, 1.0].
    """
    reward = 0.0

    # Step penalty (unconditional)
    reward += STEP_PENALTY

    if not is_valid_action:
        reward += INVALID_ACTION_PENALTY
        return max(-1.0, min(1.0, reward))

    action_type = action.action_type

    # Investigation bonus — first-time only
    if action_type in INVESTIGATION_ACTIONS:
        state_field = _INSPECTION_STATE_MAP.get(action_type)
        if state_field and not getattr(state, state_field):
            reward += INVESTIGATION_BONUS

    # Context-gated penalty: adding gradient clipping after seeing normal gradients
    if action_type == "add_callback":
        if state.gradients_inspected and state.gradients_were_normal:
            reward += CONTEXT_GATED_PENALTY

    # Wrong code fix
    if action_type == "fix_code" and is_correct_fix is False:
        reward += WRONG_CODE_FIX_PENALTY

    # Diagnosis
    if action_type == "mark_diagnosed":
        if action.diagnosis == scenario.root_cause.value:
            reward += CORRECT_DIAGNOSIS_REWARD
        else:
            reward += WRONG_DIAGNOSIS_PENALTY

    # Convergence after fix+restart
    if action_type == "restart_run":
        if state.fix_action_taken and convergence_confirmed:
            reward += TERMINAL_CONVERGENCE_REWARD

    return max(-1.0, min(1.0, reward))
