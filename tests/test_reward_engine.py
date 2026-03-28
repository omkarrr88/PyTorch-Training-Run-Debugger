"""Test reward engine — all 7 components. THE MOST CRITICAL TEST FILE."""

from __future__ import annotations

import pytest

from ml_training_debugger.models import EpisodeState, MLTrainingAction
from ml_training_debugger.reward_engine import (
    CONTEXT_GATED_PENALTY,
    CORRECT_DIAGNOSIS_REWARD,
    INVALID_ACTION_PENALTY,
    INVESTIGATION_BONUS,
    STEP_PENALTY,
    TERMINAL_CONVERGENCE_REWARD,
    WRONG_CODE_FIX_PENALTY,
    WRONG_DIAGNOSIS_PENALTY,
    compute_reward,
)
from ml_training_debugger.scenarios import sample_scenario


@pytest.fixture
def scenario():
    return sample_scenario("task_001", seed=42)


@pytest.fixture
def scenario_005():
    return sample_scenario("task_005", seed=42)


class TestStepPenalty:
    def test_flat_step_penalty(self, scenario):
        state = EpisodeState()
        action = MLTrainingAction(action_type="add_callback")
        reward = compute_reward(action, state, scenario)
        assert reward == pytest.approx(STEP_PENALTY)

    def test_step_penalty_not_multiplied_by_step_count(self, scenario):
        state = EpisodeState(step_count=30)
        action = MLTrainingAction(action_type="add_callback")
        reward = compute_reward(action, state, scenario)
        # Must be flat -0.01, NOT -0.01 * 30
        assert reward == pytest.approx(-0.01)


class TestInvestigationBonus:
    def test_first_time_bonus(self, scenario):
        state = EpisodeState(gradients_inspected=False)
        action = MLTrainingAction(action_type="inspect_gradients")
        reward = compute_reward(action, state, scenario)
        assert reward == pytest.approx(STEP_PENALTY + INVESTIGATION_BONUS)

    def test_no_bonus_on_repeat(self, scenario):
        state = EpisodeState(gradients_inspected=True)
        action = MLTrainingAction(action_type="inspect_gradients")
        reward = compute_reward(action, state, scenario)
        assert reward == pytest.approx(STEP_PENALTY)

    def test_each_inspection_type_gives_bonus(self, scenario):
        for action_type, field in [
            ("inspect_gradients", "gradients_inspected"),
            ("inspect_data_batch", "data_inspected"),
            ("inspect_model_modes", "model_modes_inspected"),
            ("inspect_model_weights", "model_weights_inspected"),
            ("inspect_code", "code_inspected"),
        ]:
            state = EpisodeState(**{field: False})
            action = MLTrainingAction(action_type=action_type)
            reward = compute_reward(action, state, scenario)
            assert reward == pytest.approx(
                STEP_PENALTY + INVESTIGATION_BONUS
            ), f"Failed for {action_type}"


class TestContextGatedPenalty:
    """The project's primary innovation — must be exact."""

    def test_no_penalty_before_inspection(self, scenario_005):
        """add_callback at step 1 (no prior inspection) -> NO penalty."""
        state = EpisodeState()
        action = MLTrainingAction(action_type="add_callback")
        reward = compute_reward(action, state, scenario_005)
        assert reward == pytest.approx(STEP_PENALTY)

    def test_penalty_after_normal_gradients(self, scenario_005):
        """inspect_gradients (normal) then add_callback -> -0.20 penalty."""
        state = EpisodeState(gradients_inspected=True, gradients_were_normal=True)
        action = MLTrainingAction(action_type="add_callback")
        reward = compute_reward(action, state, scenario_005)
        assert reward == pytest.approx(STEP_PENALTY + CONTEXT_GATED_PENALTY)

    def test_no_penalty_after_abnormal_gradients(self, scenario):
        """inspect_gradients (exploding) then add_callback -> no context penalty."""
        state = EpisodeState(gradients_inspected=True, gradients_were_normal=False)
        action = MLTrainingAction(action_type="add_callback")
        reward = compute_reward(action, state, scenario)
        assert reward == pytest.approx(STEP_PENALTY)

    def test_penalty_only_for_add_callback(self, scenario_005):
        """Other fix actions don't trigger context-gated penalty."""
        state = EpisodeState(gradients_inspected=True, gradients_were_normal=True)
        for action_type in ["modify_config", "fix_model_mode", "patch_data_loader"]:
            action = MLTrainingAction(
                action_type=action_type, target="learning_rate", value=0.001
            )
            reward = compute_reward(action, state, scenario_005)
            assert reward == pytest.approx(
                STEP_PENALTY
            ), f"Unexpected penalty for {action_type}"


class TestDiagnosisReward:
    def test_correct_diagnosis(self, scenario):
        state = EpisodeState()
        action = MLTrainingAction(action_type="mark_diagnosed", diagnosis="lr_too_high")
        reward = compute_reward(action, state, scenario)
        assert reward == pytest.approx(STEP_PENALTY + CORRECT_DIAGNOSIS_REWARD)

    def test_wrong_diagnosis(self, scenario):
        state = EpisodeState()
        action = MLTrainingAction(
            action_type="mark_diagnosed", diagnosis="data_leakage"
        )
        reward = compute_reward(action, state, scenario)
        assert reward == pytest.approx(STEP_PENALTY + WRONG_DIAGNOSIS_PENALTY)


class TestTerminalConvergence:
    def test_convergence_after_fix_and_restart(self, scenario):
        state = EpisodeState(fix_action_taken=True)
        action = MLTrainingAction(action_type="restart_run")
        reward = compute_reward(action, state, scenario, convergence_confirmed=True)
        assert reward == pytest.approx(STEP_PENALTY + TERMINAL_CONVERGENCE_REWARD)

    def test_no_convergence_without_fix(self, scenario):
        state = EpisodeState(fix_action_taken=False)
        action = MLTrainingAction(action_type="restart_run")
        reward = compute_reward(action, state, scenario, convergence_confirmed=True)
        # fix_action_taken is False, so no convergence reward
        assert reward == pytest.approx(STEP_PENALTY)


class TestInvalidAction:
    def test_invalid_action_penalty(self, scenario):
        state = EpisodeState()
        action = MLTrainingAction(action_type="restart_run")
        reward = compute_reward(action, state, scenario, is_valid_action=False)
        assert reward == pytest.approx(STEP_PENALTY + INVALID_ACTION_PENALTY)


class TestWrongCodeFix:
    def test_wrong_code_fix_penalty(self, scenario):
        state = EpisodeState(code_inspected=True)
        action = MLTrainingAction(action_type="fix_code", line=1, replacement="pass")
        reward = compute_reward(action, state, scenario, is_correct_fix=False)
        assert reward == pytest.approx(STEP_PENALTY + WRONG_CODE_FIX_PENALTY)


class TestRewardCap:
    def test_reward_capped_at_one(self, scenario):
        # Theoretical max would exceed 1.0 in some scenarios
        reward = compute_reward(
            MLTrainingAction(action_type="mark_diagnosed", diagnosis="lr_too_high"),
            EpisodeState(),
            scenario,
        )
        assert reward <= 1.0

    def test_reward_capped_at_negative_one(self, scenario):
        reward = compute_reward(
            MLTrainingAction(action_type="mark_diagnosed", diagnosis="wrong"),
            EpisodeState(),
            scenario,
        )
        assert reward >= -1.0
