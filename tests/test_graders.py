"""Test grader functions — each returns 0.0-1.0."""

from __future__ import annotations

import pytest

from ml_training_debugger.graders import (
    _submitted_diagnosis,
    grade_episode,
    grade_task_001,
    grade_task_003,
    grade_task_005,
    grade_task_006,
    grade_task_007,
)
from ml_training_debugger.models import EpisodeState
from ml_training_debugger.scenarios import sample_scenario


@pytest.fixture
def scenario_001():
    return sample_scenario("task_001", seed=42)


@pytest.fixture
def scenario_003():
    return sample_scenario("task_003", seed=42)


@pytest.fixture
def scenario_005():
    return sample_scenario("task_005", seed=42)


class TestGradeTask001:
    def test_perfect_score(self, scenario_001):
        state = EpisodeState(
            gradients_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_gradients",
                "modify_config",
                "restart_run",
                "mark_diagnosed:lr_too_high",
            ],
        )
        score = grade_task_001(state, scenario_001)
        assert score == 1.0

    def test_wrong_diagnosis(self, scenario_001):
        state = EpisodeState(
            gradients_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_gradients",
                "modify_config",
                "restart_run",
                "mark_diagnosed:data_leakage",
            ],
        )
        score = grade_task_001(state, scenario_001)
        assert score < 0.7  # Missing diagnosis credit

    def test_no_investigation(self, scenario_001):
        state = EpisodeState(
            diagnosis_submitted=True,
            actions_taken=["mark_diagnosed:lr_too_high"],
        )
        score = grade_task_001(state, scenario_001)
        assert 0.0 < score < 1.0

    def test_score_in_range(self, scenario_001):
        state = EpisodeState()
        score = grade_task_001(state, scenario_001)
        assert 0.0 <= score <= 1.0


class TestGradeTask003:
    def test_perfect_score(self, scenario_003):
        state = EpisodeState(
            data_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_data_batch",
                "patch_data_loader",
                "restart_run",
                "mark_diagnosed:data_leakage",
            ],
        )
        score = grade_task_003(state, scenario_003)
        assert score == pytest.approx(1.0)

    def test_wrong_diagnosis(self, scenario_003):
        state = EpisodeState(
            data_inspected=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_data_batch",
                "mark_diagnosed:overfitting",
            ],
        )
        score = grade_task_003(state, scenario_003)
        assert score < 0.5


class TestGradeTask005:
    def test_perfect_score_thorough(self, scenario_005):
        """Thorough agent inspects weights AND data — gets perfect score."""
        state = EpisodeState(
            gradients_inspected=True,
            gradients_were_normal=True,
            model_modes_inspected=True,
            model_weights_inspected=True,
            data_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_gradients",
                "inspect_data_batch",
                "inspect_model_weights",
                "inspect_model_modes",
                "fix_model_mode",
                "restart_run",
                "mark_diagnosed:batchnorm_eval_mode",
            ],
        )
        score = grade_task_005(state, scenario_005)
        assert score == pytest.approx(1.0)

    def test_quick_fix_partial_credit(self, scenario_005):
        """Agent that skips weight inspection gets partial credit."""
        state = EpisodeState(
            gradients_inspected=True,
            gradients_were_normal=True,
            model_modes_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_gradients",
                "inspect_model_modes",
                "fix_model_mode",
                "restart_run",
                "mark_diagnosed:batchnorm_eval_mode",
            ],
        )
        score = grade_task_005(state, scenario_005)
        # No weights, no data → 0.05+0.05+0.08+0.10+0.40 = 0.68
        assert score == pytest.approx(0.68)

    def test_red_herring_chaser(self, scenario_005):
        """Agent that chases gradient red herring loses 0.20."""
        state = EpisodeState(
            gradients_inspected=True,
            gradients_were_normal=True,
            model_modes_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_gradients",
                "add_callback",  # Wrong: chases red herring
                "inspect_model_modes",
                "fix_model_mode",
                "restart_run",
                "mark_diagnosed:batchnorm_eval_mode",
            ],
        )
        score = grade_task_005(state, scenario_005)
        # -0.20 penalty for add_callback after normal gradients
        assert score == pytest.approx(0.48)

    def test_wrong_fix_penalty(self, scenario_005):
        """Agent that tries modify_config (wrong fix) gets penalized."""
        state = EpisodeState(
            gradients_inspected=True,
            gradients_were_normal=True,
            model_modes_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_gradients",
                "modify_config",  # Wrong: LR isn't the problem
                "inspect_model_modes",
                "fix_model_mode",
                "restart_run",
                "mark_diagnosed:batchnorm_eval_mode",
            ],
        )
        score = grade_task_005(state, scenario_005)
        # -0.10 penalty for modify_config on task 5
        assert score == pytest.approx(0.58)

    def test_double_trap_devastates_score(self, scenario_005):
        """Agent that falls for BOTH traps (add_callback + modify_config) scores poorly."""
        state = EpisodeState(
            gradients_inspected=True,
            gradients_were_normal=True,
            model_modes_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_gradients",
                "add_callback",
                "modify_config",
                "inspect_model_modes",
                "fix_model_mode",
                "restart_run",
                "mark_diagnosed:batchnorm_eval_mode",
            ],
        )
        score = grade_task_005(state, scenario_005)
        # -0.20 (add_callback) + -0.10 (modify_config) = -0.30 penalties
        assert score == pytest.approx(0.38)


class TestGradeEpisode:
    def test_dispatch_to_correct_grader(self, scenario_001):
        state = EpisodeState(
            gradients_inspected=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_gradients",
                "mark_diagnosed:lr_too_high",
            ],
        )
        score = grade_episode("task_001", state, scenario_001)
        assert 0.0 <= score <= 1.0

    def test_unknown_task_returns_zero(self, scenario_001):
        state = EpisodeState()
        score = grade_episode("task_999", state, scenario_001)
        assert score == 0.0


class TestGradeTask006:
    @pytest.fixture
    def scenario_006(self):
        return sample_scenario("task_006", seed=42)

    def test_perfect_score_thorough(self, scenario_006):
        """Thorough agent inspects ALL systems including weights — gets perfect score."""
        state = EpisodeState(
            code_inspected=True,
            gradients_inspected=True,
            model_modes_inspected=True,
            model_weights_inspected=True,
            data_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_gradients",
                "inspect_data_batch",
                "inspect_model_weights",
                "inspect_model_modes",
                "inspect_code",
                "fix_code",
                "restart_run",
                "mark_diagnosed:code_bug",
            ],
        )
        score = grade_task_006(state, scenario_006)
        assert score == pytest.approx(1.0)

    def test_no_weights_inspection_partial(self, scenario_006):
        """Agent that skips weight inspection gets reduced fix/restart credit."""
        state = EpisodeState(
            code_inspected=True,
            gradients_inspected=True,
            model_modes_inspected=True,
            data_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_gradients",
                "inspect_data_batch",
                "inspect_model_modes",
                "inspect_code",
                "fix_code",
                "restart_run",
                "mark_diagnosed:code_bug",
            ],
        )
        score = grade_task_006(state, scenario_006)
        # 0.05*4 + 0.08 + 0.08 + 0.45 = 0.81
        assert score == pytest.approx(0.81)
        assert score < 1.0  # Must not be perfect without weights

    def test_minimal_investigation(self, scenario_006):
        """Agent that only inspects code, fixes, and diagnoses."""
        state = EpisodeState(
            code_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_code",
                "fix_code",
                "restart_run",
                "mark_diagnosed:code_bug",
            ],
        )
        score = grade_task_006(state, scenario_006)
        # 0.05 + 0.08 + 0.08 + 0.45 = 0.66
        assert score == pytest.approx(0.66)

    def test_wrong_diagnosis(self, scenario_006):
        """Submitting batchnorm_eval_mode on a code_bug task fails."""
        state = EpisodeState(
            code_inspected=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_code",
                "mark_diagnosed:batchnorm_eval_mode",
            ],
        )
        score = grade_task_006(state, scenario_006)
        assert score < 0.2  # Only gets code_inspected bonus

    def test_score_in_range(self, scenario_006):
        state = EpisodeState()
        score = grade_task_006(state, scenario_006)
        assert 0.0 <= score <= 1.0


class TestGradeTask007:
    def test_perfect_score_thorough(self):
        """Thorough agent inspects weights — gets perfect score."""
        scenario = sample_scenario("task_007", seed=42)
        state = EpisodeState(
            gradients_inspected=True,
            data_inspected=True,
            model_weights_inspected=True,
            model_modes_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_gradients",
                "inspect_data_batch",
                "inspect_model_weights",
                "inspect_model_modes",
                "modify_config",
                "restart_run",
                "mark_diagnosed:scheduler_misconfigured",
            ],
        )
        score = grade_task_007(state, scenario)
        assert score == pytest.approx(1.0)

    def test_no_weights_partial(self):
        """Agent without weight inspection gets reduced fix/restart credit."""
        scenario = sample_scenario("task_007", seed=42)
        state = EpisodeState(
            gradients_inspected=True,
            data_inspected=True,
            model_modes_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_gradients",
                "inspect_data_batch",
                "inspect_model_modes",
                "modify_config",
                "restart_run",
                "mark_diagnosed:scheduler_misconfigured",
            ],
        )
        score = grade_task_007(state, scenario)
        # 0.05*3 + 0.12 + 0.12 + 0.40 = 0.79
        assert score == pytest.approx(0.79)
        assert score < 1.0

    def test_wrong_fix_penalty(self):
        """Agent that patches data loader (wrong fix) gets penalized."""
        scenario = sample_scenario("task_007", seed=42)
        state = EpisodeState(
            gradients_inspected=True,
            data_inspected=True,
            fix_action_taken=True,
            restart_after_fix=True,
            diagnosis_submitted=True,
            actions_taken=[
                "inspect_gradients",
                "inspect_data_batch",
                "patch_data_loader",
                "modify_config",
                "restart_run",
                "mark_diagnosed:scheduler_misconfigured",
            ],
        )
        score = grade_task_007(state, scenario)
        # Normal partial score minus 0.10 penalty
        assert score < 0.75

    def test_wrong_diagnosis(self):
        scenario = sample_scenario("task_007", seed=42)
        state = EpisodeState(
            diagnosis_submitted=True,
            actions_taken=["mark_diagnosed:overfitting"],
        )
        score = grade_task_007(state, scenario)
        assert score < 0.5

    def test_score_in_range(self):
        scenario = sample_scenario("task_007", seed=42)
        state = EpisodeState()
        score = grade_task_007(state, scenario)
        assert 0.0 <= score <= 1.0


class TestSubmittedDiagnosis:
    def test_finds_diagnosis(self):
        state = EpisodeState(
            actions_taken=["inspect_gradients", "mark_diagnosed:lr_too_high"],
        )
        assert _submitted_diagnosis(state) == "lr_too_high"

    def test_no_diagnosis(self):
        state = EpisodeState(actions_taken=["inspect_gradients"])
        assert _submitted_diagnosis(state) is None

    def test_latest_diagnosis(self):
        state = EpisodeState(
            actions_taken=[
                "mark_diagnosed:overfitting",
                "mark_diagnosed:lr_too_high",
            ],
        )
        assert _submitted_diagnosis(state) == "lr_too_high"
