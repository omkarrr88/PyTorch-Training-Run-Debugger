"""Test grader functions — each returns 0.0-1.0."""

from __future__ import annotations

import pytest

from ml_training_debugger.graders import (
    grade_episode,
    grade_task_001,
    grade_task_003,
    grade_task_005,
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
    def test_perfect_score(self, scenario_005):
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
        assert score == 1.0

    def test_red_herring_chaser(self, scenario_005):
        """Agent that chases gradient red herring scores 0.80-0.85."""
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
        assert 0.7 <= score <= 0.90


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
