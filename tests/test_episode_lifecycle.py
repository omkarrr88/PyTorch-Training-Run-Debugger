"""Test full episode lifecycle — reset, step, state transitions."""

from __future__ import annotations

import pytest

from ml_training_debugger.models import MLTrainingAction
from server.environment import MLTrainingEnvironment


@pytest.fixture
def env():
    return MLTrainingEnvironment()


class TestReset:
    def test_reset_returns_valid_observation(self, env):
        obs = env.reset(seed=42, episode_id="test", task_id="task_001")
        assert obs.run_id == "test"
        assert obs.framework == "pytorch"
        assert len(obs.training_loss_history) == 20
        assert len(obs.val_accuracy_history) == 20
        assert obs.done is False

    def test_reset_initial_state(self, env):
        obs = env.reset(seed=42, episode_id="test", task_id="task_001")
        assert obs.episode_state.step_count == 0
        assert not obs.episode_state.gradients_inspected
        assert not obs.episode_state.diagnosis_submitted

    def test_reset_progressive_reveal(self, env):
        obs = env.reset(seed=42, episode_id="test", task_id="task_001")
        assert obs.gradient_stats == []
        assert obs.model_weight_stats is None
        assert obs.data_batch_stats is None
        assert obs.model_mode_info is None
        assert obs.code_snippet is None

    def test_reset_available_actions(self, env):
        obs = env.reset(seed=42, episode_id="test", task_id="task_001")
        assert "inspect_gradients" in obs.available_actions
        assert "mark_diagnosed" in obs.available_actions
        assert "fix_code" not in obs.available_actions
        assert "restart_run" not in obs.available_actions


class TestStepInspections:
    def test_inspect_gradients_populates_stats(self, env):
        env.reset(seed=42, episode_id="test", task_id="task_001")
        obs = env.step(MLTrainingAction(action_type="inspect_gradients"))
        assert len(obs.gradient_stats) > 0
        assert obs.episode_state.gradients_inspected

    def test_inspect_gradients_gives_investigation_bonus(self, env):
        """First-time inspection must give +0.05 bonus (total +0.04 with step penalty)."""
        env.reset(seed=42, episode_id="test", task_id="task_001")
        obs = env.step(MLTrainingAction(action_type="inspect_gradients"))
        assert obs.reward == pytest.approx(0.04)

    def test_inspect_data_batch_gives_investigation_bonus(self, env):
        """First-time data inspection must give +0.05 bonus."""
        env.reset(seed=42, episode_id="test", task_id="task_003")
        obs = env.step(MLTrainingAction(action_type="inspect_data_batch"))
        assert obs.reward == pytest.approx(0.04)

    def test_inspect_model_modes_gives_investigation_bonus(self, env):
        """First-time model modes inspection must give +0.05 bonus."""
        env.reset(seed=42, episode_id="test", task_id="task_005")
        obs = env.step(MLTrainingAction(action_type="inspect_model_modes"))
        assert obs.reward == pytest.approx(0.04)

    def test_repeat_inspection_no_bonus(self, env):
        """Second inspection of same type must NOT give bonus."""
        env.reset(seed=42, episode_id="test", task_id="task_001")
        env.step(MLTrainingAction(action_type="inspect_gradients"))
        obs = env.step(MLTrainingAction(action_type="inspect_gradients"))
        assert obs.reward == pytest.approx(-0.01)

    def test_inspect_data_batch(self, env):
        env.reset(seed=42, episode_id="test", task_id="task_003")
        obs = env.step(MLTrainingAction(action_type="inspect_data_batch"))
        assert obs.data_batch_stats is not None
        assert obs.episode_state.data_inspected

    def test_inspect_model_modes(self, env):
        env.reset(seed=42, episode_id="test", task_id="task_005")
        obs = env.step(MLTrainingAction(action_type="inspect_model_modes"))
        assert obs.model_mode_info is not None
        assert obs.episode_state.model_modes_inspected

    def test_inspect_model_weights(self, env):
        env.reset(seed=42, episode_id="test", task_id="task_001")
        obs = env.step(MLTrainingAction(action_type="inspect_model_weights"))
        assert obs.model_weight_stats is not None
        assert obs.episode_state.model_weights_inspected


class TestStepFixActions:
    def test_modify_config(self, env):
        env.reset(seed=42, episode_id="test", task_id="task_001")
        obs = env.step(
            MLTrainingAction(
                action_type="modify_config",
                target="learning_rate",
                value=0.001,
            )
        )
        assert obs.episode_state.fix_action_taken
        assert "restart_run" in obs.available_actions

    def test_restart_run_after_fix(self, env):
        env.reset(seed=42, episode_id="test", task_id="task_001")
        env.step(
            MLTrainingAction(
                action_type="modify_config",
                target="learning_rate",
                value=0.001,
            )
        )
        obs = env.step(MLTrainingAction(action_type="restart_run"))
        assert obs.episode_state.restart_after_fix


class TestStepDiagnosis:
    def test_mark_diagnosed_ends_episode(self, env):
        env.reset(seed=42, episode_id="test", task_id="task_001")
        obs = env.step(
            MLTrainingAction(
                action_type="mark_diagnosed",
                diagnosis="lr_too_high",
            )
        )
        assert obs.done is True
        assert obs.episode_state.diagnosis_submitted

    def test_step_after_done(self, env):
        env.reset(seed=42, episode_id="test", task_id="task_001")
        env.step(
            MLTrainingAction(
                action_type="mark_diagnosed",
                diagnosis="lr_too_high",
            )
        )
        obs = env.step(MLTrainingAction(action_type="inspect_gradients"))
        assert obs.done is True
        assert obs.reward == 0.0


class TestErrorHandling:
    def test_invalid_action_type(self, env):
        env.reset(seed=42, episode_id="test", task_id="task_001")
        obs = env.step(MLTrainingAction(action_type="nonexistent_action"))
        assert obs.reward == pytest.approx(-0.01 + -0.05)
        assert obs.error_log is not None

    def test_action_not_in_available(self, env):
        env.reset(seed=42, episode_id="test", task_id="task_001")
        # fix_code requires code_inspected=True
        obs = env.step(
            MLTrainingAction(
                action_type="fix_code",
                line=1,
                replacement="pass",
            )
        )
        assert obs.reward < 0

    def test_modify_config_missing_target(self, env):
        env.reset(seed=42, episode_id="test", task_id="task_001")
        obs = env.step(MLTrainingAction(action_type="modify_config"))
        assert "target" in obs.error_log.lower() or "value" in obs.error_log.lower()

    def test_mark_diagnosed_missing_diagnosis(self, env):
        env.reset(seed=42, episode_id="test", task_id="task_001")
        obs = env.step(MLTrainingAction(action_type="mark_diagnosed"))
        assert "diagnosis" in obs.error_log.lower()

    def test_mark_diagnosed_invalid_diagnosis(self, env):
        env.reset(seed=42, episode_id="test", task_id="task_001")
        obs = env.step(
            MLTrainingAction(
                action_type="mark_diagnosed",
                diagnosis="not_a_real_diagnosis",
            )
        )
        assert "invalid" in obs.error_log.lower()

    def test_step_before_reset(self, env):
        obs = env.step(MLTrainingAction(action_type="inspect_gradients"))
        assert obs.done is True


class TestFullEpisodeFlow:
    def test_task_001_full_flow(self, env):
        """Full optimal flow for Task 1."""
        obs = env.reset(seed=42, episode_id="test", task_id="task_001")
        assert not obs.done

        obs = env.step(MLTrainingAction(action_type="inspect_gradients"))
        assert obs.episode_state.gradients_inspected
        assert any(g.is_exploding for g in obs.gradient_stats)

        obs = env.step(
            MLTrainingAction(
                action_type="modify_config",
                target="learning_rate",
                value=0.001,
            )
        )
        assert obs.episode_state.fix_action_taken

        obs = env.step(MLTrainingAction(action_type="restart_run"))
        assert obs.episode_state.restart_after_fix

        obs = env.step(
            MLTrainingAction(
                action_type="mark_diagnosed",
                diagnosis="lr_too_high",
            )
        )
        assert obs.done
        assert obs.reward > 0

    def test_task_005_context_gated_penalty(self, env):
        """Task 5: inspect gradients (normal) → add_callback → penalty fires."""
        obs = env.reset(seed=42, episode_id="test", task_id="task_005")

        obs = env.step(MLTrainingAction(action_type="inspect_gradients"))
        assert obs.episode_state.gradients_inspected
        assert obs.episode_state.gradients_were_normal
        # All layers is_exploding=False
        for g in obs.gradient_stats:
            assert not g.is_exploding

        # Now add_callback should trigger context-gated penalty
        obs = env.step(MLTrainingAction(action_type="add_callback"))
        assert obs.reward == pytest.approx(-0.01 + -0.20)

    def test_task_003_data_leakage(self, env):
        """Task 3: data inspection reveals leakage."""
        obs = env.reset(seed=42, episode_id="test", task_id="task_003")

        obs = env.step(MLTrainingAction(action_type="inspect_data_batch"))
        assert obs.data_batch_stats is not None
        assert obs.data_batch_stats.class_overlap_score > 0.5
