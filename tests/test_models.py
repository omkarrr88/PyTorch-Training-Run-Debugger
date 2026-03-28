"""Test all Pydantic models instantiate and serialize correctly."""

from __future__ import annotations

import json

from openenv.core.env_server.types import Action, Observation

from ml_training_debugger.models import (
    EpisodeState,
    GradientStats,
    MLTrainingAction,
    MLTrainingObservation,
    RootCauseDiagnosis,
    TrainingConfig,
)


class TestRootCauseDiagnosis:
    def test_all_six_values_exist(self):
        assert len(RootCauseDiagnosis) == 6

    def test_values_are_strings(self):
        for d in RootCauseDiagnosis:
            assert isinstance(d.value, str)

    def test_specific_values(self):
        assert RootCauseDiagnosis.LR_TOO_HIGH.value == "lr_too_high"
        assert RootCauseDiagnosis.CODE_BUG.value == "code_bug"


class TestTrainingConfig:
    def test_default_instantiation(self):
        config = TrainingConfig()
        assert config.learning_rate == 0.001
        assert config.gradient_clip_norm is None

    def test_json_roundtrip(self):
        config = TrainingConfig(learning_rate=0.01, weight_decay=0.1)
        data = json.loads(config.model_dump_json())
        restored = TrainingConfig.model_validate(data)
        assert restored.learning_rate == 0.01
        assert restored.weight_decay == 0.1


class TestGradientStats:
    def test_exploding(self):
        stats = GradientStats(
            layer_name="fc",
            norm_history=[15.0],
            mean_norm=15.0,
            max_norm=15.0,
            is_exploding=True,
            is_vanishing=False,
        )
        assert stats.is_exploding

    def test_vanishing(self):
        stats = GradientStats(
            layer_name="conv1",
            norm_history=[1e-7],
            mean_norm=1e-7,
            max_norm=1e-7,
            is_exploding=False,
            is_vanishing=True,
        )
        assert stats.is_vanishing

    def test_normal(self):
        stats = GradientStats(
            layer_name="conv1",
            norm_history=[0.5],
            mean_norm=0.5,
            max_norm=0.5,
            is_exploding=False,
            is_vanishing=False,
        )
        assert not stats.is_exploding
        assert not stats.is_vanishing


class TestEpisodeState:
    def test_fresh_state(self):
        state = EpisodeState()
        assert state.step_count == 0
        assert not state.gradients_inspected
        assert not state.diagnosis_submitted

    def test_available_actions_initial(self):
        state = EpisodeState()
        actions = state.compute_available_actions()
        assert "inspect_gradients" in actions
        assert "mark_diagnosed" in actions
        assert "fix_code" not in actions
        assert "restart_run" not in actions
        assert "rollback_checkpoint" not in actions

    def test_fix_code_available_after_code_inspected(self):
        state = EpisodeState(code_inspected=True)
        actions = state.compute_available_actions()
        assert "fix_code" in actions

    def test_restart_run_available_after_fix(self):
        state = EpisodeState(fix_action_taken=True)
        actions = state.compute_available_actions()
        assert "restart_run" in actions

    def test_rollback_available_after_restart(self):
        state = EpisodeState(restart_after_fix=True)
        actions = state.compute_available_actions()
        assert "rollback_checkpoint" in actions

    def test_mark_diagnosed_disappears_after_submission(self):
        state = EpisodeState(diagnosis_submitted=True)
        actions = state.compute_available_actions()
        assert "mark_diagnosed" not in actions


class TestMLTrainingObservation:
    def test_extends_observation(self):
        assert issubclass(MLTrainingObservation, Observation)

    def test_has_done_and_reward(self):
        obs = MLTrainingObservation(done=True, reward=0.5)
        assert obs.done is True
        assert obs.reward == 0.5

    def test_json_serialization(self):
        obs = MLTrainingObservation(
            run_id="test",
            training_loss_history=[1.0, 2.0],
            val_accuracy_history=[0.5],
        )
        data = json.loads(obs.model_dump_json())
        assert data["run_id"] == "test"
        assert data["framework"] == "pytorch"


class TestMLTrainingAction:
    def test_extends_action(self):
        assert issubclass(MLTrainingAction, Action)

    def test_basic_action(self):
        action = MLTrainingAction(action_type="inspect_gradients")
        assert action.action_type == "inspect_gradients"

    def test_modify_config_action(self):
        action = MLTrainingAction(
            action_type="modify_config",
            target="learning_rate",
            value=0.001,
        )
        assert action.target == "learning_rate"

    def test_mark_diagnosed_action(self):
        action = MLTrainingAction(
            action_type="mark_diagnosed",
            diagnosis="lr_too_high",
        )
        assert action.diagnosis == "lr_too_high"

    def test_fix_code_action(self):
        action = MLTrainingAction(
            action_type="fix_code",
            line=13,
            replacement="loss = criterion(output, batch_y)",
        )
        assert action.line == 13
