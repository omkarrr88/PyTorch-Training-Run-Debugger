"""Tests for parametric fallback in simulation.py.

These test the fallback paths that run when real training is unavailable.
We force fallback by monkeypatching _get_real_curves to return None.
"""

from __future__ import annotations

from unittest.mock import patch

from ml_training_debugger.scenarios import sample_scenario
from ml_training_debugger.simulation import (
    gen_loss_history,
    gen_val_accuracy_history,
    gen_val_loss_history,
)


def _force_fallback(*args, **kwargs):
    return None


class TestParametricFallbackLoss:
    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_001_fallback(self) -> None:
        s = sample_scenario("task_001", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_002_fallback(self) -> None:
        s = sample_scenario("task_002", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_003_fallback(self) -> None:
        s = sample_scenario("task_003", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_004_fallback(self) -> None:
        s = sample_scenario("task_004", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_005_fallback(self) -> None:
        s = sample_scenario("task_005", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_006_fallback(self) -> None:
        s = sample_scenario("task_006", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_007_fallback(self) -> None:
        s = sample_scenario("task_007", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20


class TestParametricFallbackValAcc:
    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_001_fallback(self) -> None:
        s = sample_scenario("task_001", seed=42)
        hist = gen_val_accuracy_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_003_fallback(self) -> None:
        s = sample_scenario("task_003", seed=42)
        hist = gen_val_accuracy_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_004_fallback(self) -> None:
        s = sample_scenario("task_004", seed=42)
        hist = gen_val_accuracy_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_005_fallback(self) -> None:
        s = sample_scenario("task_005", seed=42)
        hist = gen_val_accuracy_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_006_fallback(self) -> None:
        s = sample_scenario("task_006", seed=42)
        hist = gen_val_accuracy_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_007_fallback(self) -> None:
        s = sample_scenario("task_007", seed=42)
        hist = gen_val_accuracy_history(s)
        assert len(hist) == 20


class TestParametricFallbackValLoss:
    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_001_fallback(self) -> None:
        s = sample_scenario("task_001", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_004_fallback(self) -> None:
        s = sample_scenario("task_004", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_005_fallback(self) -> None:
        s = sample_scenario("task_005", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_006_fallback(self) -> None:
        s = sample_scenario("task_006", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_task_007_fallback(self) -> None:
        s = sample_scenario("task_007", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20

    @patch("ml_training_debugger.simulation._get_real_curves", _force_fallback)
    def test_fallback_default(self) -> None:
        """Test the final fallback path for unknown root cause."""
        from ml_training_debugger.models import RootCauseDiagnosis
        from ml_training_debugger.scenarios import ScenarioParams

        # Use scheduler root cause but force fallback
        s = ScenarioParams(
            task_id="task_999",
            root_cause=RootCauseDiagnosis.SCHEDULER_MISCONFIGURED,
            seed=42,
        )
        hist = gen_val_loss_history(s)
        assert len(hist) == 20
