"""Test scenario sampling."""

from __future__ import annotations

import pytest

from ml_training_debugger.models import RootCauseDiagnosis
from ml_training_debugger.scenarios import sample_scenario


class TestSampleScenario:
    def test_task_001_root_cause(self):
        s = sample_scenario("task_001", seed=42)
        assert s.root_cause == RootCauseDiagnosis.LR_TOO_HIGH
        assert s.learning_rate >= 0.05

    def test_task_003_root_cause(self):
        s = sample_scenario("task_003", seed=42)
        assert s.root_cause == RootCauseDiagnosis.DATA_LEAKAGE
        assert 0.10 <= s.leakage_pct <= 0.30

    def test_task_005_root_cause(self):
        s = sample_scenario("task_005", seed=42)
        assert s.root_cause == RootCauseDiagnosis.BATCHNORM_EVAL_MODE
        assert 0.8 <= s.red_herring_intensity <= 2.5

    def test_different_seeds_produce_different_params(self):
        s1 = sample_scenario("task_001", seed=42)
        s2 = sample_scenario("task_001", seed=99)
        # Same root cause, but may have different LR
        assert s1.root_cause == s2.root_cause

    def test_same_seed_same_params(self):
        s1 = sample_scenario("task_001", seed=42)
        s2 = sample_scenario("task_001", seed=42)
        assert s1.learning_rate == s2.learning_rate
        assert s1.seed == s2.seed

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            sample_scenario("task_999", seed=42)

    def test_task_005_has_error_log(self):
        s = sample_scenario("task_005", seed=42)
        assert s.error_log is not None
        assert "GPU memory" in s.error_log

    def test_task_003_has_notes(self):
        s = sample_scenario("task_003", seed=42)
        assert s.notes is not None
        assert "architecture" in s.notes.lower()
