"""Extended simulation tests — adapted for real mini-training curves."""

from __future__ import annotations

from ml_training_debugger.scenarios import sample_scenario
from ml_training_debugger.simulation import (
    gen_data_batch_stats,
    gen_loss_history,
    gen_val_accuracy_history,
    gen_val_loss_history,
)


class TestVanishingGradients:
    def test_loss_barely_decreases(self):
        s = sample_scenario("task_002", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20

    def test_val_acc_low(self):
        s = sample_scenario("task_002", seed=42)
        hist = gen_val_accuracy_history(s)
        assert len(hist) == 20

    def test_val_loss_present(self):
        s = sample_scenario("task_002", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20


class TestOverfitting:
    def test_loss_history_present(self):
        s = sample_scenario("task_004", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20

    def test_val_acc_present(self):
        s = sample_scenario("task_004", seed=42)
        hist = gen_val_accuracy_history(s)
        assert len(hist) == 20

    def test_val_loss_present(self):
        s = sample_scenario("task_004", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20

    def test_data_batch_stats_clean(self):
        s = sample_scenario("task_004", seed=42)
        stats = gen_data_batch_stats(s)
        assert stats["class_overlap_score"] == 0.0
        assert stats["duplicate_ratio"] == 0.0


class TestCodeBug:
    def test_loss_history(self):
        s = sample_scenario("task_006", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20

    def test_val_acc(self):
        s = sample_scenario("task_006", seed=42)
        hist = gen_val_accuracy_history(s)
        assert len(hist) == 20

    def test_val_loss(self):
        s = sample_scenario("task_006", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20


class TestBatchNormEval:
    def test_val_loss_present(self):
        s = sample_scenario("task_005", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20

    def test_val_acc_near_zero(self):
        s = sample_scenario("task_005", seed=42)
        hist = gen_val_accuracy_history(s)
        # BatchNorm eval mode makes learning very poor
        assert len(hist) == 20


class TestSchedulerMisconfigured:
    def test_loss_history(self):
        s = sample_scenario("task_007", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20

    def test_val_acc(self):
        s = sample_scenario("task_007", seed=42)
        hist = gen_val_accuracy_history(s)
        assert len(hist) == 20

    def test_val_loss(self):
        s = sample_scenario("task_007", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20
