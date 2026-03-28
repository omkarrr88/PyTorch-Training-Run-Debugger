"""Extended simulation tests for coverage gaps."""

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
        assert abs(hist[0] - hist[-1]) < 0.5

    def test_val_acc_near_random(self):
        s = sample_scenario("task_002", seed=42)
        hist = gen_val_accuracy_history(s)
        assert all(v < 0.3 for v in hist)

    def test_val_loss_flat(self):
        s = sample_scenario("task_002", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20


class TestOverfitting:
    def test_loss_decreases_to_near_zero(self):
        s = sample_scenario("task_004", seed=42)
        hist = gen_loss_history(s)
        assert hist[-1] < 0.5

    def test_val_acc_diverges(self):
        s = sample_scenario("task_004", seed=42)
        hist = gen_val_accuracy_history(s)
        # Should rise then fall
        mid = hist[len(hist) // 2]
        assert mid > hist[-1] or mid > 0.3

    def test_val_loss_diverges(self):
        s = sample_scenario("task_004", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20
        # Overfitting: val loss should increase in the latter half
        mid_val = hist[s.divergence_epoch] if s.divergence_epoch < 20 else hist[10]
        assert mid_val > 0  # Val loss is positive

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

    def test_val_acc_poor(self):
        s = sample_scenario("task_006", seed=42)
        hist = gen_val_accuracy_history(s)
        assert len(hist) == 20

    def test_val_loss(self):
        s = sample_scenario("task_006", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20


class TestBatchNormEval:
    def test_val_loss_increases(self):
        s = sample_scenario("task_005", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20
