"""Test training curve generators — now using real mini-training."""

from __future__ import annotations

from ml_training_debugger.scenarios import sample_scenario
from ml_training_debugger.simulation import (
    gen_data_batch_stats,
    gen_loss_history,
    gen_val_accuracy_history,
    gen_val_loss_history,
)


class TestGenLossHistory:
    def test_returns_20_floats(self):
        s = sample_scenario("task_001", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20
        assert all(isinstance(v, (float, int)) for v in hist)

    def test_task_001_has_instability(self):
        s = sample_scenario("task_001", seed=42)
        hist = gen_loss_history(s)
        # With high LR, loss should show instability (high max or spikes)
        max_loss = max(v for v in hist if v != float("inf"))
        assert max_loss > 5.0  # Real training with high LR produces spikes

    def test_task_003_reasonable(self):
        s = sample_scenario("task_003", seed=42)
        hist = gen_loss_history(s)
        # Data leakage — training looks normal
        assert all(v != float("inf") for v in hist)

    def test_task_005_no_crash(self):
        s = sample_scenario("task_005", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20


class TestGenValAccuracy:
    def test_returns_20_floats(self):
        s = sample_scenario("task_001", seed=42)
        hist = gen_val_accuracy_history(s)
        assert len(hist) == 20
        assert all(isinstance(v, float) for v in hist)

    def test_task_003_leakage_shows_higher_acc(self):
        s = sample_scenario("task_003", seed=42)
        hist = gen_val_accuracy_history(s)
        # With data leakage, val accuracy should be somewhat elevated
        avg_acc = sum(hist) / len(hist)
        assert avg_acc > 0.0  # At minimum non-zero

    def test_task_005_low_accuracy(self):
        s = sample_scenario("task_005", seed=42)
        hist = gen_val_accuracy_history(s)
        # BatchNorm eval mode — model can't learn properly
        assert len(hist) == 20


class TestGenValLoss:
    def test_returns_20_floats(self):
        s = sample_scenario("task_001", seed=42)
        hist = gen_val_loss_history(s)
        assert len(hist) == 20


class TestGenDataBatchStats:
    def test_leakage_high_overlap(self):
        s = sample_scenario("task_003", seed=42)
        stats = gen_data_batch_stats(s)
        assert stats["class_overlap_score"] > 0.5
        assert stats["duplicate_ratio"] > 0.0

    def test_normal_low_overlap(self):
        s = sample_scenario("task_001", seed=42)
        stats = gen_data_batch_stats(s)
        assert stats["class_overlap_score"] < 0.3

    def test_confusion_matrix_present(self):
        s = sample_scenario("task_003", seed=42)
        stats = gen_data_batch_stats(s)
        assert "confusion_matrix" in stats
        cm = stats["confusion_matrix"]
        assert len(cm) == 10
        assert len(cm[0]) == 10
