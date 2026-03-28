"""Test parametric curve generators."""

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
        assert all(isinstance(v, float) for v in hist)

    def test_task_001_diverges(self):
        s = sample_scenario("task_001", seed=42)
        hist = gen_loss_history(s)
        assert hist[-1] == float("inf")  # NaN/inf after epoch 12

    def test_task_003_normal(self):
        s = sample_scenario("task_003", seed=42)
        hist = gen_loss_history(s)
        assert hist[0] > hist[-1]  # Loss decreases

    def test_task_005_higher_variance(self):
        s = sample_scenario("task_005", seed=42)
        hist = gen_loss_history(s)
        assert len(hist) == 20


class TestGenValAccuracy:
    def test_returns_20_floats(self):
        s = sample_scenario("task_001", seed=42)
        hist = gen_val_accuracy_history(s)
        assert len(hist) == 20
        assert all(isinstance(v, float) for v in hist)

    def test_task_003_suspiciously_high(self):
        s = sample_scenario("task_003", seed=42)
        hist = gen_val_accuracy_history(s)
        assert hist[1] > 0.80  # Suspiciously high from early epochs

    def test_task_005_degrades(self):
        s = sample_scenario("task_005", seed=42)
        hist = gen_val_accuracy_history(s)
        assert hist[0] > hist[-1]  # Degrades over time


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
