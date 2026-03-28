"""Tests for real mini-training in pytorch_engine.py."""

from __future__ import annotations

import torch

from ml_training_debugger.pytorch_engine import (
    SimpleCNN,
    SimpleMLP,
    _TRAINING_CACHE,
    run_real_training,
)
from ml_training_debugger.scenarios import sample_scenario


class TestRunRealTraining:
    def test_returns_20_epoch_curves(self) -> None:
        s = sample_scenario("task_001", seed=42)
        curves = run_real_training(s)
        assert len(curves["loss_history"]) == 20
        assert len(curves["val_loss_history"]) == 20
        assert len(curves["val_acc_history"]) == 20

    def test_all_values_are_floats(self) -> None:
        s = sample_scenario("task_003", seed=42)
        curves = run_real_training(s)
        for key in ["loss_history", "val_loss_history", "val_acc_history"]:
            for v in curves[key]:
                assert isinstance(v, float), f"{key} has non-float: {type(v)}"

    def test_caching_works(self) -> None:
        _TRAINING_CACHE.clear()
        s = sample_scenario("task_001", seed=42)
        c1 = run_real_training(s)
        c2 = run_real_training(s)
        assert c1 is c2  # Same object reference = cached

    def test_reproducible_across_calls(self) -> None:
        _TRAINING_CACHE.clear()
        s = sample_scenario("task_002", seed=42)
        c1 = run_real_training(s)
        _TRAINING_CACHE.clear()
        c2 = run_real_training(s)
        assert c1["loss_history"] == c2["loss_history"]
        assert c1["val_acc_history"] == c2["val_acc_history"]

    def test_different_seeds_different_curves(self) -> None:
        s1 = sample_scenario("task_001", seed=42)
        s2 = sample_scenario("task_001", seed=99)
        c1 = run_real_training(s1)
        c2 = run_real_training(s2)
        assert c1["loss_history"] != c2["loss_history"]

    def test_task_001_high_lr_instability(self) -> None:
        s = sample_scenario("task_001", seed=42)
        curves = run_real_training(s)
        max_loss = max(v for v in curves["loss_history"] if v != float("inf"))
        assert max_loss > 3.0  # High LR causes loss spikes

    def test_task_002_vanishing_slow_learning(self) -> None:
        s = sample_scenario("task_002", seed=42)
        curves = run_real_training(s)
        assert len(curves["loss_history"]) == 20

    def test_task_003_data_leakage(self) -> None:
        s = sample_scenario("task_003", seed=42)
        curves = run_real_training(s)
        # With leakage, val accuracy may be elevated
        assert len(curves["val_acc_history"]) == 20

    def test_task_004_overfitting(self) -> None:
        s = sample_scenario("task_004", seed=42)
        curves = run_real_training(s)
        assert len(curves["loss_history"]) == 20

    def test_task_005_batchnorm_eval(self) -> None:
        s = sample_scenario("task_005", seed=42)
        curves = run_real_training(s)
        assert len(curves["loss_history"]) == 20

    def test_task_006_code_bug(self) -> None:
        s = sample_scenario("task_006", seed=42)
        curves = run_real_training(s)
        assert len(curves["loss_history"]) == 20

    def test_task_007_scheduler(self) -> None:
        s = sample_scenario("task_007", seed=42)
        curves = run_real_training(s)
        assert len(curves["loss_history"]) == 20

    def test_mlp_architecture(self) -> None:
        """Find a scenario that uses MLP and verify training works."""
        for seed in range(1, 20):
            s = sample_scenario("task_001", seed=seed)
            if s.model_type == "mlp":
                curves = run_real_training(s)
                assert len(curves["loss_history"]) == 20
                return
        # If no MLP found in 20 seeds, test directly
        from ml_training_debugger.scenarios import ScenarioParams
        from ml_training_debugger.models import RootCauseDiagnosis
        s = ScenarioParams(
            task_id="task_001",
            root_cause=RootCauseDiagnosis.LR_TOO_HIGH,
            seed=999,
            learning_rate=0.1,
            model_type="mlp",
        )
        curves = run_real_training(s)
        assert len(curves["loss_history"]) == 20


class TestSimpleMLP:
    def test_is_nn_module(self) -> None:
        model = SimpleMLP()
        assert isinstance(model, torch.nn.Module)

    def test_param_count(self) -> None:
        model = SimpleMLP()
        count = sum(p.numel() for p in model.parameters())
        assert 10_000 < count < 500_000

    def test_forward_pass(self) -> None:
        model = SimpleMLP()
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_has_batchnorm(self) -> None:
        model = SimpleMLP()
        has_bn = any(
            isinstance(m, torch.nn.BatchNorm1d)
            for m in model.modules()
        )
        assert has_bn
