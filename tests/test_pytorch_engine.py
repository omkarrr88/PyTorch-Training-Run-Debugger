"""Test real PyTorch model instantiation and fault injection."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml_training_debugger.pytorch_engine import (
    SimpleCNN,
    create_model_and_inject_fault,
    extract_gradient_stats,
    extract_model_modes,
    extract_weight_stats,
)
from ml_training_debugger.scenarios import sample_scenario


class TestSimpleCNN:
    def test_is_nn_module(self):
        model = SimpleCNN()
        assert isinstance(model, nn.Module)

    def test_param_count(self):
        model = SimpleCNN()
        count = sum(p.numel() for p in model.parameters())
        assert 30_000 < count < 100_000  # ~50K params

    def test_forward_pass(self):
        model = SimpleCNN()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)


class TestFaultInjection:
    def test_task_001_exploding_gradients(self):
        scenario = sample_scenario("task_001", seed=42)
        model, info = create_model_and_inject_fault(scenario)
        stats = extract_gradient_stats(model, scenario)
        assert len(stats) > 0
        # At least some layers should have elevated gradients
        any_high = any(s.mean_norm > 1.0 for s in stats)
        assert any_high

    def test_task_005_eval_mode(self):
        scenario = sample_scenario("task_005", seed=42)
        model, info = create_model_and_inject_fault(scenario)
        assert not model.training  # model.eval() was called

    def test_task_005_gradients_not_exploding(self):
        scenario = sample_scenario("task_005", seed=42)
        model, info = create_model_and_inject_fault(scenario)
        stats = extract_gradient_stats(model, scenario)
        # ALL layers must have is_exploding=False
        for s in stats:
            assert not s.is_exploding, f"Layer {s.layer_name} should not be exploding"


class TestExtractGradientStats:
    def test_returns_gradient_stats(self):
        scenario = sample_scenario("task_001", seed=42)
        model, _ = create_model_and_inject_fault(scenario)
        stats = extract_gradient_stats(model, scenario)
        assert len(stats) == 4  # conv1, conv2, conv3, fc
        for s in stats:
            assert isinstance(s.mean_norm, float)
            assert isinstance(s.norm_history, list)
            assert len(s.norm_history) == 5


class TestExtractWeightStats:
    def test_returns_weight_stats(self):
        scenario = sample_scenario("task_001", seed=42)
        model, _ = create_model_and_inject_fault(scenario)
        stats = extract_weight_stats(model)
        assert len(stats) > 0
        for s in stats:
            assert isinstance(s.weight_norm, float)
            assert isinstance(s.has_nan, bool)


class TestExtractModelModes:
    def test_train_mode(self):
        model = SimpleCNN()
        model.train()
        modes = extract_model_modes(model)
        assert all(v == "train" for v in modes.values())

    def test_eval_mode(self):
        model = SimpleCNN()
        model.eval()
        modes = extract_model_modes(model)
        assert all(v == "eval" for v in modes.values())
