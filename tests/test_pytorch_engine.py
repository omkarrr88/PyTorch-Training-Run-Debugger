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


class TestTask005RedHerrings:
    """Test Task 5 red herring injection — conv1 near-vanishing, FC spike."""

    def test_conv1_near_vanishing_red_herring(self):
        """When spike layer is fc, conv1 should show near-vanishing gradient."""
        scenario = sample_scenario("task_005", seed=42)
        model, _ = create_model_and_inject_fault(scenario)
        stats = extract_gradient_stats(model, scenario)

        conv1 = next(s for s in stats if s.layer_name == "conv1")
        if scenario.red_herring_spike_layer != "conv1":
            # conv1 should be near-vanishing (but not is_vanishing since 0.0003 > 1e-6)
            assert conv1.mean_norm < 0.01
            assert not conv1.is_vanishing  # 0.0003 > 1e-6

    def test_fc_spike_not_exploding(self):
        """FC spike has elevated gradient but is_exploding=False (mean < 10.0)."""
        scenario = sample_scenario("task_005", seed=42)
        model, _ = create_model_and_inject_fault(scenario)
        stats = extract_gradient_stats(model, scenario)

        spike_layer = next(
            s for s in stats if s.layer_name == scenario.red_herring_spike_layer
        )
        assert not spike_layer.is_exploding
        # Should have non-trivial norm from the spike
        assert spike_layer.mean_norm > 0

    def test_all_layers_not_exploding(self):
        """All layers is_exploding=False — this gates gradients_were_normal."""
        scenario = sample_scenario("task_005", seed=42)
        model, _ = create_model_and_inject_fault(scenario)
        stats = extract_gradient_stats(model, scenario)
        for s in stats:
            assert not s.is_exploding, f"{s.layer_name} should not be exploding"


class TestVanishingGradientInjection:
    """Test vanishing gradient fault injection produces correct stats."""

    def test_task_002_vanishing(self):
        scenario = sample_scenario("task_002", seed=42)
        model, _ = create_model_and_inject_fault(scenario)
        stats = extract_gradient_stats(model, scenario)
        # Deeper layers should have vanishing gradients
        assert any(s.is_vanishing for s in stats)

    def test_task_002_model_in_train_mode(self):
        scenario = sample_scenario("task_002", seed=42)
        model, _ = create_model_and_inject_fault(scenario)
        assert model.training


class TestCodeBugFaultInjection:
    """Test code bug fault injection — model should be normal."""

    def test_task_006_model_trains_normally(self):
        scenario = sample_scenario("task_006", seed=42)
        model, _ = create_model_and_inject_fault(scenario)
        assert model.training  # Should be in train mode
        stats = extract_gradient_stats(model, scenario)
        # No exploding/vanishing — bug is in code only
        assert not any(s.is_exploding for s in stats)


class TestDataLeakageFaultInjection:
    """Test data leakage scenario — model should be normal."""

    def test_task_003_normal_model(self):
        scenario = sample_scenario("task_003", seed=42)
        model, _ = create_model_and_inject_fault(scenario)
        assert model.training
        stats = extract_gradient_stats(model, scenario)
        assert not any(s.is_exploding for s in stats)
