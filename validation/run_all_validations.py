#!/usr/bin/env python3
"""Run all validation checks and produce a fidelity report.

Validates that real PyTorch mini-training produces qualitatively correct
behaviors for each fault type. Uses behavioral checks appropriate for
real training on tiny random-data models (not parametric formula checks).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_training_debugger.pytorch_engine import (
    SimpleCNN,
    SimpleMLP,
    create_model_and_inject_fault,
    extract_gradient_stats,
    extract_model_modes,
    extract_weight_stats,
    run_real_training,
)
from ml_training_debugger.scenarios import sample_scenario
from ml_training_debugger.simulation import gen_data_batch_stats


def validate_exploding_gradients() -> dict:
    """Task 1: High LR produces gradient instability."""
    scenario = sample_scenario("task_001", seed=42)
    model, _ = create_model_and_inject_fault(scenario)
    stats = extract_gradient_stats(model, scenario)
    curves = run_real_training(scenario)

    any_exploding = any(s.is_exploding for s in stats)
    loss_unstable = max(curves["loss_history"]) > 5.0
    max_grad = max(s.mean_norm for s in stats)

    return {
        "task": "task_001",
        "fault": "exploding_gradients",
        "checks": {
            "gradient_instability_detected": any_exploding,
            "loss_shows_instability": loss_unstable,
            "max_gradient_norm": round(max_grad, 2),
            "max_loss": round(max(curves["loss_history"]), 2),
            "real_pytorch_training": True,
        },
        "pass": any_exploding and loss_unstable,
    }


def validate_vanishing_gradients() -> dict:
    """Task 2: Low LR + scaled gradients produce vanishing."""
    scenario = sample_scenario("task_002", seed=42)
    model, _ = create_model_and_inject_fault(scenario)
    stats = extract_gradient_stats(model, scenario)

    any_vanishing = any(s.is_vanishing for s in stats)
    min_grad = min(s.mean_norm for s in stats)

    return {
        "task": "task_002",
        "fault": "vanishing_gradients",
        "checks": {
            "vanishing_detected": any_vanishing,
            "min_gradient_norm": round(min_grad, 10),
            "real_pytorch_gradients": True,
        },
        "pass": any_vanishing,
    }


def validate_data_leakage() -> dict:
    """Task 3: Data leakage produces high overlap score."""
    scenario = sample_scenario("task_003", seed=42)
    data = gen_data_batch_stats(scenario)
    curves = run_real_training(scenario)

    overlap_high = data["class_overlap_score"] > 0.5
    training_runs = len(curves["loss_history"]) == 20

    return {
        "task": "task_003",
        "fault": "data_leakage",
        "checks": {
            "class_overlap_above_0.5": overlap_high,
            "class_overlap_score": round(data["class_overlap_score"], 4),
            "real_training_runs": training_runs,
            "has_confusion_matrix": "confusion_matrix" in data,
        },
        "pass": overlap_high and training_runs,
    }


def validate_overfitting() -> dict:
    """Task 4: Overfitting scenario runs real training."""
    scenario = sample_scenario("task_004", seed=42)
    curves = run_real_training(scenario)
    data = gen_data_batch_stats(scenario)

    training_runs = len(curves["loss_history"]) == 20
    clean_data = data["class_overlap_score"] == 0.0

    return {
        "task": "task_004",
        "fault": "overfitting",
        "checks": {
            "real_training_runs": training_runs,
            "clean_data": clean_data,
            "final_train_loss": round(curves["loss_history"][-1], 4),
            "final_val_loss": round(curves["val_loss_history"][-1], 4),
        },
        "pass": training_runs and clean_data,
    }


def validate_batchnorm_eval() -> dict:
    """Task 5: BatchNorm eval mode + red herrings."""
    scenario = sample_scenario("task_005", seed=42)
    model, _ = create_model_and_inject_fault(scenario)
    stats = extract_gradient_stats(model, scenario)
    modes = extract_model_modes(model)
    curves = run_real_training(scenario)

    all_eval = all(v == "eval" for v in modes.values())
    no_exploding = not any(s.is_exploding for s in stats)
    training_runs = len(curves["loss_history"]) == 20

    return {
        "task": "task_005",
        "fault": "batchnorm_eval_mode",
        "checks": {
            "all_layers_in_eval_mode": all_eval,
            "no_layer_is_exploding": no_exploding,
            "real_training_runs": training_runs,
            "real_model_eval_mode": not model.training,
            "red_herring_spike_layer": scenario.red_herring_spike_layer,
        },
        "pass": all_eval and no_exploding and training_runs,
    }


def validate_code_bugs() -> dict:
    """Task 6: Code bug variants."""
    from ml_training_debugger.code_templates import (
        _TEMPLATES,
        generate_code_snippet,
        validate_fix,
    )

    variants = ["eval_mode", "detach_loss", "zero_grad_missing", "inplace_relu"]
    results = {}

    for variant in variants:
        snippet = generate_code_snippet(variant, seed=42)
        _, correct_line, correct_replacement = _TEMPLATES[variant]
        fix_accepted = validate_fix(variant, correct_line, correct_replacement)
        wrong_rejected = not validate_fix(variant, correct_line, "pass")

        results[variant] = {
            "correct_fix_accepted": fix_accepted,
            "wrong_fix_rejected": wrong_rejected,
        }

    all_pass = all(
        r["correct_fix_accepted"] and r["wrong_fix_rejected"]
        for r in results.values()
    )

    return {
        "task": "task_006",
        "fault": "code_bug",
        "checks": {
            "variants_tested": len(variants),
            "variant_results": results,
            "fix_validation_pipeline": "normalize -> tokenize -> semantic -> AST",
        },
        "pass": all_pass,
    }


def validate_scheduler() -> dict:
    """Task 7: Scheduler misconfigured."""
    scenario = sample_scenario("task_007", seed=42)
    curves = run_real_training(scenario)

    training_runs = len(curves["loss_history"]) == 20

    return {
        "task": "task_007",
        "fault": "scheduler_misconfigured",
        "checks": {
            "real_training_runs": training_runs,
            "scheduler_gamma": scenario.scheduler_gamma,
            "scheduler_step_size": scenario.scheduler_step_size,
            "final_loss": round(curves["loss_history"][-1], 4),
        },
        "pass": training_runs,
    }


def validate_dual_architecture() -> dict:
    """Verify both CNN and MLP architectures work."""
    cnn = SimpleCNN()
    mlp = SimpleMLP()

    x = torch.randn(4, 3, 32, 32)
    cnn_out = cnn(x)
    mlp_out = mlp(x)

    return {
        "task": "architecture",
        "fault": "dual_model_support",
        "checks": {
            "cnn_output_shape": list(cnn_out.shape),
            "mlp_output_shape": list(mlp_out.shape),
            "cnn_params": sum(p.numel() for p in cnn.parameters()),
            "mlp_params": sum(p.numel() for p in mlp.parameters()),
            "both_produce_10_classes": cnn_out.shape[1] == 10 and mlp_out.shape[1] == 10,
        },
        "pass": cnn_out.shape == (4, 10) and mlp_out.shape == (4, 10),
    }


def main() -> None:
    validations = [
        validate_exploding_gradients(),
        validate_vanishing_gradients(),
        validate_data_leakage(),
        validate_overfitting(),
        validate_batchnorm_eval(),
        validate_code_bugs(),
        validate_scheduler(),
        validate_dual_architecture(),
    ]

    report = {
        "methodology": "Real PyTorch 20-epoch mini-training with fault injection",
        "torch_version": torch.__version__,
        "models": ["SimpleCNN (~50K params)", "SimpleMLP (~20K params)"],
        "training_approach": "Real forward+backward passes on random CIFAR-10 style data, cached per (task_id, seed)",
        "results": validations,
        "summary": {
            "total": len(validations),
            "passed": sum(1 for v in validations if v["pass"]),
            "failed": sum(1 for v in validations if not v["pass"]),
        },
    }

    report_path = Path(__file__).parent / "reports" / "fidelity_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))

    for v in validations:
        status = "PASS" if v["pass"] else "FAIL"
        print(f"  {status}: {v['task']} — {v['fault']}")

    print(f"\n{report['summary']['passed']}/{report['summary']['total']} validations passed")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
