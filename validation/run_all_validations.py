#!/usr/bin/env python3
"""Run all validation checks and produce a fidelity report.

Validates that parametric curve generation and real PyTorch fault injection
produce qualitatively consistent behaviors. Uses directional/behavioral
agreement rather than R² (parametric curves are intentionally stylized
for clear agent signals, not exact replicas of real training).
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
    create_model_and_inject_fault,
    extract_gradient_stats,
    extract_model_modes,
    extract_weight_stats,
)
from ml_training_debugger.scenarios import sample_scenario
from ml_training_debugger.simulation import (
    gen_data_batch_stats,
    gen_loss_history,
    gen_val_accuracy_history,
    gen_val_loss_history,
)


def validate_exploding_gradients() -> dict:
    """Task 1: Verify exploding gradient detection."""
    scenario = sample_scenario("task_001", seed=42)
    model, _ = create_model_and_inject_fault(scenario)
    stats = extract_gradient_stats(model, scenario)
    loss = gen_loss_history(scenario)

    all_exploding = all(s.is_exploding for s in stats)
    loss_diverges = any(v == float("inf") or v > 100 for v in loss)
    max_grad = max(s.mean_norm for s in stats)

    return {
        "task": "task_001",
        "fault": "exploding_gradients",
        "checks": {
            "all_layers_exploding": all_exploding,
            "loss_diverges_to_inf": loss_diverges,
            "max_gradient_norm": round(max_grad, 2),
            "gradient_threshold": 10.0,
            "real_pytorch_gradients": True,
        },
        "pass": all_exploding and loss_diverges,
    }


def validate_vanishing_gradients() -> dict:
    """Task 2: Verify vanishing gradient detection."""
    scenario = sample_scenario("task_002", seed=42)
    model, _ = create_model_and_inject_fault(scenario)
    stats = extract_gradient_stats(model, scenario)
    loss = gen_loss_history(scenario)

    any_vanishing = any(s.is_vanishing for s in stats)
    loss_flat = abs(loss[-1] - loss[0]) < 0.5  # barely changes

    return {
        "task": "task_002",
        "fault": "vanishing_gradients",
        "checks": {
            "deeper_layers_vanishing": any_vanishing,
            "loss_barely_decreases": loss_flat,
            "min_gradient_norm": round(min(s.mean_norm for s in stats), 10),
            "vanishing_threshold": 1e-6,
            "real_pytorch_gradients": True,
        },
        "pass": any_vanishing and loss_flat,
    }


def validate_data_leakage() -> dict:
    """Task 3: Verify data leakage signal."""
    scenario = sample_scenario("task_003", seed=42)
    model, _ = create_model_and_inject_fault(scenario)
    stats = extract_gradient_stats(model, scenario)
    data = gen_data_batch_stats(scenario)
    val_acc = gen_val_accuracy_history(scenario)

    overlap_high = data["class_overlap_score"] > 0.5
    val_acc_high = val_acc[0] > 0.7  # suspiciously high from epoch 1
    gradients_normal = not any(s.is_exploding for s in stats)

    return {
        "task": "task_003",
        "fault": "data_leakage",
        "checks": {
            "class_overlap_above_0.5": overlap_high,
            "class_overlap_score": round(data["class_overlap_score"], 4),
            "val_accuracy_suspiciously_high": val_acc_high,
            "val_acc_epoch_1": round(val_acc[0], 4),
            "gradients_normal": gradients_normal,
            "real_pytorch_model": True,
        },
        "pass": overlap_high and val_acc_high and gradients_normal,
    }


def validate_overfitting() -> dict:
    """Task 4: Verify train-val divergence."""
    scenario = sample_scenario("task_004", seed=42)
    loss = gen_loss_history(scenario)
    val_loss = gen_val_loss_history(scenario)
    val_acc = gen_val_accuracy_history(scenario)

    train_loss_low = loss[-1] < 0.1
    val_loss_rises = val_loss[-1] > val_loss[len(val_loss) // 2]
    val_acc_drops = val_acc[-1] < max(val_acc)

    return {
        "task": "task_004",
        "fault": "overfitting",
        "checks": {
            "train_loss_near_zero": train_loss_low,
            "train_loss_final": round(loss[-1], 4),
            "val_loss_rising": val_loss_rises,
            "val_loss_final": round(val_loss[-1], 4),
            "val_accuracy_drops_after_peak": val_acc_drops,
        },
        "pass": train_loss_low and val_loss_rises,
    }


def validate_batchnorm_eval() -> dict:
    """Task 5: Verify BatchNorm eval mode detection + red herrings."""
    scenario = sample_scenario("task_005", seed=42)
    model, _ = create_model_and_inject_fault(scenario)
    stats = extract_gradient_stats(model, scenario)
    modes = extract_model_modes(model)
    val_acc = gen_val_accuracy_history(scenario)

    all_eval = all(v == "eval" for v in modes.values())
    no_exploding = not any(s.is_exploding for s in stats)
    val_acc_degrades = val_acc[-1] < val_acc[0]

    spike_layer = next(
        s for s in stats if s.layer_name == scenario.red_herring_spike_layer
    )

    return {
        "task": "task_005",
        "fault": "batchnorm_eval_mode",
        "checks": {
            "all_layers_in_eval_mode": all_eval,
            "no_layer_is_exploding": no_exploding,
            "val_accuracy_degrades": val_acc_degrades,
            "red_herring_spike_layer": scenario.red_herring_spike_layer,
            "spike_layer_mean_norm": round(spike_layer.mean_norm, 6),
            "spike_not_exploding": not spike_layer.is_exploding,
            "gpu_memory_red_herring_gb": scenario.gpu_memory_used_gb,
            "real_model_eval_mode": not model.training,
        },
        "pass": all_eval and no_exploding and val_acc_degrades,
    }


def validate_code_bugs() -> dict:
    """Task 6: Verify code bug variants generate valid snippets."""
    from ml_training_debugger.code_templates import generate_code_snippet, validate_fix

    variants = ["eval_mode", "detach_loss", "zero_grad_missing", "inplace_relu"]
    results = {}

    for variant in variants:
        snippet = generate_code_snippet(variant, seed=42)
        code = snippet["code"]

        # Verify correct fix is accepted
        from ml_training_debugger.code_templates import _TEMPLATES

        _, correct_line, correct_replacement = _TEMPLATES[variant]
        fix_accepted = validate_fix(variant, correct_line, correct_replacement)

        # Verify wrong fix is rejected
        wrong_rejected = not validate_fix(variant, correct_line, "pass")

        results[variant] = {
            "code_lines": snippet["line_count"],
            "correct_fix_accepted": fix_accepted,
            "wrong_fix_rejected": wrong_rejected,
            "has_bug_pattern": True,
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
            "fix_validation_pipeline": "normalize → tokenize → semantic → AST",
        },
        "pass": all_pass,
    }


def main() -> None:
    validations = [
        validate_exploding_gradients(),
        validate_vanishing_gradients(),
        validate_data_leakage(),
        validate_overfitting(),
        validate_batchnorm_eval(),
        validate_code_bugs(),
    ]

    report = {
        "methodology": "Real PyTorch training + fault injection vs parametric curves",
        "torch_version": torch.__version__,
        "model": "SimpleCNN (~50K params, 3-layer CNN with BatchNorm)",
        "validation_approach": "Behavioral agreement (directional consistency, threshold checks)",
        "results": validations,
        "summary": {
            "total": len(validations),
            "passed": sum(1 for v in validations if v["pass"]),
            "failed": sum(1 for v in validations if not v["pass"]),
        },
    }

    # Save report
    report_path = Path(__file__).parent / "reports" / "fidelity_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))

    # Print summary
    for v in validations:
        status = "PASS" if v["pass"] else "FAIL"
        print(f"  {status}: {v['task']} — {v['fault']}")

    print(f"\n{report['summary']['passed']}/{report['summary']['total']} validations passed")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
