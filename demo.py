#!/usr/bin/env python3
"""Demo script — runs the heuristic baseline against all 7 tasks and prints results.

No API key required. Deterministic, bit-exact reproducible.

Usage:
    python demo.py
"""

from __future__ import annotations

import json
import sys
import time

from ml_training_debugger.models import MLTrainingAction
from server.environment import MLTrainingEnvironment

ALL_TASKS = [
    ("task_001", "Exploding Gradients", "easy"),
    ("task_002", "Vanishing Gradients", "easy"),
    ("task_003", "Data Leakage", "medium"),
    ("task_004", "Overfitting", "medium"),
    ("task_005", "BatchNorm Eval Mode", "hard"),
    ("task_006", "Code Bug", "hard"),
    ("task_007", "Scheduler Misconfigured", "medium-hard"),
]


def run_demo_episode(task_id: str, seed: int = 42) -> tuple[float, list[str], float]:
    """Run one heuristic episode. Returns (score, actions_taken, elapsed_seconds)."""
    env = MLTrainingEnvironment()
    start = time.time()
    obs = env.reset(seed=seed, episode_id=f"demo_{task_id}", task_id=task_id)

    # Step 1: inspect_gradients
    obs = env.step(MLTrainingAction(action_type="inspect_gradients"))

    if obs.gradient_stats:
        if any(g.is_exploding for g in obs.gradient_stats):
            env.step(MLTrainingAction(action_type="modify_config", target="learning_rate", value=0.001))
            env.step(MLTrainingAction(action_type="restart_run"))
            env.step(MLTrainingAction(action_type="mark_diagnosed", diagnosis="lr_too_high"))
            session = env._get_session()
            return (session.last_score or 0.0, session.state.actions_taken, time.time() - start)

        if any(g.is_vanishing for g in obs.gradient_stats):
            env.step(MLTrainingAction(action_type="modify_config", target="learning_rate", value=0.01))
            env.step(MLTrainingAction(action_type="restart_run"))
            env.step(MLTrainingAction(action_type="mark_diagnosed", diagnosis="vanishing_gradients"))
            session = env._get_session()
            return (session.last_score or 0.0, session.state.actions_taken, time.time() - start)

    # Step 2: inspect_data_batch
    obs = env.step(MLTrainingAction(action_type="inspect_data_batch"))
    if obs.data_batch_stats and obs.data_batch_stats.class_overlap_score > 0.5:
        env.step(MLTrainingAction(action_type="patch_data_loader"))
        env.step(MLTrainingAction(action_type="restart_run"))
        env.step(MLTrainingAction(action_type="mark_diagnosed", diagnosis="data_leakage"))
        session = env._get_session()
        return (session.last_score or 0.0, session.state.actions_taken, time.time() - start)

    # Detect overfitting pattern (defer action until after code check)
    _looks_like_overfitting = False
    if obs.val_loss_history and obs.training_loss_history and len(obs.val_loss_history) >= 10:
        early_train = sum(obs.training_loss_history[:5]) / 5
        late_train = sum(obs.training_loss_history[-5:]) / 5
        early_val = sum(obs.val_loss_history[:5]) / 5
        late_val = sum(obs.val_loss_history[-5:]) / 5
        train_dropped = late_train < early_train * 0.5
        train_loss_low = late_train < 0.15
        val_not_improving = late_val >= early_val * 0.95
        gap_widening = (late_val - late_train) > (early_val - early_train)
        if (train_dropped or train_loss_low) and (val_not_improving or gap_widening):
            if obs.data_batch_stats and obs.data_batch_stats.class_overlap_score < 0.3:
                _looks_like_overfitting = True

    # Step 3: inspect_model_modes
    obs = env.step(MLTrainingAction(action_type="inspect_model_modes"))
    if obs.model_mode_info and any(v == "eval" for v in obs.model_mode_info.values()):
        env.step(MLTrainingAction(action_type="fix_model_mode"))
        env.step(MLTrainingAction(action_type="restart_run"))
        env.step(MLTrainingAction(action_type="mark_diagnosed", diagnosis="batchnorm_eval_mode"))
        session = env._get_session()
        return (session.last_score or 0.0, session.state.actions_taken, time.time() - start)

    # Step 4: inspect_code
    obs = env.step(MLTrainingAction(action_type="inspect_code"))
    if obs.code_snippet:
        code = obs.code_snippet.code
        fixed = False
        if "model.eval()" in code and "model.train()" not in code:
            env.step(MLTrainingAction(action_type="fix_code", line=5, replacement="model.train()"))
            fixed = True
        elif ".detach()" in code:
            env.step(MLTrainingAction(action_type="fix_code", line=14, replacement="        loss = criterion(output, batch_y)"))
            fixed = True

        if fixed:
            env.step(MLTrainingAction(action_type="restart_run"))
        env.step(MLTrainingAction(action_type="mark_diagnosed", diagnosis="code_bug"))
        session = env._get_session()
        return (session.last_score or 0.0, session.state.actions_taken, time.time() - start)

    # Step 5: scheduler check
    if obs.training_loss_history and len(obs.training_loss_history) >= 10:
        early_loss = sum(obs.training_loss_history[:3]) / 3
        mid_loss = sum(obs.training_loss_history[5:8]) / 3
        finite_late = [v for v in obs.training_loss_history[-3:] if v != float("inf")]
        late_loss = sum(finite_late) / max(len(finite_late), 1)
        if early_loss > mid_loss and abs(late_loss - mid_loss) < 0.3:
            env.step(MLTrainingAction(action_type="modify_config", target="learning_rate", value=0.001))
            env.step(MLTrainingAction(action_type="restart_run"))
            env.step(MLTrainingAction(action_type="mark_diagnosed", diagnosis="scheduler_misconfigured"))
            session = env._get_session()
            return (session.last_score or 0.0, session.state.actions_taken, time.time() - start)

    # Overfitting fallback
    if _looks_like_overfitting:
        env.step(MLTrainingAction(action_type="modify_config", target="weight_decay", value=0.01))
        env.step(MLTrainingAction(action_type="restart_run"))
        env.step(MLTrainingAction(action_type="mark_diagnosed", diagnosis="overfitting"))
        session = env._get_session()
        return (session.last_score or 0.0, session.state.actions_taken, time.time() - start)

    # Final fallback
    env.step(MLTrainingAction(action_type="mark_diagnosed", diagnosis="overfitting"))
    session = env._get_session()
    return (session.last_score or 0.0, session.state.actions_taken, time.time() - start)


def main() -> None:
    print("=" * 70)
    print("PyTorch Training Run Debugger — Demo")
    print("Running heuristic baseline on all 7 tasks")
    print("=" * 70)
    print()

    total_score = 0.0
    results = []

    for task_id, name, difficulty in ALL_TASKS:
        score, actions, elapsed = run_demo_episode(task_id)
        total_score += score
        results.append((task_id, name, difficulty, score, actions, elapsed))

        diagnosis = "none"
        for a in actions:
            if a.startswith("mark_diagnosed:"):
                diagnosis = a.split(":")[1]

        print(f"  {task_id} | {name:<25} | {difficulty:<10} | score={score:.2f} | {len(actions)} steps | {elapsed:.2f}s")
        print(f"           actions: {' -> '.join(a.replace('mark_diagnosed:', 'diag:') for a in actions)}")
        print()

    avg = total_score / len(ALL_TASKS)
    print("-" * 70)
    print(f"  Average score: {avg:.2f}")
    print(f"  Tasks solved:  {sum(1 for _, _, _, s, _, _ in results if s >= 0.95)}/{len(ALL_TASKS)}")
    print()

    # Also output machine-readable JSON
    scores = {task_id: round(score, 4) for task_id, _, _, score, _, _ in results}
    print("JSON output:")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
