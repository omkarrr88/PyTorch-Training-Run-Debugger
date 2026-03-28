#!/usr/bin/env python3
"""Rule-based heuristic baseline agent.

Deterministic decision tree — no API key required. Bit-exact reproducible.
Spec reference: Section 17.

Usage:
    python baseline_heuristic.py [--url http://localhost:7860]
"""

from __future__ import annotations

import argparse
import json
import sys

from ml_training_debugger.graders import grade_episode
from ml_training_debugger.models import EpisodeState, MLTrainingAction, MLTrainingObservation
from ml_training_debugger.scenarios import sample_scenario
from server.environment import MLTrainingEnvironment

MVP_TASKS = ["task_001", "task_003", "task_005"]


def run_heuristic_episode(task_id: str, seed: int = 42) -> float:
    """Run one heuristic baseline episode. Returns grader score."""
    env = MLTrainingEnvironment()
    obs = env.reset(seed=seed, episode_id=f"baseline_{task_id}", task_id=task_id)

    # Step 1: inspect_gradients
    obs = env.step(MLTrainingAction(action_type="inspect_gradients"))

    if obs.gradient_stats:
        # Check exploding
        if any(g.is_exploding for g in obs.gradient_stats):
            obs = env.step(
                MLTrainingAction(
                    action_type="modify_config",
                    target="learning_rate",
                    value=0.001,
                )
            )
            obs = env.step(MLTrainingAction(action_type="restart_run"))
            obs = env.step(
                MLTrainingAction(
                    action_type="mark_diagnosed",
                    diagnosis="lr_too_high",
                )
            )
            session = env._get_session()
            return session.last_score if session and session.last_score is not None else 0.0

        # Check vanishing
        if any(g.is_vanishing for g in obs.gradient_stats):
            obs = env.step(
                MLTrainingAction(
                    action_type="modify_config",
                    target="learning_rate",
                    value=0.01,
                )
            )
            obs = env.step(MLTrainingAction(action_type="restart_run"))
            obs = env.step(
                MLTrainingAction(
                    action_type="mark_diagnosed",
                    diagnosis="vanishing_gradients",
                )
            )
            session = env._get_session()
            return session.last_score if session and session.last_score is not None else 0.0

    # Step 2: inspect_data_batch
    obs = env.step(MLTrainingAction(action_type="inspect_data_batch"))
    if obs.data_batch_stats and obs.data_batch_stats.class_overlap_score > 0.5:
        obs = env.step(MLTrainingAction(action_type="patch_data_loader"))
        obs = env.step(MLTrainingAction(action_type="restart_run"))
        obs = env.step(
            MLTrainingAction(
                action_type="mark_diagnosed",
                diagnosis="data_leakage",
            )
        )
        session = env._get_session()
        return session.last_score if session and session.last_score is not None else 0.0

    # Check overfitting (val_loss diverging)
    if obs.val_loss_history and len(obs.val_loss_history) >= 10:
        early = sum(obs.val_loss_history[:5]) / 5
        late = sum(obs.val_loss_history[-5:]) / 5
        if (
            late > early * 1.2
            and obs.data_batch_stats
            and obs.data_batch_stats.class_overlap_score < 0.1
        ):
            obs = env.step(
                MLTrainingAction(
                    action_type="modify_config",
                    target="weight_decay",
                    value=0.01,
                )
            )
            obs = env.step(MLTrainingAction(action_type="restart_run"))
            obs = env.step(
                MLTrainingAction(
                    action_type="mark_diagnosed",
                    diagnosis="overfitting",
                )
            )
            session = env._get_session()
            return session.last_score if session and session.last_score is not None else 0.0

    # Step 3: inspect_model_modes
    obs = env.step(MLTrainingAction(action_type="inspect_model_modes"))
    if obs.model_mode_info:
        has_eval = any(v == "eval" for v in obs.model_mode_info.values())
        if has_eval:
            obs = env.step(MLTrainingAction(action_type="fix_model_mode"))
            obs = env.step(MLTrainingAction(action_type="restart_run"))
            obs = env.step(
                MLTrainingAction(
                    action_type="mark_diagnosed",
                    diagnosis="batchnorm_eval_mode",
                )
            )
            session = env._get_session()
            return session.last_score if session and session.last_score is not None else 0.0

    # Step 4: inspect_code
    obs = env.step(MLTrainingAction(action_type="inspect_code"))
    if obs.code_snippet:
        code = obs.code_snippet.code
        if "model.eval()" in code and "model.train()" not in code:
            obs = env.step(
                MLTrainingAction(
                    action_type="fix_code",
                    line=5,
                    replacement="model.train()",
                )
            )
        elif ".detach()" in code:
            obs = env.step(
                MLTrainingAction(
                    action_type="fix_code",
                    line=14,
                    replacement="        loss = criterion(output, batch_y)",
                )
            )

        if obs.episode_state.fix_action_taken:
            obs = env.step(MLTrainingAction(action_type="restart_run"))

        obs = env.step(
            MLTrainingAction(
                action_type="mark_diagnosed",
                diagnosis="code_bug",
            )
        )
        session = env._get_session()
        return session.last_score if session and session.last_score is not None else 0.0

    # Fallback
    obs = env.step(
        MLTrainingAction(
            action_type="mark_diagnosed",
            diagnosis="overfitting",
        )
    )
    session = env._get_session()
    return session.last_score if session and session.last_score is not None else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based baseline agent")
    parser.add_argument("--url", default="http://localhost:7860")
    args = parser.parse_args()

    scores: dict[str, float] = {}
    for task_id in MVP_TASKS:
        score = run_heuristic_episode(task_id)
        scores[task_id] = round(score, 4)

    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
