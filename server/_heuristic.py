"""Rule-based heuristic baseline for the /baseline endpoint.

Extracted from app.py to keep route definitions under 400 lines.
Decision tree per spec Section 17.
"""

from __future__ import annotations

from ml_training_debugger.models import MLTrainingAction
from server.environment import MLTrainingEnvironment

ALL_TASK_IDS = [
    "task_001",
    "task_002",
    "task_003",
    "task_004",
    "task_005",
    "task_006",
    "task_007",
]


def run_baseline_all_tasks() -> dict[str, float]:
    """Run the rule-based baseline on all tasks. Returns {task_id: score}."""
    scores: dict[str, float] = {}
    for task_id in ALL_TASK_IDS:
        env = MLTrainingEnvironment()
        env.reset(seed=42, episode_id=f"baseline_{task_id}", task_id=task_id)
        scores[task_id] = round(_run_heuristic_episode(env), 4)
    return scores


def _run_heuristic_episode(
    env: MLTrainingEnvironment, task_id: str = "",
) -> float:
    """Run one heuristic baseline episode. Returns grader score."""
    # Step 1: inspect_gradients
    obs = env.step(MLTrainingAction(action_type="inspect_gradients"))

    if obs.gradient_stats:
        if any(g.is_exploding for g in obs.gradient_stats):
            env.step(MLTrainingAction(
                action_type="modify_config", target="learning_rate", value=0.001,
            ))
            env.step(MLTrainingAction(action_type="restart_run"))
            env.step(MLTrainingAction(
                action_type="mark_diagnosed", diagnosis="lr_too_high",
            ))
            return _get_score(env)

        if any(g.is_vanishing for g in obs.gradient_stats):
            env.step(MLTrainingAction(
                action_type="modify_config", target="learning_rate", value=0.01,
            ))
            env.step(MLTrainingAction(action_type="restart_run"))
            env.step(MLTrainingAction(
                action_type="mark_diagnosed", diagnosis="vanishing_gradients",
            ))
            return _get_score(env)

    # Step 2: inspect_data_batch
    obs = env.step(MLTrainingAction(action_type="inspect_data_batch"))
    if obs.data_batch_stats and obs.data_batch_stats.class_overlap_score > 0.5:
        env.step(MLTrainingAction(action_type="patch_data_loader"))
        env.step(MLTrainingAction(action_type="restart_run"))
        env.step(MLTrainingAction(
            action_type="mark_diagnosed", diagnosis="data_leakage",
        ))
        return _get_score(env)

    # Detect overfitting pattern
    looks_like_overfitting = _detect_overfitting(obs)

    # Step 3: inspect_model_modes
    obs = env.step(MLTrainingAction(action_type="inspect_model_modes"))
    if obs.model_mode_info:
        if any(v == "eval" for v in obs.model_mode_info.values()):
            env.step(MLTrainingAction(action_type="fix_model_mode"))
            env.step(MLTrainingAction(action_type="restart_run"))
            env.step(MLTrainingAction(
                action_type="mark_diagnosed", diagnosis="batchnorm_eval_mode",
            ))
            return _get_score(env)

    # Step 4: inspect_code (for Task 6)
    obs = env.step(MLTrainingAction(action_type="inspect_code"))
    if obs.code_snippet:
        code = obs.code_snippet.code
        _try_code_fix(env, code)

        session = env._get_session()
        if session and session.state.fix_action_taken:
            env.step(MLTrainingAction(action_type="restart_run"))

        env.step(MLTrainingAction(
            action_type="mark_diagnosed", diagnosis="code_bug",
        ))
        return _get_score(env)

    # Step 5: scheduler issue (loss stagnates after initial progress)
    if _detect_scheduler_issue(obs):
        env.step(MLTrainingAction(
            action_type="modify_config", target="learning_rate", value=0.001,
        ))
        env.step(MLTrainingAction(action_type="restart_run"))
        env.step(MLTrainingAction(
            action_type="mark_diagnosed", diagnosis="scheduler_misconfigured",
        ))
        return _get_score(env)

    # Overfitting fallback
    if looks_like_overfitting:
        env.step(MLTrainingAction(
            action_type="modify_config", target="weight_decay", value=0.01,
        ))
        env.step(MLTrainingAction(action_type="restart_run"))
        env.step(MLTrainingAction(
            action_type="mark_diagnosed", diagnosis="overfitting",
        ))
        return _get_score(env)

    # Final fallback
    env.step(MLTrainingAction(
        action_type="mark_diagnosed", diagnosis="overfitting",
    ))
    return _get_score(env)


def _try_code_fix(env: MLTrainingEnvironment, code: str) -> None:
    """Attempt to fix a detected code bug."""
    if "model.eval()" in code and "model.train()" not in code:
        env.step(MLTrainingAction(
            action_type="fix_code", line=5, replacement="model.train()",
        ))
    elif ".detach()" in code:
        env.step(MLTrainingAction(
            action_type="fix_code", line=14,
            replacement="        loss = criterion(output, batch_y)",
        ))
    elif "inplace=True" in code:
        env.step(MLTrainingAction(
            action_type="fix_code", line=15,
            replacement="        output = F.relu(output)",
        ))
    elif "optimizer.zero_grad()" not in code and "optimizer.step()" in code:
        env.step(MLTrainingAction(
            action_type="fix_code", line=11,
            replacement="        optimizer.zero_grad()",
        ))


def _detect_overfitting(obs: object) -> bool:
    """Detect overfitting pattern from observation."""
    if not (obs.val_loss_history and obs.training_loss_history
            and len(obs.val_loss_history) >= 10):
        return False
    early_train = sum(obs.training_loss_history[:5]) / 5
    late_train = sum(obs.training_loss_history[-5:]) / 5
    early_val = sum(obs.val_loss_history[:5]) / 5
    late_val = sum(obs.val_loss_history[-5:]) / 5
    train_dropped = late_train < early_train * 0.5
    train_loss_low = late_train < 0.15
    val_not_improving = late_val >= early_val * 0.95
    gap_widening = (late_val - late_train) > (early_val - early_train)
    return (
        (train_dropped or train_loss_low)
        and (val_not_improving or gap_widening)
        and obs.data_batch_stats
        and obs.data_batch_stats.class_overlap_score < 0.3
    )


def _detect_scheduler_issue(obs: object) -> bool:
    """Detect scheduler misconfiguration from loss history."""
    if not (obs.training_loss_history and len(obs.training_loss_history) >= 10):
        return False
    early_loss = sum(obs.training_loss_history[:3]) / 3
    mid_loss = sum(obs.training_loss_history[5:8]) / 3
    finite_late = [v for v in obs.training_loss_history[-3:] if v != float("inf")]
    late_loss = sum(finite_late) / max(len(finite_late), 1)
    return early_loss > mid_loss and abs(late_loss - mid_loss) < 0.3


def _get_score(env: MLTrainingEnvironment) -> float:
    """Extract the grader score from the environment."""
    session = env._get_session()
    if session and session.last_score is not None:
        return session.last_score
    return 0.0
