"""FastAPI app — openenv create_app() + custom hackathon routes.

Spec reference: Sections 9, 14.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core.env_server.http_server import create_app

from ml_training_debugger.models import MLTrainingAction, MLTrainingObservation
from server.environment import MLTrainingEnvironment

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)

# MVP task list
MVP_TASKS = [
    {"id": "task_001", "difficulty": "easy", "max_steps": 20},
    {"id": "task_003", "difficulty": "medium", "max_steps": 25},
    {"id": "task_005", "difficulty": "hard", "max_steps": 30},
]

# create_app takes the class (factory), not an instance
app: FastAPI = create_app(
    MLTrainingEnvironment,
    MLTrainingAction,
    MLTrainingObservation,
    env_name="pytorch_training_debugger",
    max_concurrent_envs=5,
)

# Override framework's /health route with our custom version
# Remove the framework's health route first
app.routes[:] = [
    r for r in app.routes if not (hasattr(r, "path") and r.path == "/health")
]

# Track baseline state
_baseline_lock = asyncio.Lock()
_baseline_running = False


@app.get("/health")
def health_check() -> dict:
    """Health check — required by hackathon auto-validator."""
    return {"status": "ready", "tasks": len(MVP_TASKS)}


@app.get("/tasks")
def get_tasks() -> list[dict]:
    """Return task list with IDs, difficulties, and action schema."""
    schema = MLTrainingAction.model_json_schema()
    return [{**task, "action_schema": schema} for task in MVP_TASKS]


@app.post("/grader")
def post_grader(session_id: Optional[str] = None) -> dict:
    """Return grader score for most recently completed episode.

    Edge cases per spec Section 14:
    - No episode completed → {"score": null, "error": "no_completed_episode"}
    - Episode in progress → {"score": null, "error": "episode_in_progress"}
    - Episode completed → {"score": float, "task_id": str, "steps": int}
    """
    # Try to find the environment instance
    # The framework manages environment instances internally,
    # so we use the internal baseline results for the /grader endpoint
    from server._baseline_results import get_last_grader_result

    result = get_last_grader_result(session_id)
    if result is None:
        return {"score": None, "error": "no_completed_episode"}
    return result


@app.post("/baseline", response_model=None)
async def post_baseline():
    """Trigger baseline run, return scores for all tasks.

    Returns 409 if already running.
    """
    global _baseline_running

    if _baseline_running:
        return JSONResponse(
            status_code=409,
            content={"error": "baseline_in_progress"},
        )

    _baseline_running = True
    try:
        scores = await _run_baseline()
        return {"scores": scores}
    finally:
        _baseline_running = False


async def _run_baseline() -> dict[str, float]:
    """Run the rule-based baseline internally."""

    scores: dict[str, float] = {}

    for task_info in MVP_TASKS:
        task_id = task_info["id"]
        env = MLTrainingEnvironment()
        obs = env.reset(seed=42, episode_id=f"baseline_{task_id}", task_id=task_id)

        # Run heuristic decision tree
        score = _run_heuristic_episode(env, obs, task_id)
        scores[task_id] = round(score, 4)

    return scores


def _run_heuristic_episode(
    env: MLTrainingEnvironment,
    obs: MLTrainingObservation,
    task_id: str,
) -> float:
    """Run one heuristic baseline episode. Returns grader score."""
    # Step 1: inspect_gradients
    obs = env.step(MLTrainingAction(action_type="inspect_gradients"))

    # Check for exploding gradients
    if obs.gradient_stats:
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
            if session and session.last_score is not None:
                return session.last_score
            return 0.0

        # Check for vanishing gradients
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
            if session and session.last_score is not None:
                return session.last_score
            return 0.0

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
        if session and session.last_score is not None:
            return session.last_score
        return 0.0

    # Check for overfitting (val_loss diverging)
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
            if session and session.last_score is not None:
                return session.last_score
            return 0.0

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
            if session and session.last_score is not None:
                return session.last_score
            return 0.0

    # Step 4: inspect_code (for Task 6)
    obs = env.step(MLTrainingAction(action_type="inspect_code"))
    if obs.code_snippet:
        # Simple pattern matching for known bugs
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
        else:
            # Can't reliably fix — just diagnose
            pass

        if obs.episode_state.fix_action_taken:
            obs = env.step(MLTrainingAction(action_type="restart_run"))

        obs = env.step(
            MLTrainingAction(
                action_type="mark_diagnosed",
                diagnosis="code_bug",
            )
        )
        session = env._get_session()
        if session and session.last_score is not None:
            return session.last_score
        return 0.0

    # Fallback
    obs = env.step(
        MLTrainingAction(
            action_type="mark_diagnosed",
            diagnosis="overfitting",
        )
    )
    session = env._get_session()
    if session and session.last_score is not None:
        return session.last_score
    return 0.0
