"""FastAPI app — openenv create_app() + custom hackathon routes.

Spec reference: Sections 9, 14, 15.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from openenv.core.env_server.http_server import create_app

from ml_training_debugger.models import MLTrainingAction, MLTrainingObservation
from server._baseline_results import get_last_grader_result
from server.environment import MLTrainingEnvironment


# Structured JSON logging (Spec S15)
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id
        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id
        if hasattr(record, "step_count"):
            log_data["step_count"] = record.step_count
        if hasattr(record, "action_type"):
            log_data["action_type"] = record.action_type
        if hasattr(record, "score"):
            log_data["score"] = record.score
        return json.dumps(log_data)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# All 7 tasks (Spec S11 + Task 7 extension)
ALL_TASKS = [
    {"id": "task_001", "difficulty": "easy", "max_steps": 20},
    {"id": "task_002", "difficulty": "easy", "max_steps": 20},
    {"id": "task_003", "difficulty": "medium", "max_steps": 25},
    {"id": "task_004", "difficulty": "medium", "max_steps": 25},
    {"id": "task_005", "difficulty": "hard", "max_steps": 30},
    {"id": "task_006", "difficulty": "hard", "max_steps": 30},
    {"id": "task_007", "difficulty": "hard", "max_steps": 25},
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
app.routes[:] = [
    r for r in app.routes if not (hasattr(r, "path") and r.path == "/health")
]

# Thread-safe baseline lock (Fix #14)
_baseline_lock = asyncio.Lock()


@app.get("/")
def root():
    """Redirect root to dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


@app.get("/health")
def health_check() -> dict:
    """Health check — required by hackathon auto-validator."""
    return {"status": "ready", "tasks": len(ALL_TASKS)}


@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard() -> str:
    """Serve live diagnostic dashboard. Spec Section 19."""
    import pathlib

    html_path = pathlib.Path(__file__).parent / "dashboard.html"
    return html_path.read_text()


@app.get("/validation-report")
def get_validation_report() -> dict:
    """Serve pre-computed simulation fidelity report. Spec Section 18."""
    import pathlib

    report_path = (
        pathlib.Path(__file__).parent.parent
        / "validation"
        / "reports"
        / "fidelity_report.json"
    )
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {"error": "Validation report not yet generated. Run: python validation/run_all_validations.py"}


@app.get("/curriculum")
def get_curriculum() -> dict:
    """Recommended task order for RL agent training (easy → hard, with difficulty scaling)."""
    curriculum: list[dict] = []
    for task in ALL_TASKS:
        for level in [1, 3, 5]:
            curriculum.append({
                "task_id": task["id"],
                "difficulty": task["difficulty"],
                "difficulty_level": level,
                "max_steps": task["max_steps"],
            })
    return {"curriculum": curriculum, "total_episodes": len(curriculum)}


@app.get("/leaderboard")
def get_leaderboard() -> dict:
    """Sorted leaderboard of completed episode scores."""
    from server._baseline_results import _last_results

    entries = [
        v for k, v in _last_results.items() if k != "_latest" and isinstance(v, dict)
    ]
    sorted_entries = sorted(entries, key=lambda x: x.get("score", 0), reverse=True)
    return {"entries": sorted_entries, "total": len(sorted_entries)}


@app.get("/replay/{episode_id}")
def get_replay(episode_id: str) -> dict:
    """Return full action/observation trace for a completed episode."""
    from server._baseline_results import _last_results

    result = _last_results.get(episode_id)
    if result is None:
        return {"error": f"Episode '{episode_id}' not found"}
    return {"episode_id": episode_id, **result}


@app.get("/tasks")
def get_tasks() -> list[dict]:
    """Return task list with IDs, difficulties, and action schema."""
    schema = MLTrainingAction.model_json_schema()
    return [{**task, "action_schema": schema} for task in ALL_TASKS]


@app.post("/grader")
def post_grader(session_id: Optional[str] = None) -> dict:
    """Return grader score for most recently completed episode.

    Edge cases per spec Section 14:
    - No episode completed → {"score": null, "error": "no_completed_episode"}
    - Episode completed → {"score": float, "task_id": str, "steps": int}
    """
    result = get_last_grader_result(session_id)
    if result is None:
        return {"score": None, "error": "no_completed_episode"}
    return result


@app.post("/baseline", response_model=None)
async def post_baseline():
    """Trigger baseline run, return scores for all tasks.

    Returns 409 if already running. Uses asyncio.Lock for thread safety.
    """
    if _baseline_lock.locked():
        return JSONResponse(
            status_code=409,
            content={"error": "baseline_in_progress"},
        )

    async with _baseline_lock:
        scores = await asyncio.get_event_loop().run_in_executor(
            None, _run_baseline_sync
        )
        return {"scores": scores}


def _run_baseline_sync() -> dict[str, float]:
    """Run the rule-based baseline synchronously."""
    scores: dict[str, float] = {}

    for task_info in ALL_TASKS:
        task_id = task_info["id"]
        env = MLTrainingEnvironment()
        env.reset(seed=42, episode_id=f"baseline_{task_id}", task_id=task_id)
        score = _run_heuristic_episode(env, task_id)
        scores[task_id] = round(score, 4)

    return scores


def _run_heuristic_episode(
    env: MLTrainingEnvironment,
    task_id: str,
) -> float:
    """Run one heuristic baseline episode. Returns grader score.

    Decision tree per spec Section 17.
    """
    # Step 1: inspect_gradients
    obs = env.step(MLTrainingAction(action_type="inspect_gradients"))

    if obs.gradient_stats:
        # Check exploding
        if any(g.is_exploding for g in obs.gradient_stats):
            env.step(
                MLTrainingAction(
                    action_type="modify_config",
                    target="learning_rate",
                    value=0.001,
                )
            )
            env.step(MLTrainingAction(action_type="restart_run"))
            env.step(
                MLTrainingAction(
                    action_type="mark_diagnosed",
                    diagnosis="lr_too_high",
                )
            )
            return _get_score(env)

        # Check vanishing
        if any(g.is_vanishing for g in obs.gradient_stats):
            env.step(
                MLTrainingAction(
                    action_type="modify_config",
                    target="learning_rate",
                    value=0.01,
                )
            )
            env.step(MLTrainingAction(action_type="restart_run"))
            env.step(
                MLTrainingAction(
                    action_type="mark_diagnosed",
                    diagnosis="vanishing_gradients",
                )
            )
            return _get_score(env)

    # Step 2: inspect_data_batch
    obs = env.step(MLTrainingAction(action_type="inspect_data_batch"))
    if obs.data_batch_stats and obs.data_batch_stats.class_overlap_score > 0.5:
        env.step(MLTrainingAction(action_type="patch_data_loader"))
        env.step(MLTrainingAction(action_type="restart_run"))
        env.step(
            MLTrainingAction(
                action_type="mark_diagnosed",
                diagnosis="data_leakage",
            )
        )
        return _get_score(env)

    # Detect overfitting pattern (used later, after ruling out code bugs)
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
        if (
            (train_dropped or train_loss_low)
            and (val_not_improving or gap_widening)
            and obs.data_batch_stats
            and obs.data_batch_stats.class_overlap_score < 0.3
        ):
            _looks_like_overfitting = True

    # Step 3: inspect_model_modes
    obs = env.step(MLTrainingAction(action_type="inspect_model_modes"))
    if obs.model_mode_info:
        has_eval = any(v == "eval" for v in obs.model_mode_info.values())
        if has_eval:
            env.step(MLTrainingAction(action_type="fix_model_mode"))
            env.step(MLTrainingAction(action_type="restart_run"))
            env.step(
                MLTrainingAction(
                    action_type="mark_diagnosed",
                    diagnosis="batchnorm_eval_mode",
                )
            )
            return _get_score(env)

    # Step 4: inspect_code (for Task 6)
    obs = env.step(MLTrainingAction(action_type="inspect_code"))
    if obs.code_snippet:
        code = obs.code_snippet.code
        if "model.eval()" in code and "model.train()" not in code:
            env.step(
                MLTrainingAction(
                    action_type="fix_code",
                    line=5,
                    replacement="model.train()",
                )
            )
        elif ".detach()" in code:
            env.step(
                MLTrainingAction(
                    action_type="fix_code",
                    line=14,
                    replacement="        loss = criterion(output, batch_y)",
                )
            )

        # Try restart if fix was applied
        session = env._get_session()
        if session and session.state.fix_action_taken:
            env.step(MLTrainingAction(action_type="restart_run"))

        env.step(
            MLTrainingAction(
                action_type="mark_diagnosed",
                diagnosis="code_bug",
            )
        )
        return _get_score(env)

    # Step 5: Check for scheduler issue (loss stagnates)
    if obs.training_loss_history and len(obs.training_loss_history) >= 10:
        early_loss = sum(obs.training_loss_history[:3]) / 3
        mid_loss = sum(obs.training_loss_history[5:8]) / 3
        finite_late = [v for v in obs.training_loss_history[-3:] if v != float("inf")]
        late_loss = sum(finite_late) / max(len(finite_late), 1)
        if early_loss > mid_loss and abs(late_loss - mid_loss) < 0.3:
            env.step(
                MLTrainingAction(
                    action_type="modify_config",
                    target="learning_rate",
                    value=0.001,
                )
            )
            env.step(MLTrainingAction(action_type="restart_run"))
            env.step(
                MLTrainingAction(
                    action_type="mark_diagnosed",
                    diagnosis="scheduler_misconfigured",
                )
            )
            return _get_score(env)

    # Overfitting fallback — only if code inspection didn't find a bug
    if _looks_like_overfitting:
        env.step(
            MLTrainingAction(
                action_type="modify_config",
                target="weight_decay",
                value=0.01,
            )
        )
        env.step(MLTrainingAction(action_type="restart_run"))
        env.step(
            MLTrainingAction(
                action_type="mark_diagnosed",
                diagnosis="overfitting",
            )
        )
        return _get_score(env)

    # Final fallback
    env.step(
        MLTrainingAction(
            action_type="mark_diagnosed",
            diagnosis="overfitting",
        )
    )
    return _get_score(env)


def _get_score(env: MLTrainingEnvironment) -> float:
    """Extract the grader score from the environment."""
    session = env._get_session()
    if session and session.last_score is not None:
        return session.last_score
    return 0.0


def main() -> None:
    """Entry point for running the server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
