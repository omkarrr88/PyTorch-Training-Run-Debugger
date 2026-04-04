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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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
def root() -> RedirectResponse:
    """Redirect root to dashboard."""
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
    return {
        "error": "Validation report not yet generated. "
        "Run: python validation/run_all_validations.py"
    }


@app.get("/curriculum")
def get_curriculum() -> dict:
    """Recommended task order (easy to hard, with difficulty scaling)."""
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
        v for k, v in _last_results.items()
        if k != "_latest" and isinstance(v, dict)
    ]
    sorted_entries = sorted(
        entries, key=lambda x: x.get("score", 0), reverse=True
    )
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
    """Return grader score for most recently completed episode."""
    result = get_last_grader_result(session_id)
    if result is None:
        return {"score": None, "error": "no_completed_episode"}
    return result


@app.post("/baseline", response_model=None)
async def post_baseline() -> JSONResponse | dict:
    """Trigger baseline run, return scores for all tasks."""
    if _baseline_lock.locked():
        return JSONResponse(
            status_code=409,
            content={"error": "baseline_in_progress"},
        )

    async with _baseline_lock:
        from server._heuristic import run_baseline_all_tasks

        scores = await asyncio.get_event_loop().run_in_executor(
            None, run_baseline_all_tasks
        )
        return {"scores": scores}


def main() -> None:
    """Entry point for running the server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
