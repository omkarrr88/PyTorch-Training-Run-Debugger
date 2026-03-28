"""Shared state for grader results across endpoints."""

from __future__ import annotations

from typing import Optional

# Store last completed episode results
_last_results: dict[str, dict] = {}


def store_grader_result(
    session_id: str, score: float, task_id: str, steps: int
) -> None:
    """Store a grader result for retrieval."""
    _last_results[session_id] = {
        "score": round(score, 4),
        "task_id": task_id,
        "steps": steps,
    }
    _last_results["_latest"] = _last_results[session_id]


def get_last_grader_result(session_id: Optional[str] = None) -> dict | None:
    """Get grader result for a session, or the most recent one."""
    if session_id:
        return _last_results.get(session_id)
    return _last_results.get("_latest")
