"""Shared state for grader results across endpoints."""

from __future__ import annotations

import threading
from typing import Optional

# Thread-safe store for completed episode results
_lock = threading.Lock()
_last_results: dict[str, dict] = {}


def store_grader_result(
    session_id: str, score: float, task_id: str, steps: int
) -> None:
    """Store a grader result for retrieval."""
    entry = {
        "score": round(score, 4),
        "task_id": task_id,
        "steps": steps,
    }
    with _lock:
        _last_results[session_id] = entry
        _last_results["_latest"] = entry


def get_last_grader_result(session_id: Optional[str] = None) -> dict | None:
    """Get grader result for a session, or the most recent one."""
    with _lock:
        if session_id:
            return _last_results.get(session_id)
        return _last_results.get("_latest")
