"""Integration tests for HTTP endpoints.

Covers: /health, /tasks, /grader, /baseline, /dashboard.
Also tests the internal _run_heuristic_episode and _run_baseline_sync.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import ALL_TASKS, app
from server._heuristic import (
    _get_score,
    _run_heuristic_episode,
    run_baseline_all_tasks as _run_baseline_sync,
)
from server.environment import MLTrainingEnvironment


@pytest.fixture
def client():
    return TestClient(app)


# ---------- /health ----------


class TestHealthEndpoint:
    def test_returns_ready(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["tasks"] == 7

    def test_task_count_matches_all_tasks(self, client):
        resp = client.get("/health")
        assert resp.json()["tasks"] == len(ALL_TASKS)


# ---------- /tasks ----------


class TestTasksEndpoint:
    def test_returns_six_tasks(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        tasks = resp.json()
        assert len(tasks) == 7
        ids = [t["id"] for t in tasks]
        assert "task_001" in ids
        assert "task_006" in ids

    def test_tasks_have_action_schema(self, client):
        resp = client.get("/tasks")
        tasks = resp.json()
        for task in tasks:
            assert "action_schema" in task
            assert "properties" in task["action_schema"]

    def test_tasks_have_difficulty_and_max_steps(self, client):
        resp = client.get("/tasks")
        for task in resp.json():
            assert "difficulty" in task
            assert task["difficulty"] in ("easy", "medium", "hard", "medium-hard")
            assert "max_steps" in task
            assert task["max_steps"] > 0


# ---------- /grader ----------


class TestGraderEndpoint:
    def test_no_completed_episode(self, client):
        import server._baseline_results as br

        br._last_results.clear()
        resp = client.post("/grader")
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] is None
        assert data["error"] == "no_completed_episode"

    def test_grader_after_completed_episode(self, client):
        """Run a quick episode then verify /grader returns a score."""
        import server._baseline_results as br

        br._last_results.clear()
        # Run a minimal episode via the internal function
        env = MLTrainingEnvironment()
        env.reset(seed=42, episode_id="grader_test", task_id="task_001")
        score = _run_heuristic_episode(env, "task_001")
        assert 0.0 <= score <= 1.0

        # Now the grader endpoint should return the stored result
        resp = client.post("/grader")
        data = resp.json()
        assert data["score"] is not None
        assert 0.0 <= data["score"] <= 1.0

    def test_grader_with_session_id(self, client):
        """Grader can filter by session_id."""
        import server._baseline_results as br

        br._last_results.clear()
        resp = client.post("/grader?session_id=nonexistent_session")
        data = resp.json()
        assert data["score"] is None


# ---------- /baseline ----------


class TestBaselineEndpoint:
    def test_baseline_returns_scores(self, client):
        resp = client.post("/baseline")
        assert resp.status_code == 200
        data = resp.json()
        assert "scores" in data
        scores = data["scores"]
        assert len(scores) == 7
        for task_id, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{task_id}: {score}"

    def test_baseline_scores_in_valid_range(self, client):
        resp = client.post("/baseline")
        scores = resp.json()["scores"]
        values = list(scores.values())
        assert all(0.0 <= v <= 1.0 for v in values), "Scores must be in [0.0, 1.0]"
        assert len(values) >= 3, "Need at least 3 tasks"


# ---------- /dashboard ----------


class TestDashboardEndpoint:
    def test_returns_html(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "Plotly" in resp.text
        assert "WebSocket" in resp.text


# ---------- Internal heuristic functions ----------


class TestRunHeuristicEpisode:
    """Test the internal baseline heuristic logic in app.py."""

    def test_task_001_exploding(self):
        env = MLTrainingEnvironment()
        env.reset(seed=42, episode_id="h_001", task_id="task_001")
        score = _run_heuristic_episode(env, "task_001")
        assert score == 1.0

    def test_task_002_vanishing(self):
        env = MLTrainingEnvironment()
        env.reset(seed=42, episode_id="h_002", task_id="task_002")
        score = _run_heuristic_episode(env, "task_002")
        assert score == 1.0

    def test_task_003_leakage(self):
        env = MLTrainingEnvironment()
        env.reset(seed=42, episode_id="h_003", task_id="task_003")
        score = _run_heuristic_episode(env, "task_003")
        assert score >= 0.9

    def test_task_004_overfitting(self):
        env = MLTrainingEnvironment()
        env.reset(seed=42, episode_id="h_004", task_id="task_004")
        score = _run_heuristic_episode(env, "task_004")
        assert 0.0 < score <= 1.0

    def test_task_005_batchnorm(self):
        env = MLTrainingEnvironment()
        env.reset(seed=42, episode_id="h_005", task_id="task_005")
        score = _run_heuristic_episode(env, "task_005")
        assert 0.0 < score <= 1.0

    def test_task_006_code_bug(self):
        env = MLTrainingEnvironment()
        env.reset(seed=42, episode_id="h_006", task_id="task_006")
        score = _run_heuristic_episode(env, "task_006")
        assert score >= 0.4


class TestGetScore:
    def test_no_session(self):
        env = MLTrainingEnvironment()
        assert _get_score(env) == 0.0

    def test_with_session(self):
        env = MLTrainingEnvironment()
        env.reset(seed=42, episode_id="gs_test", task_id="task_001")
        _run_heuristic_episode(env, "task_001")
        assert _get_score(env) >= 0.0


class TestRunBaselineSync:
    def test_returns_all_tasks(self):
        scores = _run_baseline_sync()
        assert len(scores) == 7
        for task_id in [
            "task_001",
            "task_002",
            "task_003",
            "task_004",
            "task_005",
            "task_006",
            "task_007",
        ]:
            assert task_id in scores
            assert 0.0 <= scores[task_id] <= 1.0

    def test_reproducible(self):
        scores1 = _run_baseline_sync()
        scores2 = _run_baseline_sync()
        assert scores1 == scores2
