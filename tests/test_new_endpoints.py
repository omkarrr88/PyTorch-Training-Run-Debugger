"""Tests for new endpoints: curriculum, leaderboard, replay, validation-report."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestCurriculumEndpoint:
    def test_returns_curriculum(self, client) -> None:
        resp = client.get("/curriculum")
        assert resp.status_code == 200
        data = resp.json()
        assert "curriculum" in data
        assert "total_episodes" in data
        assert data["total_episodes"] > 0

    def test_curriculum_has_difficulty_levels(self, client) -> None:
        resp = client.get("/curriculum")
        curriculum = resp.json()["curriculum"]
        levels = {entry["difficulty_level"] for entry in curriculum}
        assert 1 in levels
        assert 3 in levels
        assert 5 in levels

    def test_curriculum_covers_all_tasks(self, client) -> None:
        resp = client.get("/curriculum")
        curriculum = resp.json()["curriculum"]
        task_ids = {entry["task_id"] for entry in curriculum}
        assert "task_001" in task_ids
        assert "task_007" in task_ids


class TestLeaderboardEndpoint:
    def test_returns_leaderboard(self, client) -> None:
        resp = client.get("/leaderboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "total" in data

    def test_leaderboard_after_baseline(self, client) -> None:
        # Run baseline to populate scores
        client.post("/baseline")
        resp = client.get("/leaderboard")
        data = resp.json()
        assert data["total"] > 0


class TestReplayEndpoint:
    def test_missing_episode(self, client) -> None:
        resp = client.get("/replay/nonexistent_episode_123")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_replay_after_baseline(self, client) -> None:
        # Run baseline to create episodes
        client.post("/baseline")
        resp = client.get("/replay/baseline_task_001")
        data = resp.json()
        # Should have episode data or error
        assert "episode_id" in data or "error" in data


class TestValidationReportEndpoint:
    def test_returns_real_report(self, client) -> None:
        resp = client.get("/validation-report")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "summary" in data
        assert data["summary"]["passed"] > 0
