"""Integration tests for HTTP endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_returns_ready(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["tasks"] == 6


class TestTasksEndpoint:
    def test_returns_six_tasks(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        tasks = resp.json()
        assert len(tasks) == 6
        ids = [t["id"] for t in tasks]
        assert "task_001" in ids
        assert "task_006" in ids

    def test_tasks_have_action_schema(self, client):
        resp = client.get("/tasks")
        tasks = resp.json()
        for task in tasks:
            assert "action_schema" in task
            assert "properties" in task["action_schema"]


class TestGraderEndpoint:
    def test_no_completed_episode(self, client):
        import server._baseline_results as br

        br._last_results.clear()  # Reset shared state for clean test
        resp = client.post("/grader")
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] is None
        assert data["error"] == "no_completed_episode"


class TestDashboardEndpoint:
    def test_returns_html(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "Plotly" in resp.text
        assert "WebSocket" in resp.text
