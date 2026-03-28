"""WebSocket integration tests.

Verifies the /ws endpoint works with correct message formats.
Auto-validators test: connect -> reset -> step -> diagnose.

Key discovery: WSResetMessage has a `data: Dict[str, Any]` field.
Task selection via WS: {"type": "reset", "data": {"task_id": "task_003"}}
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from server.app import app


class TestWebSocketEndpoint:
    """Test WebSocket /ws endpoint."""

    def test_ws_endpoint_exists(self) -> None:
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/ws" in paths

    def test_ws_reset_returns_observation(self) -> None:
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            resp = ws.receive_json()

            assert resp["type"] == "observation"
            obs = resp["data"]["observation"]
            assert len(obs["training_loss_history"]) == 20
            assert len(obs["val_accuracy_history"]) == 20
            assert len(obs["val_loss_history"]) == 20
            assert obs["framework"] == "pytorch"
            assert obs["epoch"] == 20
            assert isinstance(obs["available_actions"], list)
            assert len(obs["available_actions"]) > 0
            assert obs["episode_state"]["step_count"] == 0

    def test_ws_reset_with_task_selection(self) -> None:
        """Task selection via WS using data field."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            # Task 3 is data leakage — has specific notes
            ws.send_json({"type": "reset", "data": {"task_id": "task_003", "seed": 42}})
            resp = ws.receive_json()

            assert resp["type"] == "observation"
            obs = resp["data"]["observation"]
            assert "architecture upgraded" in obs.get("notes", "").lower()
            assert obs["error_log"] is None  # Task 3 has no error log

    def test_ws_task_selection_all_tasks(self) -> None:
        """Verify all 6 tasks can be selected via WS."""
        client = TestClient(app)
        task_ids = ["task_001", "task_002", "task_003", "task_004", "task_005", "task_006"]

        for task_id in task_ids:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "reset", "data": {"task_id": task_id, "seed": 42}})
                resp = ws.receive_json()
                assert resp["type"] == "observation", f"{task_id} failed reset"
                obs = resp["data"]["observation"]
                assert len(obs["training_loss_history"]) == 20, f"{task_id} missing loss history"

    def test_ws_step_inspect_gradients(self) -> None:
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            ws.receive_json()

            ws.send_json(
                {"type": "step", "data": {"action_type": "inspect_gradients"}}
            )
            resp = ws.receive_json()

            assert resp["type"] == "observation"
            obs = resp["data"]["observation"]
            assert len(obs["gradient_stats"]) == 4
            assert obs["episode_state"]["gradients_inspected"] is True
            for g in obs["gradient_stats"]:
                assert "layer_name" in g
                assert "mean_norm" in g
                assert "is_exploding" in g
                assert "is_vanishing" in g

    def test_ws_full_episode_flow(self) -> None:
        """Full episode: reset -> inspect -> fix -> restart -> diagnose."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            # Reset to task_001 (exploding gradients)
            ws.send_json({"type": "reset", "data": {"task_id": "task_001", "seed": 42}})
            resp = ws.receive_json()
            obs = resp["data"]["observation"]
            assert obs["error_log"] is not None

            # Inspect gradients
            ws.send_json(
                {"type": "step", "data": {"action_type": "inspect_gradients"}}
            )
            resp = ws.receive_json()
            obs = resp["data"]["observation"]
            assert any(g["is_exploding"] for g in obs["gradient_stats"])

            # Fix: reduce learning rate
            ws.send_json(
                {
                    "type": "step",
                    "data": {
                        "action_type": "modify_config",
                        "target": "learning_rate",
                        "value": 0.001,
                    },
                }
            )
            resp = ws.receive_json()
            obs = resp["data"]["observation"]
            assert obs["episode_state"]["fix_action_taken"] is True

            # Restart
            ws.send_json({"type": "step", "data": {"action_type": "restart_run"}})
            resp = ws.receive_json()
            obs = resp["data"]["observation"]
            assert obs["episode_state"]["restart_after_fix"] is True

            # Diagnose
            ws.send_json(
                {
                    "type": "step",
                    "data": {
                        "action_type": "mark_diagnosed",
                        "diagnosis": "lr_too_high",
                    },
                }
            )
            resp = ws.receive_json()
            done = resp["data"].get("done", False)
            obs = resp["data"]["observation"]
            assert done or obs["episode_state"]["diagnosis_submitted"]

    def test_ws_task_005_red_herrings(self) -> None:
        """Task 5 via WS — verify red herrings and correct diagnosis path."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {"task_id": "task_005", "seed": 42}})
            resp = ws.receive_json()
            obs = resp["data"]["observation"]
            # Task 5 has GPU memory warning
            assert obs.get("error_log") is not None
            assert obs["gpu_memory_used_gb"] > 14.0  # 91% of 16GB

            # Inspect gradients — all should be non-exploding
            ws.send_json(
                {"type": "step", "data": {"action_type": "inspect_gradients"}}
            )
            resp = ws.receive_json()
            obs = resp["data"]["observation"]
            for g in obs["gradient_stats"]:
                assert not g["is_exploding"]

            # Inspect model modes — should reveal eval mode
            ws.send_json(
                {"type": "step", "data": {"action_type": "inspect_model_modes"}}
            )
            resp = ws.receive_json()
            obs = resp["data"]["observation"]
            assert any(v == "eval" for v in obs["model_mode_info"].values())

    def test_ws_task_006_code_inspection(self) -> None:
        """Task 6 via WS — verify code inspection and fix."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {"task_id": "task_006", "seed": 42}})
            ws.receive_json()

            # Inspect code
            ws.send_json(
                {"type": "step", "data": {"action_type": "inspect_code"}}
            )
            resp = ws.receive_json()
            obs = resp["data"]["observation"]
            assert obs["code_snippet"] is not None
            assert obs["code_snippet"]["filename"] == "train.py"
            assert obs["code_snippet"]["line_count"] > 0

    def test_ws_invalid_message_returns_error(self) -> None:
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            ws.receive_json()

            # Wrong format — "action" instead of "data"
            ws.send_json(
                {"type": "step", "action": {"action_type": "inspect_gradients"}}
            )
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_ws_step_data_batch(self) -> None:
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            ws.receive_json()

            ws.send_json(
                {"type": "step", "data": {"action_type": "inspect_data_batch"}}
            )
            resp = ws.receive_json()
            obs = resp["data"]["observation"]
            assert obs["data_batch_stats"] is not None
            assert "class_overlap_score" in obs["data_batch_stats"]
            assert obs["episode_state"]["data_inspected"] is True
