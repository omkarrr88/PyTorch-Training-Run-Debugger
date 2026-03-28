---
name: OpenEnv framework WS message format
description: The openenv-core WS endpoint expects specific message formats. Task selection via data field WORKS. Critical for tests and agent integration.
type: feedback
---

The openenv-core framework's WebSocket endpoint at `/ws` uses Pydantic-validated message formats:

- **Reset (default task)**: `{"type": "reset"}`
- **Reset (select task)**: `{"type": "reset", "data": {"task_id": "task_003", "seed": 42}}` — WORKS! The `data` field passes kwargs to `reset()`.
- **Step**: `{"type": "step", "data": {"action_type": "inspect_gradients"}}` — use `"data"` NOT `"action"`

**Key discovery (2026-03-28):** `WSResetMessage` has `data: Dict[str, Any]` which passes through to `reset(**kwargs)`. Task selection via WS is NOT broken — just needs the `data` wrapper. Top-level extra fields like `{"type": "reset", "task_id": "..."}` fail with "Extra inputs not permitted."

**Why:** The framework's `WSResetMessage` uses Pydantic with `extra="forbid"` on top-level fields, but the `data` dict is `Dict[str, Any]` and passes freely.

**HTTP endpoints** are stateless by framework design — each `/reset` and `/step` creates a fresh environment instance and destroys it after. WS is the only stateful interface for full episodes.

**Response format:** `{"type": "observation", "data": {"observation": {...}, "reward": float, "done": bool}}`
