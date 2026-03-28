# Implementation Plan: Fix All Hackathon Gaps

## Task Type
- [x] Backend (→ Claude direct — all fixes are Python/server-side)

## Key Discovery

**WS task selection WORKS!** The correct format is:
```json
{"type": "reset", "data": {"task_id": "task_003", "seed": 42}}
```
The framework's `WSResetMessage` has a `data: Dict[str, Any]` field that passes kwargs to `reset()`. This was previously thought broken but actually works — just needs the `data` wrapper.

**Impact**: The "CRITICAL" WS task selection issue is actually just a documentation/test gap, not a code bug.

---

## Implementation Steps

### Step 1: Fix WS Tests to Use Correct Task Selection Format
**Files**: `tests/test_websocket.py`
**What**: Update tests to verify `{"type": "reset", "data": {"task_id": "task_003"}}` works. Add tests for all 6 tasks via WS.
**Deliverable**: Tests proving WS task selection works for all tasks.

### Step 2: Update README WS Documentation
**Files**: `README.md`
**What**: Update WS reset format docs to show the `data` field:
```json
{"type": "reset", "data": {"task_id": "task_003", "seed": 42}}
```
**Deliverable**: Correct documentation.

### Step 3: Fix HTTP /step Session Isolation
**Files**: `server/environment.py`, `server/app.py`
**What**: Add a module-level shared session store so HTTP `/reset` and `/step` share state. The framework creates a new env instance per WS connection but HTTP requests use the app-level routes.
**Approach**: Use a module-level `_shared_sessions` dict in `_baseline_results.py` (or a new module) that the environment reads from. When HTTP `/reset` creates a session, store it. When HTTP `/step` runs, look up the session.
**Alternative**: If the framework already handles HTTP session state internally, this may not be fixable without patching the framework. In that case, document that WS is the primary interface and HTTP is for single-action calls only.
**Deliverable**: HTTP reset+step work for full episodes, OR clear documentation that WS is the primary interface.

### Step 4: Run Real Validation Suite & Store Results
**Files**: `validation/validate_*.py` (create missing scripts), `server/app.py` (update endpoint)
**What**:
- Create validation scripts for all 6 fault types (only exploding_gradients exists)
- Run them locally, capture R² scores
- Store results in `validation/reports/fidelity_report.json`
- Update `/validation-report` endpoint to serve real pre-computed data
**Deliverable**: Real fidelity scores served at `/validation-report`.

### Step 5: Verify Dashboard Real-Time Updates
**Files**: `server/dashboard.html`
**What**: Start server, open dashboard in browser, run an episode via the dashboard's built-in controls (the HTML has task select + run button). Verify charts update. If they don't, fix the WS connection in the dashboard JS.
**Deliverable**: Dashboard shows live episode data.

### Step 6: Update EXPLANATION.md and README with WS Format
**Files**: `EXPLANATION.md`, `README.md`
**What**: Fix the WS documentation to show the correct task selection format.
**Deliverable**: Accurate docs.

### Step 7: Docker Size — Document the Reality
**Files**: `README.md`
**What**: Add a note explaining why the image is ~1.5GB:
> "PyTorch CPU-only requires libtorch_cpu.so (426MB) for real torch.nn.Module and torch.autograd support. This is the minimum for a PyTorch-native environment — the trade-off for real gradient computation vs synthetic data."
**Deliverable**: Judges understand the trade-off is intentional.

### Step 8: Run Full Smoke Test
**What**: Execute the complete pre-submission checklist against Docker container.
**Deliverable**: All gates pass.

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| tests/test_websocket.py | Modify | Add WS task selection tests for all 6 tasks |
| README.md | Modify | Fix WS reset format, add Docker size note |
| EXPLANATION.md | Modify | Fix WS reset format |
| server/app.py:93-137 | Modify | Update /validation-report with real data |
| validation/validate_*.py | Create | Validation scripts for all fault types |
| validation/reports/fidelity_report.json | Create | Pre-computed R² scores |

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| HTTP /step session isolation may not be fixable | Document WS as primary interface; HTTP for single calls |
| Validation R² may be low for some fault types | Use directional agreement as fallback metric |
| Dashboard WS may not connect | Check browser console, fix WS URL construction |

## SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A
- GEMINI_SESSION: N/A
