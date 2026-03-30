# CLAUDE.md — PyTorch Training Run Debugger

OpenEnv RL environment for the Meta PyTorch OpenEnv Hackathon x Scaler School of Technology.
An AI agent debugs broken PyTorch training runs by investigating gradients, weights, data, model modes, and source code to diagnose and fix real ML failure patterns.

**Spec:** `ml-training-debugger-spec.md` is the single source of truth. If this file and the spec conflict, the spec wins.

**Runtime:** Python 3.12 · PyTorch CPU-only · openenv-core v0.2.2

---

## Non-Negotiable Rules

### MVP-First Execution
Ship Tasks 1, 3, 5 (easy/medium/hard) + rule-based baseline + Docker + HF deploy **before** touching anything else. A deployed MVP that passes auto-validation beats a half-finished 6-task environment. Priority order after MVP: Task 6 > Tasks 2 & 4 > dashboard > validation suite > LLM baseline.

### Context-Gated Penalty Must Be Exact
The -0.20 penalty for `add_callback` fires **only when both** `gradients_inspected == True` AND `gradients_were_normal == True`. It must **never** fire before `inspect_gradients` has been called. This is the project's primary innovation. Get the gate conditions wrong and the differentiator is broken. Test both paths:
- `add_callback` at step 1 (no prior inspection) -> **no penalty**
- `inspect_gradients` (normal) then `add_callback` -> **-0.20 penalty**

### Task 6 Diagnosis Is Always `code_bug`
Regardless of the specific bug variant (`eval_mode`, `detach_loss`, `zero_grad_missing`, `inplace_relu`), Task 6's correct diagnosis is **always** `code_bug`. Submitting `batchnorm_eval_mode` on Task 6's `eval_mode` variant is a wrong diagnosis (-0.30). The grader enforces this with a strict equality check.

### PyTorch-Native Only — No NumPy
Every computation in core modules uses `torch.Tensor`, not `numpy.ndarray`. `import torch` must appear in `models.py`, `simulation.py`, `pytorch_engine.py`, `reward_engine.py`, and `graders.py`. This is a Meta PyTorch hackathon — judges will notice. The only exception is test utilities and the validation suite where `scipy`/`matplotlib` are acceptable.

### Grader != Reward Function
These are separate modules with separate purposes. The **reward function** (`reward_engine.py`) returns a float per step for RL training signal. The **grader** (`graders.py`) returns a normalized 0.0-1.0 score at episode end for the `/grader` endpoint and auto-validation. The grader evaluates `EpisodeState` holistically — it is **not** a sum of step rewards. Never conflate them.

### Opaque Task IDs
Task IDs are `task_001` through `task_007`. The agent must never be able to infer the diagnosis from the task ID. Do not use descriptive names anywhere the agent can observe them.

---

## Architecture Constraints

### Framework Integration (Verified)
```
openenv-core v0.2.2 → create_app() → returns standard FastAPI instance
```

- `MLTrainingAction` extends `Action` from `openenv.core.env_server.types`
- `MLTrainingObservation` extends `Observation` from `openenv.core.env_server.types` (has built-in `done`, `reward`, `metadata`)
- `MLTrainingEnvironment` extends `Environment` from `openenv.core.env_server.interfaces` (must implement `reset()`, `step()`, `state` property)
- `MLTrainingEnvClient` in `client.py` extends `EnvClient` with typed `action_type` and `observation_type` — used by baseline scripts
- `create_app()` takes the **class** (factory), not an instance
- Custom routes (`/tasks`, `/grader`, `/baseline`, `/health`) are added directly to the returned FastAPI app via `@app.get()`/`@app.post()` decorators
- Framework auto-provides: `POST /reset`, `POST /step`, `GET /state`, `WS /ws`, `GET /schema`, `GET /docs`, `/mcp`

### Key Constraints (see spec for full detail)
- **Real PyTorch models:** `pytorch_engine.py` instantiates `SimpleCNN` (~50K params) at every `reset()`, runs 1-2 real forward+backward passes. Gradient and weight stats come from real `torch.autograd` and `model.state_dict()`.
- **Typed Pydantic models everywhere:** No `Dict[str, Any]`. `available_actions` is dynamically computed from `EpisodeState`, never hardcoded.
- **Session isolation:** Each WebSocket client gets its own `EpisodeState` keyed by session ID. `SUPPORTS_CONCURRENT_SESSIONS = True`.

---

## Coding Standards

### Formatting & Linting
- **black** for formatting (line length 88)
- **ruff** for linting
- **isort** for import ordering (profile=black)
- Run all three before every commit

### Type Hints
Type annotations on **every** function signature and return type. No `Any` in public APIs. Use `Optional[X]` for nullable fields, `Literal[...]` for closed string unions, `list[X]` (lowercase) for Python 3.12+.

### Testing
- **pytest** for all tests
- Every module in `ml_training_debugger/` has a corresponding `tests/test_*.py`
- Minimum test coverage: 80%
- Critical tests that must exist:
  - `test_reward_engine.py`: context-gated penalty fires/doesn't fire under correct conditions
  - `test_graders.py`: each grader returns 0.0-1.0, correct diagnosis scores high, wrong diagnosis scores low
  - `test_pytorch_engine.py`: model instantiation, fault injection, gradient/weight extraction produces real tensors
  - `test_code_templates.py`: all 4 bug variants generate valid code, fix validation accepts correct fixes and rejects wrong ones (including whitespace/comment variations)
  - `test_episode_lifecycle.py`: full episode flow reset->inspect->fix->restart->diagnose produces expected state transitions

### File Size Limits
- 400 lines typical, 800 max per file
- `models.py` may exceed 400 lines due to many Pydantic models — this is acceptable
- `pytorch_engine.py` must stay under 300 lines (isolate model definitions if needed)

### Error Handling
`step()` must **never** raise an unhandled exception. All invalid actions return a valid observation with `-0.05` penalty and an error note. All edge cases (step after done, step before reset, malformed JSON) return structured error responses.

---

## Key Risks to Watch

### Task 6 Code Fix Validation
LLM agents will submit fixes with trailing spaces, inline comments, or minor reformatting. Use the multi-strategy validation pipeline:
1. Normalize whitespace + strip comments
2. Token-stream comparison via `tokenize` module
3. 2-3 semantic equivalence patterns per bug variant
4. `ast.parse()` fallback to verify buggy pattern is absent

Test with intentionally messy fixes: `"  loss = criterion(output, batch_y)  # fixed  "` must pass.

### Red-Herring Penalty Gating
The `gradients_were_normal` flag is set **inside** the `inspect_gradients` handler, based on whether `is_exploding` is False on **all** layers. The threshold for `is_exploding` is `mean_norm > 10.0`. The threshold for `is_vanishing` is `mean_norm < 1e-6`. In Task 5, the FC spike has `is_exploding: False` (it spiked but the mean norm stays below 10.0), so `gradients_were_normal` is set to True. This is the gate that makes the penalty fire when the agent then calls `add_callback`.

### Docker Image Size
Current: 885MB. Uses torch 2.5.1+cpu with multi-stage build and `strip --strip-unneeded`. The irreducible minimum is `libtorch_cpu.so` (329MB stripped). Use `python:3.12-slim` base. Do NOT install CUDA.

### Baseline Reproducibility
The rule-based baseline must produce **bit-exact identical** scores on two consecutive runs. This requires:
- `torch.manual_seed(seed)` at every `reset()` with a deterministic seed per task
- No floating-point non-determinism in the parametric curve generators
- The heuristic decision tree is pure logic with no randomness

### Auto-Validator Endpoints
These endpoints are checked programmatically. They must respond correctly or you are disqualified:
- `GET /health` -> `{"status": "ready", "tasks": N}` (200) — N is the number of active tasks (7 for full)
- `GET /tasks` -> list of tasks with IDs and action schema (200)
- `POST /grader` -> `{"score": float}` after a completed episode (200)
- `POST /baseline` -> scores for all tasks (200)
- `WS /ws` -> responds to `reset` message

---

## Reward Constants (Do Not Change)

See spec Section 12 for full rationale. Summary:

| Event | Value | Gate |
|---|---|---|
| Step penalty | -0.01 | Unconditional, flat (never multiply by step_count) |
| Investigation bonus | +0.05 | First-time only per inspection type |
| Context-gated penalty | -0.20 | `gradients_inspected AND gradients_were_normal` |
| Invalid action | -0.05 | Action not in `available_actions` |
| Wrong code fix | -0.10 | `fix_code` with wrong line/replacement |
| Correct diagnosis | +0.50 | `diagnosis == true_root_cause` |
| Wrong diagnosis | -0.30 | `diagnosis != true_root_cause` |
| Terminal convergence | +0.40 | `fix_action_taken AND restart_after_fix AND convergence` |

---

## Success Criteria — "Perfect" Submission

All of these must be true:
- [ ] `openenv validate` passes
- [ ] `docker build && docker run` starts server on port 7860 in <60s
- [ ] HF Space deploys, responds to `reset()`, tagged with `openenv`
- [ ] `baseline_heuristic.py` produces identical scores on two runs
- [ ] 3+ tasks with graders returning scores in [0.0, 1.0] with meaningful variance
- [ ] Hard task (Task 5 or 6) genuinely challenges frontier models (score < 0.7 for heuristic)
- [ ] Context-gated penalty fires correctly and does not fire prematurely
- [ ] All typed Pydantic models, no `Dict[str, Any]`
- [ ] `import torch` in every core module, zero numpy imports in core
- [ ] README documents: environment description, action/observation spaces, task descriptions with difficulty, setup instructions, baseline scores
- [ ] POST `/baseline`, POST `/grader`, GET `/tasks` all respond correctly
- [ ] Test suite passes with >80% coverage

---

## Commands

```bash
# Development (from project root: ML Debugger/)
source .venv/bin/activate
uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

# Tests
pytest tests/ -v --cov=ml_training_debugger --cov-report=term-missing

# Formatting
black ml_training_debugger/ server/ tests/
ruff check ml_training_debugger/ server/ tests/ --fix
isort ml_training_debugger/ server/ tests/ --profile black

# Docker
docker build -t pytorch-debugger .
docker run -p 7860:7860 pytorch-debugger

# Smoke test
curl http://localhost:7860/health
curl http://localhost:7860/tasks
python baseline_heuristic.py > run1.json
python baseline_heuristic.py > run2.json
diff run1.json run2.json  # Must be empty

# OpenEnv validation
openenv validate
```
