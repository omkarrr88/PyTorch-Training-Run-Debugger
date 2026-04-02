# ROADMAP — PyTorch Training Run Debugger

**Timeline:** March 28 - April 8, 2026 (11 days)
**Runtime:** Python 3.12 · PyTorch CPU-only · openenv-core v0.2.2
**Governing documents:** `ml-training-debugger-spec.md` (source of truth), `PRD.md` (requirements), `CLAUDE.md` (coding rules)
**Iron rule:** No phase begins until the previous phase's acceptance criteria are met. The single exception: Phase 0 and Phase 1 file creation can overlap on Day 1.

---

## Phase 0: Setup & Validation (Days 1-2)

**Goal:** A running skeleton server that proves the toolchain works end-to-end. Zero business logic — just plumbing.

### 0.1 Files to Create

| File | Purpose | Lines (est.) |
|---|---|---|
| `ML Debugger/` (this directory) | Project root directory (git init here) | — |
| `pyproject.toml` | Project metadata, dependencies (torch CPU, openenv-core, pydantic>=2.0, fastapi, uvicorn, pytest, black, ruff, isort) | ~40 |
| `requirements.txt` | Flat dependency list mirroring pyproject.toml (Docker uses this). **Exclude openai** — deferred to Phase 3. | ~10 |
| `.python-version` | `3.12` | 1 |
| `openenv.yaml` | Full metadata — start with 3 MVP tasks (task_001, task_003, task_005), expand later | ~50 |
| `Dockerfile` | `python:3.12-slim`, torch CPU-only, openenv-core, app deps, port 7860 | ~15 |
| `.dockerignore` | Exclude `.venv/`, `__pycache__/`, `.git/`, `validation/reports/*.png` | ~10 |
| `.gitignore` | `.venv/`, `__pycache__/`, `*.pyc`, `.env`, `run*.json` | ~15 |
| `ml_training_debugger/__init__.py` | Package init, version string | ~3 |
| `ml_training_debugger/models.py` | **Stub only:** `RootCauseDiagnosis` enum, `EpisodeState`, `TrainingConfig`, `GradientStats`, `DataBatchStats`, `ModelWeightStats`, `CodeSnippet`, `MLTrainingObservation` (extends `Observation`), `MLTrainingAction` (extends `Action`). All fields typed, all values defaulted. | ~200 |
| `ml_training_debugger/client.py` | **Stub:** `MLTrainingEnvClient` extending `EnvClient` with `action_type = MLTrainingAction` and `observation_type = MLTrainingObservation`. Used by baseline scripts. | ~20 |
| `server/__init__.py` | Empty | 0 |
| `server/environment.py` | **Stub:** `MLTrainingEnvironment(Environment)` with `reset()` returning a hardcoded observation, `step()` echoing back, `state` property | ~50 |
| `server/app.py` | `create_app(MLTrainingEnvironment, MLTrainingAction, MLTrainingObservation)` + stub routes for `/tasks`, `/grader`, `/baseline`, `/health` | ~60 |
| `tests/__init__.py` | Empty | 0 |
| `tests/test_models.py` | Validate all Pydantic models instantiate, serialize to JSON, and round-trip | ~60 |
| `tests/conftest.py` | Shared fixtures: sample `EpisodeState`, sample `ScenarioParams`, sample observation | ~40 |

### 0.2 Dependencies to Install

```bash
# Create venv inside ML Debugger/ project root
python3 -m venv .venv && source .venv/bin/activate

# Core runtime
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openenv-core pydantic>=2.0 fastapi uvicorn

# Dev tools
pip install pytest pytest-cov pytest-asyncio black ruff isort httpx websockets

# NOTE: openai is deferred to Phase 3 (LLM baseline). Do NOT install now.
```

### 0.3 Validation Steps (Must All Pass)

| # | Command | Expected Result |
|---|---|---|
| 1 | `python -c "import torch; print(torch.__version__)"` | Version string, no CUDA |
| 2 | `python -c "from openenv.core.env_server.http_server import create_app"` | No import error |
| 3 | `python -c "from ml_training_debugger.models import MLTrainingAction, MLTrainingObservation"` | No import error |
| 4 | `python -c "from ml_training_debugger.client import MLTrainingEnvClient"` | No import error |
| 5 | `uvicorn server.app:app --host 0.0.0.0 --port 7860` | Server starts, no crash |
| 6 | `curl http://localhost:7860/health` | `{"status": "ready", "tasks": 3}` |
| 7 | `curl http://localhost:7860/tasks` | JSON with task list |
| 8 | `curl http://localhost:7860/docs` | Swagger UI loads |
| 9 | `pytest tests/test_models.py -v` | All pass |
| 10 | `docker build -t pytorch-debugger .` | Builds in <5min, image <500MB |
| 11 | `docker run -p 7860:7860 pytorch-debugger` then `curl /health` | Returns `{"status": "ready", "tasks": 3}` |
| 12 | `openenv validate` | Passes (or identify what needs fixing) |
| 13 | `black --check . && ruff check . && isort --check .` | Clean |

### 0.4 Acceptance Criteria

- [ ] Skeleton server starts on port 7860 and responds to `/health`, `/tasks`, `/docs`, `/ws`
- [ ] `/health` returns `{"status": "ready", "tasks": 3}` (task count matches active tasks)
- [ ] All Pydantic models instantiate without error and serialize to valid JSON
- [ ] `client.py` imports without error
- [ ] Docker image builds under 500MB and container starts cleanly
- [ ] `openenv validate` passes or all failures are documented with a fix plan
- [ ] `pytest` runs with zero failures
- [ ] Git repo initialized, first commit made

---

## Phase 1: MVP — Tasks 1, 3, 5 + Core Engine (Days 2-6)

**Goal:** A fully functional 3-task environment that passes all auto-validation gates, deployed to HF Spaces. This is the survival milestone — everything after this is differentiation.

### 1.1 Files to Create

| File | Purpose | Lines (est.) | Depends On |
|---|---|---|---|
| `ml_training_debugger/scenarios.py` | `ScenarioParams` dataclass, `sample_scenario(task_id, seed)` for tasks 001/003/005. Parameter ranges from spec Section 11. | ~120 | `models.py` |
| `ml_training_debugger/pytorch_engine.py` | `SimpleCNN(torch.nn.Module)`, `inject_fault(model, scenario)`, `extract_gradient_stats(model)`, `extract_weight_stats(model)`. Real torch.autograd. | ~250 | `scenarios.py` |
| `ml_training_debugger/simulation.py` | `gen_loss_history(scenario)`, `gen_val_accuracy_history(scenario)`, `gen_val_loss_history(scenario)`. All `torch.Tensor` ops. Parametric curves per spec Section 6. | ~180 | `scenarios.py` |
| `ml_training_debugger/reward_engine.py` | `compute_reward(action, episode_state, scenario) -> float`. All 7 reward components per spec Section 12. Context-gated penalty logic. | ~100 | `models.py` |
| `ml_training_debugger/graders.py` | `grade_task_001(state, scenario)`, `grade_task_003(...)`, `grade_task_005(...)`. Each returns float in [0.0, 1.0]. Per spec Section 11 grader breakdowns. | ~150 | `models.py` |
| `baseline_heuristic.py` | Deterministic decision tree agent using `MLTrainingEnvClient`. Runs all MVP tasks, prints JSON scores. | ~150 | `client.py`, server running |
| `README.md` | Environment description, action/observation spaces, task descriptions with difficulty, setup instructions, baseline scores table | ~200 | Everything |

### 1.2 Files to Edit

| File | Changes | Why |
|---|---|---|
| `ml_training_debugger/models.py` | Finalize all field types, add `available_actions` computation logic to `EpisodeState`, add red herring fields (notes, gpu_memory) | Stubs from Phase 0 become real |
| `ml_training_debugger/client.py` | Wire typed client to connect via WebSocket or HTTP as needed by baseline | Stub becomes functional |
| `server/environment.py` | Full `reset()` and `step()` implementations. See spec Sections 9, 13 for lifecycle. | Stubs become real |
| `server/app.py` | Wire `/tasks`, `/grader`, `/baseline`, `/health` to return real data. `/health` returns `{"status": "ready", "tasks": 3}`. | Stubs become real |
| `openenv.yaml` | Finalize observation_space, action_space, reward section. Verify task IDs and max_steps per spec Section 14. | Was skeletal in Phase 0 |
| `Dockerfile` | Add `COPY` for all new source files. Verify build still works. | New files added |

### 1.3 Tests to Create

| Test File | What It Covers | Critical Assertions |
|---|---|---|
| `tests/test_scenarios.py` | `sample_scenario()` for each MVP task | Returns correct root cause enum; params within defined ranges; different seeds produce different params |
| `tests/test_pytorch_engine.py` | Model instantiation, fault injection, gradient/weight extraction | `SimpleCNN` is a real `torch.nn.Module`; `extract_gradient_stats` returns `GradientStats` with real float norms; exploding fault produces `is_exploding=True`; batchnorm eval fault produces `model.training==False` |
| `tests/test_simulation.py` | Parametric curve generators | All outputs are `list[float]` of length 20; exploding LR produces diverging loss; leakage produces inflated val_acc; batchnorm produces slow val_acc degradation |
| `tests/test_reward_engine.py` | All 7 reward components | **Critical:** context-gated penalty fires when `gradients_inspected=True AND gradients_were_normal=True` then `add_callback`; does NOT fire when `add_callback` without prior inspection; step penalty is flat -0.01; investigation bonus is +0.05 first-time only |
| `tests/test_graders.py` | Graders for tasks 001, 003, 005 | Each returns float in [0.0, 1.0]; correct diagnosis + fix + restart = 1.0; wrong diagnosis < 0.5; partial completion scores between 0 and 1 |
| `tests/test_episode_lifecycle.py` | Full reset→inspect→fix→restart→diagnose flow | State transitions match spec Section 13; `available_actions` updates correctly; `done=True` after `mark_diagnosed`; step limit triggers `done=True` |

### 1.4 Task-Specific Implementation

See spec Section 11 for complete task specifications. Key implementation notes per task:

**Task 1 (`task_001`, easy):** Unambiguous signal. LR from spec ranges → real gradients explode → `is_exploding=True` on all layers. Straightforward grader.

**Task 3 (`task_003`, medium):** Red herring note about architecture upgrade. Data leakage confirmed via `class_overlap_score`. Normal model (no gradient/weight anomaly). Mild gradient elevation on one layer (`is_exploding=False`).

**Task 5 (`task_005`, hard):** The differentiator task. `gradients_were_normal=True` set inside `inspect_gradients` handler because `is_exploding=False` on ALL layers (FC spike mean_norm < 10.0). Context-gated penalty fires when agent then calls `add_callback`. Red herrings: FC spike, GPU 91%, conv1 near-vanishing, error_log warning.

### 1.5 Endpoint Responses

**`GET /health`:** `{"status": "ready", "tasks": 3}` (200) — or `{"status": "initializing"}` (503) during startup.

**`GET /tasks`:** Task list with IDs, difficulties, max_steps, and MLTrainingAction JSON schema.

**`POST /grader`:** `{"score": float, "task_id": str, "steps": int}` (200) — or `{"score": null, "error": "no_completed_episode"}` (200) if no episode. See spec Section 14 for edge cases.

**`POST /baseline`:** Runs baseline logic internally, returns `{"scores": {"task_001": float, "task_003": float, "task_005": float}}`. Returns 409 if already running.

### 1.6 Baseline Heuristic Decision Tree

See spec Section 17 for the complete decision tree. Summary:
```
1. reset(task_id)
2. inspect_gradients
3. IF any layer is_exploding → fix LR → restart → diagnose lr_too_high
4. IF any layer is_vanishing → fix LR → restart → diagnose vanishing_gradients
5. inspect_data_batch
6. IF class_overlap_score > 0.5 → patch_data_loader → restart → diagnose data_leakage
7. IF val_loss diverging → modify weight_decay → restart → diagnose overfitting
8. inspect_model_modes
9. IF any layer in "eval" → fix_model_mode → restart → diagnose batchnorm_eval_mode
10. inspect_code → attempt fix → restart → diagnose code_bug
11. FALLBACK: diagnose overfitting
```

### 1.7 Deploy to HF Spaces

| Step | Action | Verification |
|---|---|---|
| 1 | Create HF Space (Docker type), tag with `openenv` | Space page shows openenv tag |
| 2 | Push Dockerfile + source to Space repo | Build triggers automatically |
| 3 | Wait for build to complete | Build log shows success |
| 4 | Test health endpoint | `curl https://<space-url>/health` returns `{"status": "ready", "tasks": 3}` |
| 5 | Test reset via WebSocket | `wscat -c wss://<space-url>/ws` then send `{"type": "reset", "task_id": "task_001"}` |
| 6 | Run `openenv validate` against deployed space | All checks pass |

### 1.8 Acceptance Criteria

- [ ] `reset(task_id)` for tasks 001, 003, 005 returns valid `MLTrainingObservation` with correct initial state
- [ ] `step()` dispatches all 14 action types correctly (investigation, fix, terminal)
- [ ] `inspect_gradients` on Task 1 → `is_exploding=True` on all layers (real torch.autograd)
- [ ] `inspect_gradients` on Task 5 → `is_exploding=False` on all layers, `gradients_were_normal=True`
- [ ] `inspect_data_batch` on Task 3 → `class_overlap_score > 0.5`
- [ ] `inspect_model_modes` on Task 5 → all layers in "eval" mode
- [ ] Context-gated penalty: `inspect_gradients`(normal) then `add_callback` → reward includes -0.20
- [ ] Context-gated penalty: `add_callback` without prior inspection → NO -0.20 penalty
- [ ] Grader for Task 1: correct path scores 1.0, wrong diagnosis scores < 0.5
- [ ] Grader for Task 5: agent that chases red herring scores 0.80-0.85 (penalty applied)
- [ ] `baseline_heuristic.py` runs twice → `diff run1.json run2.json` is empty
- [ ] `POST /baseline` returns scores for all 3 tasks, all in [0.0, 1.0]
- [ ] `POST /grader` returns score after completed episode
- [ ] `GET /tasks` returns 3 tasks with action schema
- [ ] `GET /health` returns `{"status": "ready", "tasks": 3}`
- [ ] Docker builds <500MB, starts <60s, serves on port 7860
- [ ] HF Space deployed, responds to `reset()`, tagged `openenv`
- [ ] `openenv validate` passes
- [ ] `pytest --cov` shows >80% coverage on all Phase 1 modules
- [ ] `import torch` in every core module; zero `import numpy` in core
- [ ] README has: description, action/observation spaces, 3 task descriptions, setup instructions, baseline scores

---

## Phase 2: Stretch — Tasks 2, 4, 6 + Code Debugging (Days 7-9)

**Goal:** Full 6-task environment with code-level debugging. Task 6 is the single highest-impact differentiator for Meta judges.

**Prerequisites:** Phase 1 acceptance criteria ALL met. HF Space deployed and passing auto-validation.

### 2.1 Priority Order (Strict)

1. **Task 6** first — it is the strongest differentiator and the hardest to implement
2. **Task 2** second — structurally identical to Task 1 (vanishing vs. exploding), fastest to add
3. **Task 4** third — medium difficulty overfitting, similar pattern to existing tasks

### 2.2 Files to Create

| File | Purpose | Lines (est.) | Depends On |
|---|---|---|---|
| `ml_training_debugger/code_templates.py` | 4 bug variant templates, `generate_code_snippet(bug_type, seed)`, `validate_fix(bug_type, line, replacement)` with multi-strategy pipeline per spec Section 22 | ~250 | `models.py` |
| `tests/test_code_templates.py` | All 4 variants generate valid code; fix validation accepts correct fixes; rejects wrong fixes; handles whitespace/comment variations | ~150 | `code_templates.py` |

### 2.3 Files to Edit

| File | Changes | Complexity |
|---|---|---|
| `ml_training_debugger/scenarios.py` | Add `sample_scenario` cases for task_002, task_004, task_006. Task 006 includes `bug_type` field. | Low |
| `ml_training_debugger/pytorch_engine.py` | Add fault injection for vanishing gradients, overfitting, code bug variants. | Medium |
| `ml_training_debugger/simulation.py` | Add curve generators for vanishing (flat loss), overfitting (train-val divergence), code bug variants. | Medium |
| `ml_training_debugger/reward_engine.py` | Add wrong code fix penalty (-0.10). No other changes. | Low |
| `ml_training_debugger/graders.py` | Add `grade_task_002`, `grade_task_004`, `grade_task_006`. Task 006: diagnosis must be `code_bug` always. | Medium |
| `server/environment.py` | `step()` handlers for `inspect_code` and `fix_code`. Update `available_actions`. | Medium |
| `server/app.py` | Update `/tasks` to return 6 tasks. Update `/health` to return `"tasks": 6`. | Low |
| `openenv.yaml` | Add task_002, task_004, task_006. | Low |
| `baseline_heuristic.py` | Extend decision tree for vanishing, overfitting, code bug. | Medium |
| `README.md` | Add descriptions for Tasks 2, 4, 6. Update baseline scores. | Low |

### 2.4 Task 6 Code Fix Validation

The `validate_fix()` pipeline is defined in spec Section 22 (Known Risks). Key layers:

1. **Normalize:** strip whitespace + inline comments → compare against known correct strings
2. **Tokenize:** Python `tokenize` module, filter noise tokens, compare streams
3. **Semantic patterns:** 2-3 per variant (e.g. `"criterion("` present AND `".detach()"` absent)
4. **AST fallback:** `ast.parse()` full code with replacement, verify buggy pattern absent

Test cases that MUST pass: correct fix, trailing whitespace, inline comments, different indentation.
Test cases that MUST fail: bug still present, `pass`, wrong line number.

### 2.5 Tests to Create/Extend

| Test File | New Coverage |
|---|---|
| `tests/test_code_templates.py` | **New file.** All 4 variants, validate_fix accepts/rejects correctly, 5+ whitespace/comment variations per variant |
| `tests/test_scenarios.py` | Extend: sample_scenario for task_002, 004, 006 |
| `tests/test_simulation.py` | Extend: vanishing flat loss, overfitting divergence, code bug symptoms |
| `tests/test_graders.py` | Extend: graders 002, 004, 006. Task 006: `code_bug` required; `batchnorm_eval_mode` on eval_mode variant = wrong |
| `tests/test_reward_engine.py` | Extend: wrong code fix penalty (-0.10) |
| `tests/test_episode_lifecycle.py` | Extend: `inspect_code` → `fix_code` available; `fix_code` before `inspect_code` → invalid |

### 2.6 Acceptance Criteria

- [ ] All 6 tasks return valid observations from `reset()` and process all action types in `step()`
- [ ] Task 6: `inspect_code` returns `CodeSnippet` with real PyTorch code containing the sampled bug
- [ ] Task 6: `fix_code` correct → `fix_action_taken=True`, no penalty
- [ ] Task 6: `fix_code` wrong → -0.10 penalty
- [ ] Task 6: `mark_diagnosed(code_bug)` → correct (+0.50)
- [ ] Task 6: `mark_diagnosed(batchnorm_eval_mode)` on eval_mode variant → wrong (-0.30)
- [ ] `validate_fix` accepts 5+ whitespace/comment variations per variant
- [ ] `validate_fix` rejects all invalid fixes
- [ ] Graders for all 6 tasks return [0.0, 1.0] with meaningful variance
- [ ] `baseline_heuristic.py` handles all 6 tasks, still bit-exact reproducible
- [ ] `POST /baseline` returns scores for all 6 tasks
- [ ] `GET /tasks` returns 6 tasks
- [ ] `GET /health` returns `{"status": "ready", "tasks": 6}`
- [ ] All new tests pass; overall coverage >80%
- [ ] Updated openenv.yaml lists all 6 tasks
- [ ] HF Space redeployed with 6 tasks, auto-validation still passes

---

## Phase 3: Polish — Dashboard, Validation Suite, LLM Baseline (Days 10-11)

**Goal:** Transform a technically correct submission into a visually impressive, deeply validated, winning submission.

**Prerequisites:** Phase 2 acceptance criteria ALL met. 6-task environment deployed.

### 3.1 Priority Order Within Phase 3

1. **Dashboard** — transforms judging experience (highest ROI for judges)
2. **Full test suite + README polish** — ensures no auto-validation failure
3. **Validation suite** — answers "how realistic are your curves?"
4. **LLM baseline** — demonstrates heuristic-reasoning gap (lowest priority)

### 3.2 Files to Create

| File | Purpose | Lines (est.) | Priority |
|---|---|---|---|
| `server/dashboard.html` | Single-file SPA. 4 panels per spec Section 19. Plotly.js via CDN. | ~400 | 1st |
| `validation/requirements.txt` | `torch`, `matplotlib`, `scipy` | ~3 | 3rd |
| `validation/conftest.py` | Shared fixtures: CIFAR-10 subset loader, model definitions | ~50 | 3rd |
| `validation/validate_exploding_gradients.py` | Real training, compare to parametric curve, R² > 0.85 | ~80 | 3rd |
| `validation/validate_data_leakage.py` | Real training with leakage, compare | ~80 | 3rd |
| `validation/validate_batchnorm_eval.py` | Real training with `model.eval()`, compare | ~80 | 3rd |
| `validation/validate_vanishing_gradients.py` | Real gradient decay, compare | ~80 | 3rd |
| `validation/validate_overfitting.py` | Real train-val divergence, compare | ~80 | 3rd |
| `validation/validate_code_bugs.py` | Run 4 bug variants, confirm symptoms | ~80 | 3rd |
| `validation/reports/` | Pre-computed fidelity scores + comparison plots | — | 3rd |
| `baseline_inference.py` | LLM agent (GPT-4o, temp=0.0, seed=42). Runs all 6 tasks. **Now install openai.** | ~200 | 4th |

### 3.3 Files to Edit

| File | Changes | Priority |
|---|---|---|
| `server/app.py` | Add `GET /dashboard` and `GET /validation-report` routes | 1st/3rd |
| `requirements.txt` | Add `openai` (only now, for LLM baseline) | 4th |
| `Dockerfile` | `COPY validation/reports/` and `COPY server/dashboard.html` | 1st |
| `README.md` | Final polish: dashboard description, validation suite, measured baseline scores | 2nd |
| `openenv.yaml` | Add dashboard and validation-report to endpoints | 1st |

### 3.4 Dashboard Panels

See spec Section 19 for full specification. Summary:
1. **Training Metrics** — Plotly.js line charts for loss/accuracy with restart markers
2. **Gradient & Weight Heatmap** — color-coded per-layer grid (green/yellow/red/blue)
3. **Action Timeline** — horizontal bars per step, color-coded by type, reward bars
4. **Episode Summary** — task ID, state flags, available actions, grader score

Tech: single HTML file, Plotly.js CDN, native WebSocket, CSS Grid. Zero Docker bloat.

### 3.5 Validation Suite

Run locally (NOT in Docker build). Each script: real training → capture metrics → compare to parametric → assert R² > 0.85 → save plots. Pre-computed reports committed to git and served via `/validation-report`. See spec Section 18.

### 3.6 Tests to Create/Extend

| Test File | Coverage |
|---|---|
| `tests/test_dashboard.py` | `GET /dashboard` returns 200 with HTML containing "Plotly" and "WebSocket" |
| `tests/test_endpoints.py` | Integration: full episode via HTTP (reset→step→grader), verify response schemas |
| `tests/test_baseline_reproducibility.py` | Run baseline twice, assert identical JSON |
| Existing test files | Fill coverage gaps to >80% on every module |

### 3.7 Acceptance Criteria

- [ ] `GET /dashboard` serves HTML that renders in a browser with 4 panels
- [ ] Dashboard connects to WebSocket and updates in real time during a baseline run
- [ ] Validation suite passes all scripts with R² > 0.85 (run locally)
- [ ] Pre-computed validation reports exist in `validation/reports/`
- [ ] `GET /validation-report` serves fidelity data
- [ ] LLM baseline runs, scores higher than heuristic on Tasks 5 and 6 (if implemented)
- [ ] README is complete: all 6 tasks, both baselines, dashboard description, setup instructions
- [ ] `pytest --cov` shows >80% coverage across all modules
- [ ] Final `openenv validate` passes
- [ ] Final Docker build <500MB, starts <60s
- [ ] HF Space redeployed with dashboard + all features

---

## Pre-Submission Gate Checklist

**Every item must be checked before submitting. Failure on any starred (*) item = disqualification.**

### Auto-Validation Gates (*)

- [ ] * **HF Space deploys** — `curl https://<space-url>/health` returns `{"status": "ready", "tasks": N}` with HTTP 200
- [ ] * **HF Space responds to reset** — WebSocket connection to `/ws`, send reset message, receive valid observation
- [ ] * **OpenEnv spec compliance** — `openenv validate` passes (openenv.yaml present, typed models, step/reset/state work)
- [ ] * **Dockerfile builds** — `docker build -t pytorch-debugger .` succeeds
- [ ] * **Docker runs** — `docker run -p 7860:7860 pytorch-debugger` starts and serves on port 7860
- [ ] * **Baseline reproduces** — `python baseline_heuristic.py > run1.json && python baseline_heuristic.py > run2.json && diff run1.json run2.json` produces no output
- [ ] * **3+ tasks with graders** — `GET /tasks` returns ≥3 tasks; `POST /grader` returns score in [0.0, 1.0] after each task completes
- [ ] * **Graders produce varying scores** — different agent behaviors produce different scores (not always same value)

### Required Endpoint Gates (*)

- [ ] * **`GET /tasks`** — returns JSON with task IDs, difficulties, action schema
- [ ] * **`POST /grader`** — returns `{"score": float}` after a completed episode
- [ ] * **`POST /baseline`** — triggers baseline, returns scores for all tasks
- [ ] * **`GET /health`** — returns `{"status": "ready", "tasks": N}`

### Submission Artifacts (*)

- [ ] * **Public GitHub repo** — contains all code, README, requirements, openenv.yaml
- [ ] * **HF Spaces demo link** — deployed, tagged `openenv`, accessible
- [ ] * **README complete** — environment description, action/observation space definitions, task descriptions with difficulty, setup instructions, baseline scores

### Quality Gates (Not DQ, but impact scoring)

- [ ] All typed Pydantic models — no `Dict[str, Any]`
- [ ] `import torch` in every core module — zero `import numpy` in core
- [ ] Context-gated penalty fires correctly (manually tested both paths)
- [ ] Task 5 red herrings present: FC spike, GPU 91%, conv1 near-vanishing, error_log warning
- [ ] Task 6 code fix validation handles whitespace and comment variations
- [ ] Task 6 diagnosis is always `code_bug` regardless of bug variant
- [ ] Grader and reward function are separate modules
- [ ] Step penalty is flat -0.01 (not multiplied by step_count)
- [ ] Episode state is isolated per WebSocket session
- [ ] Test suite passes with >80% coverage
- [ ] Code formatted with black, linted with ruff, imports sorted with isort

### Final Smoke Test Sequence

Run this entire sequence the night before submission:

```bash
# 1. Clean build
docker build --no-cache -t pytorch-debugger .
docker run -d -p 7860:7860 --name smoke-test pytorch-debugger

# 2. Wait for startup
sleep 10
curl -f http://localhost:7860/health || echo "FAIL: health"

# 3. Tasks endpoint
curl -f http://localhost:7860/tasks | python -m json.tool || echo "FAIL: tasks"

# 4. Baseline reproducibility
python baseline_heuristic.py > run1.json 2>/dev/null
python baseline_heuristic.py > run2.json 2>/dev/null
diff run1.json run2.json && echo "PASS: reproducible" || echo "FAIL: non-reproducible"

# 5. Baseline via endpoint
curl -f -X POST http://localhost:7860/baseline | python -m json.tool || echo "FAIL: baseline endpoint"

# 6. Grader via endpoint (after baseline has completed episodes)
curl -f -X POST http://localhost:7860/grader | python -m json.tool || echo "FAIL: grader endpoint"

# 7. OpenEnv validation
openenv validate || echo "FAIL: openenv validate"

# 8. Test suite
pytest tests/ -v --cov=ml_training_debugger --cov-report=term-missing

# 9. Cleanup
docker stop smoke-test && docker rm smoke-test

echo "=== Smoke test complete ==="
```

### If Something Fails at Submission Time

| Failure | Triage |
|---|---|
| HF Space won't deploy | Check Dockerfile CMD, port 7860, build logs. Redeploy. |
| Baseline non-reproducible | Check `torch.manual_seed()` in `reset()`. Check for `random` module usage. |
| Grader returns same score | Check that `sample_scenario` uses different seeds. Check grader logic has branching. |
| `openenv validate` fails | Read error message. Usually missing field in openenv.yaml or wrong model base class. |
| Docker image >500MB | Check `docker images` size. Remove unused deps. Ensure torch is CPU-only. |
| Test coverage <80% | Run `pytest --cov` with `--cov-report=html`. Find uncovered branches. Add targeted tests. |
