# PRD — PyTorch Training Run Debugger

**Product:** OpenEnv RL environment for ML training failure diagnosis
**Hackathon:** Meta PyTorch OpenEnv Hackathon x Scaler School of Technology, Round 1
**Deadline:** April 8, 2026 (submission window opens March 28)
**Runtime:** Python 3.12 · PyTorch CPU-only · openenv-core v0.2.2
**Source of truth:** `ml-training-debugger-spec.md` for all implementation detail beyond this PRD

---

## 1. Overview

### 1.1 What We Are Building

An OpenEnv-compliant reinforcement learning environment where an AI agent receives a snapshot of a broken PyTorch training run and must investigate, diagnose, fix, and verify the failure through a multi-step interactive process. The environment exposes real PyTorch model internals (gradients from `torch.autograd`, weights from `model.state_dict()`) and covers 6 failure scenarios across 3 difficulty tiers.

### 1.2 Problem Being Solved

MLOps teams spend 15-25% of engineer time debugging silent training failures — runs that produce no error, no crash, just bad metrics. Each misdiagnosed restart wastes GPU compute at $2-8/hour/card. The diagnostic process is hard because:

- Multiple symptoms can point to multiple causes simultaneously
- Some bugs produce no error — just mysteriously bad performance
- Fixing the wrong thing wastes hours of compute and restarts
- Static analysis catches some bugs but cannot reason through ambiguous runtime signals

No existing OpenEnv environment covers this domain. The OpenEnv Hub currently contains a demo echo environment and a code execution environment. This fills a genuine gap.

### 1.3 Why This Domain Wins

1. **Strategic alignment** — PyTorch debugging for a Meta PyTorch hackathon. Judges from Meta and Hugging Face will see their own framework as the core subject matter.
2. **Novel reward design** — Context-gated penalties that encode evidence-based reasoning into the reward signal. No existing OpenEnv environment attempts this.
3. **Code-level debugging** — Task 6 requires the agent to read and fix actual PyTorch code. Directly addresses Meta's interest: can an AI agent debug PyTorch?
4. **Ecosystem gap** — Zero competition in the OpenEnv ecosystem for ML training failure diagnosis.

### 1.4 Key Differentiators

| Differentiator | What It Is | Why It Matters |
|---|---|---|
| Context-gated reward shaping | Penalty fires only when agent ignores evidence it already gathered; no penalty for reasonable priors | Encodes evidence-based decision making — a capability no other OpenEnv environment has |
| PyTorch-native internals | Real `torch.nn.Module` models, real `torch.autograd` gradients, real `state_dict()` snapshots | Every model-level observation is grounded in real PyTorch computation, not synthetic data |
| Code-level debugging (Task 6) | Agent reads PyTorch code, identifies buggy line, submits code fix | Tests code understanding, not just metric interpretation — aligned with Meta's core interest |

---

## 2. Target Users

### 2.1 Primary: Hackathon Judges (Meta + Hugging Face Engineers)

**What they evaluate:**
- Real-world utility (30%) — Is this a genuine task? Would someone use this to train/evaluate agents?
- Task & grader quality (25%) — Well-defined tasks, accurate graders, meaningful difficulty progression?
- Environment design (20%) — Clean state management, sensible action/observation spaces, good reward shaping?
- Code quality & spec compliance (15%) — OpenEnv spec, clean structure, typed models, working Dockerfile?
- Creativity & novelty (10%) — Novel domain, interesting mechanics, original approach?

**What impresses them:**
- Real `import torch` in core modules (not numpy wrappers)
- A live dashboard where they can watch an agent investigate in real time
- Deterministic graders that produce different scores for different agent quality levels
- The context-gated penalty — nuanced reward design that goes beyond standard practice

**What disqualifies:**
- HF Space doesn't deploy or respond to `reset()`
- Plagiarized or trivially modified existing environments
- Graders that always return the same score
- No baseline inference script
- Dockerfile doesn't build

### 2.2 Secondary: RL Researchers and Agent Developers

**What they need:**
- A challenging benchmark that differentiates heuristic agents from reasoning-capable ones
- Clear, typed action/observation schemas for agent integration
- Reproducible baseline scores for comparison
- Environments that produce meaningful reward signal across the full trajectory (not just sparse terminal reward)

### 2.3 Tertiary: Auto-Validation System (Phase 1 Gate)

A non-human "user" that must pass before any human judge sees the submission:
- Pings HF Space URL — must return 200 and respond to `reset()`
- Validates `openenv.yaml`, typed models, `step()`/`reset()`/`state()` endpoints
- Runs `docker build` on submitted repo
- Runs baseline script twice — scores must be identical
- Enumerates tasks, runs each grader — scores must be in [0.0, 1.0]

---

## 3. Success Metrics

### 3.1 Evaluation Criteria Targets

| Criterion | Weight | Target Score | How We Hit It |
|---|---|---|---|
| Real-world utility | 30% | 26-30 | ML debugging is a $B+ problem; every PyTorch team encounters these failures; fills a genuine OpenEnv gap |
| Task & grader quality | 25% | 21-25 | 6 tasks (3 MVP), 3 difficulty tiers, deterministic graders, hard tasks challenge frontier models |
| Environment design | 20% | 17-20 | Progressive reveal, context-gated penalties, dynamic `available_actions`, proper episode boundaries |
| Code quality & spec compliance | 15% | 13-15 | Full OpenEnv spec, typed Pydantic models, working Dockerfile + HF Space, two baselines |
| Creativity & novelty | 10% | 9-10 | Context-gated rewards, real PyTorch model internals, code fix task — all new to OpenEnv |
| **Total** | **100%** | **86-100** | |

### 3.2 Quantitative Success Criteria

| Metric | Target | Measurement |
|---|---|---|
| Auto-validation | Pass all 5 gates | `openenv validate` + smoke test sequence |
| Grader score range | Meaningful variance per task | Heuristic baseline ~0.30-0.85 across tasks (not flat) |
| Heuristic-LLM gap | Measurable difference | LLM scores higher than heuristic on Tasks 5 and 6 |
| `reset()` latency | <200ms | Model instantiation + 2 forward passes + parametric curves |
| `step()` latency | <10ms | Action dispatch + reward computation + state update |
| Baseline reproducibility | Bit-exact across runs | `diff run1.json run2.json` produces no output |
| Docker image size | <500MB | PyTorch CPU-only + python:3.12-slim |
| Test coverage | >80% | `pytest --cov` |

### 3.3 Qualitative Success Criteria

- A judge can open `/dashboard`, trigger a baseline run, and understand the agent's reasoning at a glance
- Task 5 (BatchNorm eval mode) visibly differentiates disciplined investigation from red-herring chasing
- Task 6 (code bug) produces a "wow" moment — an agent reading and fixing PyTorch code in front of Meta judges
- The context-gated penalty creates a story: "this agent gathered evidence and then ignored it"

---

## 4. Functional Requirements

> **Complete typed specifications for all data models, actions, observations, tasks, reward components, and error handling are in `ml-training-debugger-spec.md` Sections 10-16.** This section provides a product-level summary.

### 4.1 Agent Interaction Loop

```
reset(task_id) → initial observation (loss curves, config, error log — no gradients/weights/data/code)
     ↓
step(action)   → updated observation + reward + done flag (progressive reveal)
     ↓
  ... repeat ...
     ↓
step(mark_diagnosed) → terminal observation, done=True, episode scored by grader
```

### 4.2 Observation Space Summary

The `MLTrainingObservation` extends `Observation` from openenv-core. Key design:
- **Always visible from reset:** loss/accuracy histories, config, error_log, GPU memory, episode state, available actions
- **Progressively revealed:** gradient stats (real torch.autograd), weight stats (real state_dict), data batch stats, model mode info, code snippets — each populated only after the corresponding `inspect_*` action
- All fields are typed Pydantic models with explicit types. See spec Section 10 for complete field definitions.

### 4.3 Action Space Summary

The `MLTrainingAction` extends `Action` from openenv-core. 14 action types in 3 categories:
- **Investigation** (5): `inspect_gradients`, `inspect_data_batch`, `inspect_model_modes`, `inspect_model_weights`, `inspect_code`
- **Fix** (7): `modify_config`, `add_callback`, `replace_optimizer`, `patch_data_loader`, `fix_model_mode`, `fix_code`, `rollback_checkpoint`
- **Terminal** (2): `restart_run`, `mark_diagnosed`

Dynamic availability: `restart_run` requires `fix_action_taken`, `fix_code` requires `code_inspected`, `mark_diagnosed` disappears after submission. See spec Section 10 for complete action definitions and required fields.

### 4.4 Diagnosis Enum (RootCauseDiagnosis)

Closed set of 6 values. Grader is a single equality check — no fuzzy matching.

| Value | Description |
|---|---|
| `lr_too_high` | Learning rate too large for the architecture |
| `vanishing_gradients` | LR too low or architecture too deep, gradients decay to near-zero |
| `data_leakage` | Validation samples appearing in training batches |
| `overfitting` | Model memorizing training data, failing to generalize |
| `batchnorm_eval_mode` | Model left in eval mode, BatchNorm using running statistics |
| `code_bug` | Bug in the PyTorch training code (Task 6 — always this, regardless of bug variant) |

### 4.5 Reward Function Summary

Per-step signal. **Separate from the grader** (see 4.6). Range: [-1.0, 1.0] hard cap.

| Event | Reward | Gate Condition |
|---|---|---|
| Any step taken | -0.01 | Unconditional, flat constant (never multiplied by step_count) |
| First-time inspection (per type) | +0.05 | Not previously inspected for that type |
| `add_callback` after normal gradients | -0.20 | `gradients_inspected == True AND gradients_were_normal == True` |
| Invalid action | -0.05 | Action not in current `available_actions` |
| Wrong code fix | -0.10 | `fix_code` with incorrect line or replacement |
| Correct diagnosis | +0.50 | `diagnosis == true_root_cause` |
| Wrong diagnosis | -0.30 | `diagnosis != true_root_cause` |
| Convergence after fix+restart | +0.40 | `fix_action_taken AND restart_after_fix AND convergence_confirmed` |

See spec Section 12 for full design rationale.

### 4.6 Grader Function

Returns a single normalized 0.0-1.0 score at episode end. Evaluates `EpisodeState` holistically — checks which key actions were taken, whether the correct fix was applied, whether the diagnosis is correct, and efficiency. **Not a sum of step rewards.** One grader function per task. All graders are deterministic.

Exposed via `POST /grader`. Returns score for the most recently completed episode.

### 4.7 The Six Tasks

| Task | ID | Difficulty | Root Cause | Key Signal | Heuristic Score |
|---|---|---|---|---|---|
| Exploding Gradients | `task_001` | Easy | `lr_too_high` | All layers `is_exploding: True`, NaN in error_log | ~0.85 |
| Vanishing Gradients | `task_002` | Easy | `vanishing_gradients` | Deeper layers `is_vanishing: True`, flat loss | ~0.80 |
| Silent Data Leakage | `task_003` | Medium | `data_leakage` | High val accuracy from epoch 1, `class_overlap_score` 0.68-0.88 | ~0.70 |
| Overfitting | `task_004` | Medium | `overfitting` | Train-val divergence, loss→0.01 while val climbs | ~0.65 |
| BatchNorm Eval Mode | `task_005` | Hard | `batchnorm_eval_mode` | Slow val degradation + compound red herrings | ~0.45 |
| PyTorch Code Bug | `task_006` | Hard | `code_bug` (always) | Anomalous metrics, root cause only visible in code | ~0.30 |

**MVP tasks:** 1, 3, 5 (satisfies the 3-task minimum with easy→medium→hard range).

See spec Section 11 for complete task specifications including fault parameters, red herrings, solution paths, and grader breakdowns.

### 4.8 Baseline Agents

**Rule-based baseline (submission default, `baseline_heuristic.py`):**
- Deterministic decision tree: inspect_gradients → check exploding/vanishing → inspect_data → check leakage → check overfitting → inspect_model_modes → inspect_code → fallback
- No API key required. Bit-exact reproducible.
- Used for Phase 1 auto-validation reproducibility checks.

**LLM baseline (optional, `baseline_inference.py`):**
- GPT-4o at temperature=0.0, seed=42
- Requires `OPENAI_API_KEY` environment variable
- Supplementary demonstration of heuristic vs. reasoning score gap
- Not used for Phase 1 reproducibility — scores reported only after empirical measurement

### 4.9 Required Endpoints

| Endpoint | Method | Required By | Response |
|---|---|---|---|
| `/ws` | WebSocket | OpenEnv framework | Handles `reset`, `step`, `state` messages |
| `/tasks` | GET | Hackathon | Task list with IDs, difficulties, MLTrainingAction JSON schema |
| `/grader` | POST | Hackathon | `{"score": float, "task_id": str, "steps": int}` for last completed episode |
| `/baseline` | POST | Hackathon | Triggers baseline run, returns `{"scores": {"task_001": float, ...}}` |
| `/health` | GET | Hackathon | `{"status": "ready", "tasks": N}` — N is active task count |
| `/dashboard` | GET | Bonus | Live diagnostic dashboard (HTML/JS, Plotly.js via CDN) |
| `/validation-report` | GET | Bonus | Pre-computed PyTorch fidelity reports |

Framework auto-provides: `POST /reset`, `POST /step`, `GET /state`, `GET /schema`, `GET /docs`, `/mcp`.

### 4.10 Error Handling

`step()` must never raise an unhandled exception. All invalid actions return a valid observation with -0.05 penalty and an error note. See spec Section 16 for the complete error handling matrix covering all edge cases (invalid actions, malformed JSON, step before reset, etc.).

---

## 5. Non-Functional Requirements

### 5.1 OpenEnv Spec Compliance

| Requirement | Implementation |
|---|---|
| `openenv.yaml` present | Name, version, description, framework, tags, observation/action space, tasks with IDs+difficulties+max_steps, reward config, endpoints |
| Typed Pydantic models | `MLTrainingAction` extends `Action`, `MLTrainingObservation` extends `Observation`, all fields explicitly typed |
| `step()`/`reset()`/`state()` | Implemented in `MLTrainingEnvironment` extending `Environment` from `openenv.core.env_server.interfaces` |
| `openenv validate` passes | Tested before every submission |

### 5.2 Framework Integration

| Requirement | Implementation |
|---|---|
| `openenv-core` v0.2.2 | `create_app()` returns standard FastAPI instance — **verified** |
| Custom routes compose | `/tasks`, `/grader`, `/baseline`, `/health` added via `@app.get()`/`@app.post()` on the returned FastAPI app |
| Framework-provided routes | `/reset`, `/step`, `/state`, `/ws`, `/schema`, `/docs`, `/mcp` — do not reimplement |
| Factory pattern | `create_app(MLTrainingEnvironment, ...)` takes the class, not an instance |
| Concurrent sessions | `SUPPORTS_CONCURRENT_SESSIONS = True`, session state keyed by session ID |
| Typed client | `client.py` extends `EnvClient` with typed action/observation — used by baseline scripts |

### 5.3 Docker & Deployment

| Requirement | Target |
|---|---|
| Base image | `python:3.12-slim` |
| PyTorch | CPU-only wheel (`--index-url https://download.pytorch.org/whl/cpu`), ~150MB |
| Total image size | <500MB |
| Build time | <5 min (no real training during build; validation reports pre-computed) |
| HF Spaces | Tagged with `openenv`, port 7860 |
| Health check | `/health` returns `{"status": "ready", "tasks": N}` within 60s of container start |

### 5.4 Reproducibility

| Requirement | Implementation |
|---|---|
| Deterministic episodes | `torch.manual_seed(seed)` at every `reset()`, seed derived deterministically from task ID |
| Baseline bit-exact | Rule-based baseline produces identical scores on two consecutive runs |
| Exploit resistance | Parameters randomized per `reset()` from defined ranges; opaque task IDs |
| Grader determinism | Same `EpisodeState` always produces same score |

### 5.5 Performance

| Requirement | Target |
|---|---|
| `reset()` latency | <200ms (model instantiation + 2 forward passes + parametric curves) |
| `step()` latency | <10ms (action dispatch + reward + state update) |
| Memory | <512MB RSS (small CNN ~50K params, no GPU, no large datasets) |

### 5.6 Code Quality

| Requirement | Standard |
|---|---|
| Formatting | black (line length 88) |
| Linting | ruff |
| Import ordering | isort (profile=black) |
| Type hints | Every function signature and return type |
| Tests | pytest, >80% coverage, every module has corresponding test file |
| PyTorch-native | All core computation uses `torch.Tensor`, zero numpy in core modules |

---

## 6. Prioritized Scope

### Tier 1: MVP (Must Ship First)

**Deadline within deadline:** Deploy to HF Spaces by Day 6 (April 2). Everything after is additive.

| Deliverable | Description | DQ Risk if Missing |
|---|---|---|
| Task 1 (`task_001`) | Exploding gradients — easy | Yes (need 3+ tasks) |
| Task 3 (`task_003`) | Silent data leakage — medium | Yes (need 3+ tasks) |
| Task 5 (`task_005`) | BatchNorm eval mode — hard | Yes (need easy→hard range) |
| Context-gated penalty | -0.20 for `add_callback` after `gradients_were_normal` | No (but kills differentiation) |
| Rule-based baseline | `baseline_heuristic.py`, deterministic, no API key | Yes (baseline required) |
| Reward engine | All 7 reward components implemented exactly | Yes (reward logic required) |
| Graders (3) | One per MVP task, 0.0-1.0, deterministic | Yes (graders required) |
| `openenv.yaml` | Full metadata, 3+ tasks listed | Yes (spec compliance) |
| Required endpoints | `/tasks`, `/grader`, `/baseline`, `/health` | Yes (auto-validator checks) |
| Dockerfile | Builds and runs, port 7860 | Yes (auto-validator checks) |
| HF Space | Deployed, tagged `openenv`, responds to `reset()` | Yes (auto-validator pings) |
| README | Environment description, action/observation spaces, task descriptions, setup instructions, baseline scores | Yes (submission requirement) |

### Tier 2: Strongest Differentiator (Add Immediately After MVP)

| Deliverable | Description | Why This Order |
|---|---|---|
| Task 6 (`task_006`) | PyTorch code bug — hard, code-level debugging | Single highest-impact feature for Meta judges |
| Code fix validation | Multi-strategy pipeline (tokenize, AST, semantic patterns) | Required for Task 6 to work with LLM agents |
| Grader for Task 6 | `code_bug` diagnosis, code fix scoring | Completes Task 6 |

### Tier 3: Full Task Coverage (Time Permitting)

| Deliverable | Description |
|---|---|
| Task 2 (`task_002`) | Vanishing gradients — easy (similar to Task 1, fast to implement) |
| Task 4 (`task_004`) | Overfitting — medium (train-val divergence, regularization fix) |
| Graders for Tasks 2 & 4 | Same pattern as existing graders |

### Tier 4: Polish & Extras (Only After Tiers 1-3 Complete)

| Deliverable | Description | Priority Within Tier |
|---|---|---|
| Live dashboard | HTML/JS at `/dashboard`, Plotly.js via CDN, 4-panel layout | 1st — transforms judging experience |
| PyTorch validation suite | 6 scripts proving parametric curves match real training, R² > 0.85 | 2nd — answers "how realistic?" |
| Validation report endpoint | `GET /validation-report` serving pre-computed fidelity plots | With validation suite |
| LLM baseline | `baseline_inference.py`, GPT-4o, measures heuristic-LLM gap | 3rd — supplementary demonstration |

### Implementation Timeline (11 days: March 28 - April 8)

| Days | Focus | Exit Criteria |
|---|---|---|
| 1-2 | Skeleton server + Task 1 end-to-end | `reset()` → `step()` → `grader` works for one task, Docker builds |
| 3-5 | Tasks 3 & 5 + reward engine + baseline | All 3 MVP tasks pass grader, `baseline_heuristic.py` reproduces |
| 6 | **Deploy MVP to HF Spaces** | Auto-validation passes. This is the insurance policy. |
| 7-8 | Task 6 (code debugging) | Code fix validation works for all 4 bug variants |
| 9-10 | Tasks 2 & 4 + dashboard | Full 6-task environment, dashboard shows agent behavior |
| 11 | Polish, README, final smoke test | Submission-ready |

### What We Will NOT Build (Explicit Exclusions)

- No game or toy environments
- No numpy in core modules (torch.Tensor only)
- No free-text diagnosis (closed enum only)
- No grader that sums step rewards (holistic evaluation only)
- No cumulative step penalty (flat -0.01 only, never -0.01 * step_count)
- No accommodation support or non-RL features
- No multi-GPU or CUDA dependencies (CPU-only PyTorch)
