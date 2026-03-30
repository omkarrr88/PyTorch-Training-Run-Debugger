---
name: ML Debugger Project Overview
description: PyTorch Training Run Debugger — OpenEnv RL environment for Meta PyTorch Hackathon. Core architecture, 7 tasks, dual model, real training, key modules.
type: project
---

## What This Is

A complete OpenEnv RL environment where an AI agent debugs broken PyTorch training runs. Built for the **Meta PyTorch OpenEnv Hackathon x Scaler School of Technology** (Round 1 deadline: April 8, 2026).

**Runtime**: Python 3.12 · PyTorch 2.5.1 CPU-only · openenv-core v0.2.2

## Architecture

```
server/app.py          → FastAPI app via create_app() from openenv-core
server/environment.py  → MLTrainingEnvironment(Environment) — reset(), step(), state
server/_baseline_results.py → Shared grader result storage
server/dashboard.html  → Live 4-panel Plotly.js dashboard

ml_training_debugger/
  models.py            → All Pydantic models (Action, Observation, EpisodeState, etc.)
  scenarios.py         → ScenarioParams + sample_scenario() — 7 tasks, model_type, difficulty_level
  pytorch_engine.py    → SimpleCNN + SimpleMLP, fault injection, gradient/weight extraction, run_real_training() with caching
  simulation.py        → Calls run_real_training() for curves, parametric fallback
  reward_engine.py     → 7-component reward function (per-step RL signal)
  graders.py           → Per-task grader functions (0.0-1.0 holistic score at episode end)
  code_templates.py    → Task 6 code bug templates + multi-strategy fix validation
  client.py            → MLTrainingEnvClient extending GenericEnvClient
```

## The 7 Tasks

| Task | Root Cause | Difficulty | Heuristic Score |
|------|-----------|------------|-----------------|
| task_001 | lr_too_high | Easy | 1.00 |
| task_002 | vanishing_gradients | Easy | 1.00 |
| task_003 | data_leakage | Medium | 1.00 |
| task_004 | overfitting | Medium | 0.45 |
| task_005 | batchnorm_eval_mode | Hard | 1.00 |
| task_006 | code_bug (4 variants) | Hard | 1.00 |
| task_007 | scheduler_misconfigured | Med-Hard | 1.00 |

## Model Architectures (Dual)
- **SimpleCNN**: 3-layer CNN with BatchNorm, ~50K params (used for task_005, task_006)
- **SimpleMLP**: 3-layer MLP with BatchNorm1d, ~20K params
- Randomly selected per task/seed via `_pick_model_type(rng)`

## Real Training Curves
- `run_real_training()` in pytorch_engine.py runs 20 real forward+backward epochs
- Cached per (task_id, seed, model_type) — first call ~2s, subsequent instant
- Replaces parametric formulas — judges see real training dynamics, not `torch.exp()`

## Key Endpoints

- `GET /health` → `{"status": "ready", "tasks": 7}`
- `GET /tasks` → Task list with action schema
- `POST /grader` → Score after completed episode
- `POST /baseline` → Run heuristic baseline, return all scores
- `GET /dashboard` → Live diagnostic dashboard (Plotly.js)
- `GET /validation-report` → Pre-computed fidelity report (8/8 pass)
- `GET /curriculum` → Recommended task order with difficulty scaling
- `GET /leaderboard` → Sorted episode scores
- `GET /replay/{episode_id}` → Episode trace
- `WS /ws` → Primary agent interface
- Framework: `/reset`, `/step`, `/state`, `/schema`, `/docs`

## WebSocket Message Format

- Reset (select task): `{"type": "reset", "data": {"task_id": "task_003", "seed": 42}}`
- Reset (default): `{"type": "reset"}`
- Step: `{"type": "step", "data": {"action_type": "inspect_gradients"}}`
- Response: `{"type": "observation", "data": {"observation": {...}, "reward": float, "done": bool}}`

## Key Design Decisions

- **Grader ≠ Reward**: graders.py (holistic 0.0-1.0) vs reward_engine.py (per-step float)
- **Task IDs are opaque**: task_001-task_007
- **Task 6 diagnosis is ALWAYS `code_bug`** regardless of variant
- **Context-gated penalty**: -0.20 fires ONLY when `gradients_inspected=True AND gradients_were_normal=True`
- **Step penalty is flat -0.01** (never multiplied by step_count)
- **Difficulty scaling**: 1-5 via `difficulty_level` parameter in reset()
- **Confusion matrix** included in data batch stats
