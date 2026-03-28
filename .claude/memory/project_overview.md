---
name: ML Debugger Project Overview
description: PyTorch Training Run Debugger — OpenEnv RL environment for Meta PyTorch Hackathon. Core architecture, 6 tasks, key modules, and how they connect.
type: project
---

## What This Is

A complete OpenEnv RL environment where an AI agent debugs broken PyTorch training runs. Built for the **Meta PyTorch OpenEnv Hackathon x Scaler School of Technology** (Round 1 deadline: April 8, 2026).

**Runtime**: Python 3.12 · PyTorch CPU-only · openenv-core v0.2.2

## Architecture

```
server/app.py          → FastAPI app via create_app() from openenv-core
server/environment.py  → MLTrainingEnvironment(Environment) — reset(), step(), state
server/_baseline_results.py → Shared grader result storage across endpoints

ml_training_debugger/
  models.py            → All Pydantic models (Action, Observation, EpisodeState, etc.)
  scenarios.py         → ScenarioParams dataclass + sample_scenario(task_id, seed)
  pytorch_engine.py    → SimpleCNN model, fault injection, gradient/weight extraction
  simulation.py        → Parametric curve generation (loss/accuracy histories) — all torch ops
  reward_engine.py     → 7-component reward function (per-step RL signal)
  graders.py           → Per-task grader functions (0.0-1.0 holistic score at episode end)
  code_templates.py    → Task 6 code bug templates + multi-strategy fix validation
  client.py            → MLTrainingEnvClient extending GenericEnvClient
```

## The 6 Tasks

| Task | Root Cause | Difficulty | Heuristic Score |
|------|-----------|------------|-----------------|
| task_001 | lr_too_high (exploding gradients) | Easy | 1.00 |
| task_002 | vanishing_gradients | Easy | 1.00 |
| task_003 | data_leakage (class_overlap_score) | Medium | 1.00 |
| task_004 | overfitting (train-val divergence) | Medium | 1.00 |
| task_005 | batchnorm_eval_mode (red herrings) | Hard | 0.35 |
| task_006 | code_bug (4 variants) | Hard | 1.00 |

## Key Endpoints

- `GET /health` → `{"status": "ready", "tasks": 6}`
- `GET /tasks` → Task list with action schema
- `POST /grader` → Score after completed episode
- `POST /baseline` → Run heuristic baseline, return all scores
- `GET /dashboard` → Live diagnostic dashboard (Plotly.js)
- `GET /validation-report` → Pre-computed fidelity report
- `WS /ws` → Primary agent interface (framework-provided)
- Framework also provides: `/reset`, `/step`, `/state`, `/schema`, `/docs`

## WebSocket Message Format (Critical!)

- Reset: `{"type": "reset"}` — NO extra fields (task_id NOT accepted via WS, defaults to task_001)
- Step: `{"type": "step", "data": {"action_type": "inspect_gradients"}}` — use `"data"` NOT `"action"`
- HTTP step wraps differently: `POST /step {"action": {"action_type": "..."}}`

## Key Design Decisions

- **Grader ≠ Reward**: `graders.py` (holistic 0.0-1.0 at episode end) vs `reward_engine.py` (per-step float)
- **Task IDs are opaque**: `task_001`-`task_006` — agent can't infer diagnosis from ID
- **Task 6 diagnosis is ALWAYS `code_bug`** regardless of bug variant (eval_mode, detach_loss, etc.)
- **Context-gated penalty**: -0.20 fires ONLY when `gradients_inspected=True AND gradients_were_normal=True` then `add_callback`
- **Step penalty is flat -0.01** (never multiplied by step_count)
