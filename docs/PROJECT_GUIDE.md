# PyTorch Training Run Debugger — Complete Project Guide

## What Is This?

A game where an AI agent plays detective to fix broken PyTorch training runs. The agent sees a failing training run, investigates clues (gradients, data, code), applies a fix, and submits a diagnosis. Built as an [OpenEnv](https://github.com/openenv) RL environment for the **Meta PyTorch OpenEnv Hackathon**.

---

## How a Game Works

```
1. Agent receives a broken training run (loss curves, config, error log)
2. Agent investigates (inspect gradients, data, weights, model modes, code)
3. Agent applies a fix (reduce LR, patch data, fix code, etc.)
4. Agent restarts training and confirms recovery
5. Agent submits diagnosis ("the problem was lr_too_high")
6. Grader scores the agent 0.0 to 1.0
```

---

## The 7 Tasks

| Task | Problem | Difficulty | Root Cause | Key Clue |
|------|---------|-----------|------------|----------|
| `task_001` | Gradients explode | Easy | `lr_too_high` | All layers `is_exploding: true` |
| `task_002` | Gradients vanish | Easy | `vanishing_gradients` | Deep layers `is_vanishing: true` |
| `task_003` | Test data leaked into training | Medium | `data_leakage` | `class_overlap_score > 0.5` |
| `task_004` | Model memorizes, doesn't learn | Medium | `overfitting` | Train loss drops, val loss rises |
| `task_005` | BatchNorm stuck in eval mode | Hard | `batchnorm_eval_mode` | Model modes show "eval" + red herrings |
| `task_006` | Bug in Python training code | Hard | `code_bug` | Bug visible in code snippet |
| `task_007` | LR scheduler decays too fast | Medium-Hard | `scheduler_misconfigured` | Early progress then stagnation |

---

## Reward System

Every action earns or costs points (capped at -1.0 to 1.0):

| Event | Reward | When |
|-------|--------|------|
| Any step taken | **-0.01** | Always (encourages efficiency) |
| First-time inspection | **+0.05** | Once per inspection type |
| Correct diagnosis | **+0.50** | Diagnosis matches root cause |
| Wrong diagnosis | **-0.30** | Diagnosis doesn't match |
| Fix works + training recovers | **+0.40** | After fix + restart + convergence |
| Invalid action | **-0.05** | Action not available |
| Wrong code fix | **-0.10** | `fix_code` with wrong line/replacement |
| **Context-gated penalty** | **-0.20** | Inspected gradients, saw they're normal, then added gradient clipping anyway |

### The Context-Gated Penalty (Core Innovation)

- Agent checks gradients -> finds them **normal** -> adds gradient clipping = **-0.20 penalty** (ignoring evidence)
- Agent adds gradient clipping **before** checking gradients = **no penalty** (reasonable prior)

This teaches: *don't ignore what you've already learned*.

---

## Architecture

```
ml_training_debugger/          # Core logic
  models.py                    # All data types (Pydantic)
  scenarios.py                 # Creates the 7 tasks with random params
  pytorch_engine.py            # Real PyTorch model + fault injection
  simulation.py                # Loss/accuracy curve generation
  reward_engine.py             # Per-step reward calculation
  graders.py                   # Final 0.0-1.0 scoring per task
  code_templates.py            # Buggy code for Task 6
  client.py                    # Client for connecting to the environment

server/                        # Web server
  app.py                       # FastAPI + all endpoints
  environment.py               # Game logic (reset, step, state)

tests/                         # 183 tests, 97% coverage
baseline_heuristic.py          # Rule-based agent (deterministic)
baseline_inference.py          # LLM agent (Llama/GPT-4o)
```

---

## API Endpoints

### GET /health

Server status check.

**Response:**
```json
{
  "status": "ready",
  "tasks": 7
}
```

---

### GET /tasks

List all available tasks with action schema.

**Response:**
```json
[
  {
    "id": "task_001",
    "difficulty": "easy",
    "max_steps": 20,
    "action_schema": {
      "title": "MLTrainingAction",
      "type": "object",
      "properties": {
        "action_type": { "type": "string" },
        "target": { "type": ["string", "null"] },
        "value": { "type": ["number", "integer", "string", "null"] },
        "diagnosis": { "type": ["string", "null"] },
        "line": { "type": ["integer", "null"] },
        "replacement": { "type": ["string", "null"] }
      },
      "required": ["action_type"]
    }
  }
]
```

---

### POST /baseline

Run the heuristic baseline agent on all 7 tasks.

**Response:**
```json
{
  "scores": {
    "task_001": 1.00,
    "task_002": 1.00,
    "task_003": 1.00,
    "task_004": 0.45,
    "task_005": 0.35,
    "task_006": 1.00,
    "task_007": 1.00
  }
}
```

Returns `409` if baseline is already running.

---

### POST /grader

Get the grader score for the last completed episode.

**Query params:** `session_id` (optional)

**Response:**
```json
{
  "score": 0.85,
  "task_id": "task_001",
  "steps": 5
}
```

If no episode completed:
```json
{
  "score": null,
  "error": "no_completed_episode"
}
```

---

### GET /dashboard

Live diagnostic dashboard (HTML page with Plotly.js charts). Open in a browser.

**Panels:**
1. Training metrics (loss/accuracy curves)
2. Gradient & weight heatmap
3. Action timeline with rewards
4. Episode summary with state flags

---

### GET /validation-report

Pre-computed fidelity report comparing parametric curves to real PyTorch training runs.

---

### GET /curriculum

Recommended task order for progressive training (easy to hard, 3 difficulty levels each).

**Response:**
```json
{
  "curriculum": [
    { "task_id": "task_001", "difficulty": "easy", "difficulty_level": 1, "max_steps": 20 },
    { "task_id": "task_001", "difficulty": "easy", "difficulty_level": 3, "max_steps": 20 },
    { "task_id": "task_001", "difficulty": "easy", "difficulty_level": 5, "max_steps": 20 }
  ],
  "total_episodes": 21
}
```

---

### GET /leaderboard

Sorted episode scores from baseline runs.

**Response:**
```json
{
  "entries": [
    { "score": 1.00, "task_id": "task_001", "steps": 5, "episode_id": "baseline_task_001" }
  ],
  "total": 7
}
```

---

### GET /replay/{episode_id}

Full action/observation trace for a completed episode.

**Response:**
```json
{
  "episode_id": "baseline_task_001",
  "score": 1.00,
  "task_id": "task_001",
  "steps": 5
}
```

---

## WebSocket Interface (Primary Agent Interface)

**Endpoint:** `ws://localhost:7860/ws`

This is the main way agents interact with the environment. HTTP endpoints are stateless — WebSocket maintains session state across a full episode.

### Reset (Start New Episode)

**Send:**
```json
{
  "type": "reset",
  "seed": 42,
  "kwargs": {
    "task_id": "task_003",
    "difficulty_level": 3
  }
}
```

Without `kwargs`, defaults to `task_001`.

**Receive:**
```json
{
  "type": "observation",
  "observation": {
    "run_id": "ep_12345",
    "framework": "pytorch",
    "epoch": 20,
    "training_loss_history": [2.3, 2.1, 1.9, ...],
    "val_loss_history": [2.4, 2.2, 2.0, ...],
    "val_accuracy_history": [0.3, 0.35, 0.4, ...],
    "gradient_stats": [],
    "model_weight_stats": null,
    "data_batch_stats": null,
    "model_mode_info": null,
    "code_snippet": null,
    "current_config": {
      "learning_rate": 0.001,
      "weight_decay": 0.0001,
      "batch_size": 64,
      "hidden_dim": 64,
      "num_layers": 3,
      "optimizer": "adam",
      "dropout_rate": 0.0,
      "gradient_clip_norm": null
    },
    "error_log": null,
    "gpu_memory_used_gb": 6.2,
    "gpu_memory_total_gb": 16.0,
    "available_actions": [
      "inspect_gradients",
      "inspect_data_batch",
      "inspect_model_modes",
      "inspect_model_weights",
      "inspect_code",
      "modify_config",
      "add_callback",
      "replace_optimizer",
      "patch_data_loader",
      "fix_model_mode",
      "mark_diagnosed"
    ],
    "episode_state": {
      "step_count": 0,
      "gradients_inspected": false,
      "gradients_were_normal": false,
      "data_inspected": false,
      "model_modes_inspected": false,
      "model_weights_inspected": false,
      "code_inspected": false,
      "fix_action_taken": false,
      "restart_after_fix": false,
      "diagnosis_submitted": false,
      "actions_taken": []
    },
    "notes": null,
    "done": false,
    "reward": null,
    "metadata": {}
  }
}
```

### Step (Take an Action)

**Investigation actions** (no extra fields needed):
```json
{"type": "step", "action": {"action_type": "inspect_gradients"}}
{"type": "step", "action": {"action_type": "inspect_data_batch"}}
{"type": "step", "action": {"action_type": "inspect_model_modes"}}
{"type": "step", "action": {"action_type": "inspect_model_weights"}}
{"type": "step", "action": {"action_type": "inspect_code"}}
```

**Fix actions:**
```json
{"type": "step", "action": {"action_type": "modify_config", "target": "learning_rate", "value": 0.001}}
{"type": "step", "action": {"action_type": "add_callback"}}
{"type": "step", "action": {"action_type": "replace_optimizer"}}
{"type": "step", "action": {"action_type": "patch_data_loader"}}
{"type": "step", "action": {"action_type": "fix_model_mode"}}
{"type": "step", "action": {"action_type": "fix_code", "line": 5, "replacement": "model.train()"}}
```

**Terminal actions:**
```json
{"type": "step", "action": {"action_type": "restart_run"}}
{"type": "step", "action": {"action_type": "mark_diagnosed", "diagnosis": "lr_too_high"}}
```

**Receive (after each step):**
```json
{
  "type": "observation",
  "observation": {
    "...same structure as reset response...",
    "gradient_stats": [
      {
        "layer_name": "conv1",
        "norm_history": [0.5, 0.6, 0.7],
        "mean_norm": 51.1,
        "max_norm": 98.3,
        "is_exploding": true,
        "is_vanishing": false
      }
    ],
    "episode_state": {
      "step_count": 1,
      "gradients_inspected": true,
      "actions_taken": ["inspect_gradients"]
    },
    "done": false,
    "reward": 0.04
  }
}
```

When `done: true`, the episode is over.

---

## All 14 Action Types

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `inspect_gradients` | none | View per-layer gradient stats |
| `inspect_data_batch` | none | View data batch statistics |
| `inspect_model_modes` | none | View train/eval mode per layer |
| `inspect_model_weights` | none | View per-layer weight stats |
| `inspect_code` | none | View source code (Task 6) |
| `modify_config` | `target`, `value` | Change a hyperparameter |
| `add_callback` | none | Add gradient clipping callback |
| `replace_optimizer` | none | Switch optimizer |
| `patch_data_loader` | none | Fix data pipeline |
| `fix_model_mode` | none | Switch model to train mode |
| `fix_code` | `line`, `replacement` | Fix a line of code |
| `restart_run` | none | Restart training (requires fix first) |
| `mark_diagnosed` | `diagnosis` | Submit final diagnosis |
| `rollback_checkpoint` | none | Rollback to checkpoint |

### Valid `target` values for modify_config
`learning_rate`, `weight_decay`, `batch_size`, `hidden_dim`, `num_layers`, `optimizer`, `dropout_rate`, `gradient_clip_norm`

### Valid `diagnosis` values for mark_diagnosed
`lr_too_high`, `vanishing_gradients`, `data_leakage`, `overfitting`, `batchnorm_eval_mode`, `code_bug`, `scheduler_misconfigured`

---

## Dynamic Action Availability

Actions appear/disappear based on episode state:

| Action | Available When |
|--------|---------------|
| `fix_code` | Only after `inspect_code` (code_inspected = true) |
| `restart_run` | Only after a fix action (fix_action_taken = true) |
| `rollback_checkpoint` | Only after restart (restart_after_fix = true) |
| `mark_diagnosed` | Only while diagnosis_submitted = false |

---

## Observation Fields — Progressive Reveal

On reset, the agent sees loss curves, config, and error log. Everything else is `null` until inspected:

| Field | Starts As | Populated After |
|-------|-----------|----------------|
| `training_loss_history` | 20 floats | Always visible |
| `val_accuracy_history` | 20 floats | Always visible |
| `val_loss_history` | 20 floats | Always visible |
| `current_config` | Full config | Always visible |
| `error_log` | String or null | Always visible |
| `gradient_stats` | `[]` | `inspect_gradients` |
| `model_weight_stats` | `null` | `inspect_model_weights` |
| `data_batch_stats` | `null` | `inspect_data_batch` |
| `model_mode_info` | `null` | `inspect_model_modes` |
| `code_snippet` | `null` | `inspect_code` |

---

## Data Types

### GradientStats (per layer)
```json
{
  "layer_name": "conv1",
  "norm_history": [0.5, 0.6, 0.7],
  "mean_norm": 12.5,
  "max_norm": 25.3,
  "is_exploding": true,
  "is_vanishing": false
}
```
- Exploding: `mean_norm > 10.0`
- Vanishing: `mean_norm < 0.000001`

### ModelWeightStats (per layer)
```json
{
  "layer_name": "conv1",
  "weight_norm": 1.234,
  "weight_mean": 0.001,
  "weight_std": 0.05,
  "weight_min": -0.15,
  "weight_max": 0.16,
  "dead_neuron_pct": 0.0,
  "has_nan": false,
  "has_inf": false
}
```

### DataBatchStats
```json
{
  "label_distribution": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25},
  "feature_mean": 0.5,
  "feature_std": 0.2,
  "null_count": 0,
  "class_overlap_score": 0.15,
  "batch_size": 64,
  "duplicate_ratio": 0.0,
  "confusion_matrix": [[10, 2, 1], [1, 9, 3], [2, 1, 11]]
}
```

### CodeSnippet (Task 6 only)
```json
{
  "code": "import torch\nimport torch.nn as nn\n...",
  "filename": "train.py",
  "line_count": 50,
  "imports": ["torch", "torch.nn", "torch.optim"],
  "hint": "Look for .detach() preventing gradient flow"
}
```

### EpisodeState
```json
{
  "step_count": 0,
  "gradients_inspected": false,
  "gradients_were_normal": false,
  "data_inspected": false,
  "model_modes_inspected": false,
  "model_weights_inspected": false,
  "code_inspected": false,
  "fix_action_taken": false,
  "restart_after_fix": false,
  "diagnosis_submitted": false,
  "actions_taken": []
}
```

---

## Grading Breakdown (per task)

Each task has its own grader that scores 0.0 to 1.0 based on what the agent did:

### Task 1 — Exploding Gradients
| Component | Points |
|-----------|--------|
| Inspected gradients | +0.05 |
| Applied config fix | +0.20 |
| Restarted training | +0.35 |
| Correct diagnosis (`lr_too_high`) | +0.40 |

### Task 2 — Vanishing Gradients
| Component | Points |
|-----------|--------|
| Inspected gradients | +0.05 |
| Applied config fix | +0.20 |
| Restarted training | +0.35 |
| Correct diagnosis (`vanishing_gradients`) | +0.40 |

### Task 3 — Data Leakage
| Component | Points |
|-----------|--------|
| Inspected data | +0.05 |
| Patched data loader | +0.30 |
| Restarted training | +0.30 |
| Correct diagnosis (`data_leakage`) | +0.35 |

### Task 4 — Overfitting
| Component | Points |
|-----------|--------|
| Inspected data | +0.05 |
| Applied fix (config or callback) | +0.25 |
| Restarted training | +0.30 |
| Correct diagnosis (`overfitting`) | +0.40 |

### Task 5 — BatchNorm Eval Mode (with red herrings)
| Component | Points |
|-----------|--------|
| Inspected gradients | +0.05 |
| Inspected model modes | +0.05 |
| **Fell for red herring** (add_callback after normal gradients) | **-0.20** |
| Fixed model mode | +0.25 |
| Restarted training | +0.30 |
| Correct diagnosis (`batchnorm_eval_mode`) | +0.40 |

### Task 6 — Code Bug
| Component | Points |
|-----------|--------|
| Inspected code | +0.05 |
| Fixed code correctly | +0.30 |
| Restarted training | +0.25 |
| Correct diagnosis (`code_bug`) | +0.40 |

### Task 7 — Scheduler Misconfigured
| Component | Points |
|-----------|--------|
| Inspected gradients | +0.05 |
| Inspected data | +0.05 |
| Applied config fix | +0.25 |
| Restarted training | +0.25 |
| Correct diagnosis (`scheduler_misconfigured`) | +0.40 |

---

## Baseline Scores

| Task | Heuristic | Llama 3.3 70B | Llama 3.1 8B |
|------|-----------|---------------|--------------|
| task_001 | **1.00** | 1.00 | 0.60 |
| task_002 | **1.00** | 1.00 | 0.05 |
| task_003 | **1.00** | 0.40 | 0.40 |
| task_004 | 0.45 | 0.45 | **0.60** |
| task_005 | **1.00** | 1.00 | 1.00 |
| task_006 | **1.00** | — | 0.60-1.00 |
| task_007 | **1.00** | — | 0.60 |
| **Average** | **0.92** | ~0.69 | 0.55 |

---

## Walkthrough: Solving Task 1 (Exploding Gradients)

```
Step 1: Reset
  Send:    {"type": "reset", "kwargs": {"task_id": "task_001"}}
  See:     Loss history going to infinity, error_log says "NaN at epoch 12"

Step 2: Inspect gradients
  Send:    {"type": "step", "action": {"action_type": "inspect_gradients"}}
  See:     All layers is_exploding: true, mean_norm > 10.0
  Reward:  +0.04 (-0.01 step + 0.05 investigation)

Step 3: Reduce learning rate
  Send:    {"type": "step", "action": {"action_type": "modify_config", "target": "learning_rate", "value": 0.001}}
  Reward:  -0.01 (step penalty)

Step 4: Restart training
  Send:    {"type": "step", "action": {"action_type": "restart_run"}}
  See:     Convergence detected!
  Reward:  +0.39 (-0.01 step + 0.40 convergence)

Step 5: Submit diagnosis
  Send:    {"type": "step", "action": {"action_type": "mark_diagnosed", "diagnosis": "lr_too_high"}}
  See:     done: true
  Reward:  +0.49 (-0.01 step + 0.50 correct diagnosis)

Grader score: 1.0 (perfect)
```

---

## Walkthrough: Task 5 Trap (Red Herring)

```
Step 1: Reset task_005
Step 2: Inspect gradients
  -> FC layer has a spike (mean_norm=4.2, but is_exploding: false)
  -> gradients_were_normal is set to TRUE (nothing actually exploding)

Step 3 (BAD): Add gradient clipping
  -> Reward: -0.21 (-0.01 step - 0.20 context-gated penalty!)
  -> Agent IGNORED the evidence that gradients were normal

Step 3 (GOOD): Inspect model modes instead
  -> Sees all layers in "eval" mode — that's the real problem!

Step 4: Fix model mode
Step 5: Restart training
Step 6: Diagnose batchnorm_eval_mode -> correct!
```

---

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install pytest pytest-cov

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Test
pytest tests/ -v --cov=ml_training_debugger
curl http://localhost:7860/health
curl http://localhost:7860/tasks | python3 -m json.tool
curl -X POST http://localhost:7860/baseline | python3 -m json.tool

# Docker
docker build -t pytorch-debugger .
docker run -p 7860:7860 pytorch-debugger
```

---

## Tech Stack

| Component | Purpose |
|-----------|---------|
| Python 3.12 | Runtime |
| PyTorch (CPU-only) | Real neural networks, real gradients |
| FastAPI | Web server |
| OpenEnv | RL environment framework (step/reset/state API) |
| Pydantic v2 | Typed data models |
| Plotly.js | Dashboard charts |
| Docker | Containerized deployment |
