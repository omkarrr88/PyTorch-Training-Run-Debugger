---
title: PyTorch Training Run Debugger
emoji: 🔧
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - pytorch
  - reinforcement-learning
---

# PyTorch Training Run Debugger

An OpenEnv RL environment where AI agents debug broken PyTorch training runs.

Built for the Meta PyTorch OpenEnv Hackathon x Scaler School of Technology, 2026.

[Live Demo](https://ujjwalpardeshi-pytorch-training-debugger.hf.space/dashboard) | [API Health](https://ujjwalpardeshi-pytorch-training-debugger.hf.space/health) | [API Docs](https://ujjwalpardeshi-pytorch-training-debugger.hf.space/docs)

---

## Why I Built This

Every ML engineer has been there: your model trains for hours, doesn't crash, doesn't throw errors, but the loss just won't go down. You stare at TensorBoard, tweak the learning rate, restart, repeat. It's tedious, time-consuming, and hard to teach. I wanted to turn that debugging experience into an RL environment so agents can learn to do it too.

## How It Works

The environment drops the agent into a broken PyTorch training run. The agent sees loss curves, config, and error logs — but not much else. It has to actively investigate (inspect gradients, look at data, check model modes, read the code) to figure out what's wrong.

Once it thinks it knows the problem, it applies a fix, restarts training, and submits a diagnosis. The grader scores the whole episode — not just whether the answer was right, but whether the agent investigated properly before acting.

There are 7 tasks covering common ML failures: exploding/vanishing gradients, data leakage, overfitting, BatchNorm stuck in eval mode, bugs in the training loop, and misconfigured LR schedulers. The hard tasks have red herrings that punish agents for jumping to conclusions.

## What's Under the Hood

- **Real PyTorch, not fake data.** Gradients come from `torch.autograd`, weights from `model.state_dict()`. The env runs actual `torch.nn.Module` models (SimpleCNN, SimpleMLP), does 20 real forward+backward passes per reset, and caches the results.
- **Context-gated rewards.** If an agent adds gradient clipping after already seeing that gradients are normal, it gets penalized. If it does it before inspecting, no penalty. The reward depends on what the agent knows, not just what it does.
- **Code-level debugging.** Task 6 presents buggy Python training loops. The agent reads the code, finds the bug, and submits a fix. Four bug variants: `model.eval()` left in, `.detach()` killing gradients, missing `zero_grad()`, and `inplace=True` on ReLU.
- **Red herrings on hard tasks.** Task 5 plants a suspicious gradient spike and a GPU memory warning. Both are distractions. The real problem is only visible through model mode inspection.

## Tasks

| ID | Difficulty | Root Cause | What Goes Wrong |
|----|-----------|------------|-----------------|
| `task_001` | Easy | `lr_too_high` | Gradients explode, NaN in loss |
| `task_002` | Easy | `vanishing_gradients` | Deeper layers vanish, loss stays flat |
| `task_003` | Medium | `data_leakage` | Suspiciously high val accuracy from epoch 1 |
| `task_004` | Medium | `overfitting` | Train loss drops, val loss climbs |
| `task_005` | Hard | `batchnorm_eval_mode` | Slow degradation, gradient red herrings |
| `task_006` | Hard | `code_bug` | Buggy training loop (4 variants) |
| `task_007` | Hard | `scheduler_misconfigured` | LR decays too aggressively |

Easy tasks have one obvious signal. Medium tasks need multiple inspections. Hard tasks actively mislead you.

## Actions

**Investigate:** `inspect_gradients`, `inspect_data_batch`, `inspect_model_modes`, `inspect_model_weights`, `inspect_code`

**Fix:** `modify_config`, `add_callback`, `patch_data_loader`, `fix_model_mode`, `fix_code`, `replace_optimizer`

**Terminal:** `restart_run` (needs a fix first), `mark_diagnosed` (submit diagnosis)

Actions are dynamic — `fix_code` only unlocks after code inspection, `restart_run` only after a fix.

## Reward Signal

| Event | Reward |
|-------|--------|
| Any step | -0.01 |
| First-time inspection | +0.05 |
| Correct diagnosis | +0.50 |
| Wrong diagnosis | -0.30 |
| Convergence after fix+restart | +0.40 |
| Invalid action | -0.05 |
| Context-gated penalty | -0.20 |

The context-gated penalty fires when: agent inspected gradients, saw they were normal, and still applied gradient clipping. It's a penalty for ignoring evidence.

## Grading

Each task has a holistic grader (separate from the per-step reward) that looks at the full episode: did the agent investigate the right things, apply the correct fix, restart training, and diagnose accurately? Scores are 0-1.

## Baseline Results

| Task | Heuristic | Llama 3.1 8B |
|------|-----------|--------------|
| task_001 (Easy) | 1.00 | 0.60 |
| task_002 (Easy) | 1.00 | 0.05 |
| task_003 (Medium) | 1.00 | 0.40 |
| task_004 (Medium) | 1.00 | 0.60 |
| task_005 (Hard) | 0.80 | 0.38-0.55 |
| task_006 (Hard) | 0.81 | 0.60-1.00 |
| task_007 (Hard) | 0.79 | 0.60 |
| **Average** | **0.91** | **0.52** |

The heuristic is strong because it knows the task structure. An LLM has to figure it out from observations.

## Setup

```bash
# Local
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openenv-core pydantic fastapi uvicorn
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Docker
docker build -t pytorch-debugger .
docker run -p 7860:7860 pytorch-debugger

# Baselines
python3 baseline_heuristic.py
API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o HF_TOKEN=sk-... python3 inference.py
```

## Project Structure

```
ml_training_debugger/
    models.py            - Data models (Action, Observation, EpisodeState)
    scenarios.py         - Task parameter sampling
    pytorch_engine.py    - Real PyTorch models and fault injection
    simulation.py        - 20-epoch training with fault injection
    reward_engine.py     - Per-step reward with context gating
    graders.py           - Per-task holistic scoring
    code_templates.py    - Task 6 bug variants + fix validation
server/
    environment.py       - MLTrainingEnvironment (reset/step/state)
    app.py               - FastAPI app + endpoints
    dashboard.html       - Live diagnostic dashboard (Plotly.js)
inference.py             - LLM agent (OpenAI client, hackathon format)
baseline_heuristic.py    - Rule-based agent (no API key needed)
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tasks` | GET | Task list with action schema |
| `/grader` | POST | Score for last completed episode |
| `/baseline` | POST | Run heuristic on all tasks |
| `/dashboard` | GET | Live diagnostic dashboard |
| `/docs` | GET | Swagger UI |

WebSocket at `/ws` for full episode sessions (reset, step, observe).
