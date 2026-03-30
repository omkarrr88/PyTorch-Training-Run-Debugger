# PyTorch Training Run Debugger

**OpenEnv RL Environment** | Meta PyTorch OpenEnv Hackathon x Scaler School of Technology

An AI agent debugs broken PyTorch training runs by investigating gradients, model weights, data pipelines, and source code to diagnose and fix real ML failure patterns.

---

## The Problem

ML teams spend 15-25% of engineer time debugging silent training failures — runs that produce no error, no crash, just mysteriously bad metrics. Each misdiagnosed restart wastes GPU compute at $2-8/hour/card. The diagnostic process is hard because multiple symptoms point to multiple causes, some bugs produce no error at all, and fixing the wrong thing wastes hours.

No existing OpenEnv environment covers this domain.

## What This Does

The environment recreates the experience of an ML engineer facing a broken training job. The agent receives a snapshot of a failing training run and must:

1. **Investigate** — inspect gradients, data batches, model weights, model modes, and code
2. **Diagnose** — identify the root cause from 7 known ML failure types
3. **Fix** — apply the correct intervention
4. **Verify** — restart training and confirm recovery before submitting

The agent starts with limited information (loss curves, config, error log) and must actively choose what to investigate. Each inspection reveals new data — gradient norms, class overlap scores, model train/eval modes, or buggy source code. This makes it a genuine investigation, not just a classification task.

## What Makes This Different

### Real PyTorch Model Internals

Every gradient comes from real `torch.autograd`. Every weight stat comes from real `model.state_dict()`. The environment instantiates actual `torch.nn.Module` models (SimpleCNN ~50K params, SimpleMLP ~20K params), runs 20 real forward+backward epochs per reset, and extracts real tensor statistics. Not synthetic formulas — real PyTorch computation, cached for instant replay.

### Context-Gated Reward Shaping

Standard RL environments use stateless rewards: "did action X happen?" This environment tracks the agent's information state and conditions penalties on what the agent has already observed.

An agent that adds gradient clipping *before* inspecting gradients follows a reasonable prior — **no penalty**. An agent that inspects gradients, sees they are normal, and *then* adds gradient clipping is ignoring counter-evidence — **-0.20 penalty**.

The gate requires two conditions to be jointly true (`gradients_inspected AND gradients_were_normal`), both of which depend on prior agent actions. This encodes a transferable skill into the reward signal: don't ignore what you've already learned.

### Code-Level Debugging

Task 6 presents actual buggy PyTorch training loops. The agent reads real Python code, identifies the buggy line, and submits a line-by-line fix. Four bug variants: `model.eval()` instead of `model.train()`, `.detach()` killing gradient flow, missing `optimizer.zero_grad()`, and `inplace=True` on ReLU corrupting the computation graph.

Fix validation uses a 4-strategy pipeline: whitespace normalization, token-stream comparison via Python's `tokenize` module, semantic pattern matching, and `ast.parse()` fallback. This handles the messy fixes that LLM agents actually produce (trailing spaces, inline comments, different indentation).

### Red Herring Injection

Task 5 (BatchNorm eval mode) deliberately plants misleading signals: a gradient spike in the FC layer that doesn't cross the exploding threshold, a GPU memory warning at 91%, and near-vanishing gradients in conv1. The real problem is only visible through model mode inspection. This separates agents that follow rigid patterns from agents that can reason through ambiguity.

## Tasks

7 failure scenarios across 3 difficulty tiers, each with configurable difficulty level (1-5):

| ID | Difficulty | Root Cause | What Goes Wrong |
|----|-----------|------------|-----------------|
| `task_001` | Easy | `lr_too_high` | All gradient layers explode, NaN in loss. Direct signal — inspect gradients, reduce LR. |
| `task_002` | Easy | `vanishing_gradients` | Deeper layers show vanishing norms, loss stays flat. Model can't learn. |
| `task_003` | Medium | `data_leakage` | Suspiciously high val accuracy from epoch 1. `class_overlap_score > 0.5` confirms test data leaked into training. Red herring note about "architecture upgrade." |
| `task_004` | Medium | `overfitting` | Train loss drops to near-zero while val loss climbs. Classic memorization pattern. |
| `task_005` | Hard | `batchnorm_eval_mode` | Slow degradation with compound red herrings. Gradients look normal. The real problem: all layers stuck in eval mode. |
| `task_006` | Hard | `code_bug` | Metrics are anomalous but gradients/data/modes look fine. Root cause is in the Python training loop — 4 possible bug variants. |
| `task_007` | Med-Hard | `scheduler_misconfigured` | Training improves initially then stagnates. LR scheduler decays too aggressively (low gamma, small step size). |

### How Difficulty Scales

Easy tasks have one obvious signal (all gradients exploding). Medium tasks require checking multiple sources and ruling out alternatives. Hard tasks deliberately mislead — the most obvious signal is wrong, and the real problem is hidden behind layers of investigation.

## Observation Space

| Field | Type | When Visible |
|-------|------|-------------|
| `training_loss_history` | `list[float]` (20 epochs) | Always |
| `val_accuracy_history` | `list[float]` (20 epochs) | Always |
| `val_loss_history` | `list[float]` (20 epochs) | Always |
| `current_config` | `TrainingConfig` | Always |
| `error_log` | `str` or `null` | Always |
| `gradient_stats` | `list[GradientStats]` | After `inspect_gradients` |
| `model_weight_stats` | `list[ModelWeightStats]` | After `inspect_model_weights` |
| `data_batch_stats` | `DataBatchStats` | After `inspect_data_batch` |
| `model_mode_info` | `dict[str, str]` | After `inspect_model_modes` |
| `code_snippet` | `CodeSnippet` | After `inspect_code` |
| `available_actions` | `list[str]` | Always (dynamic) |
| `episode_state` | `EpisodeState` | Always |

Fields like `gradient_stats`, `data_batch_stats`, `model_mode_info`, and `code_snippet` start as `null` and are only populated after the agent explicitly requests them. The agent must decide what to investigate.

## Action Space

14 action types in 3 categories:

**Investigation** — reveal hidden observation fields:
- `inspect_gradients` — per-layer gradient norms, is_exploding/is_vanishing flags
- `inspect_data_batch` — label distribution, class overlap score, confusion matrix
- `inspect_model_modes` — train/eval mode per layer
- `inspect_model_weights` — weight norms, dead neurons, NaN/Inf detection
- `inspect_code` — the actual Python training loop (Task 6)

**Fix** — apply an intervention:
- `modify_config` — change learning_rate, weight_decay, batch_size, optimizer, etc.
- `add_callback` — add gradient clipping
- `patch_data_loader` — fix data pipeline
- `fix_model_mode` — switch model to train mode
- `fix_code` — fix a specific line of code (requires line number + replacement)
- `replace_optimizer` — switch optimizer

**Terminal** — end the episode:
- `restart_run` — restart training (only available after a fix)
- `mark_diagnosed` — submit diagnosis from 7 possible root causes

Actions are dynamically available based on episode state: `fix_code` requires prior code inspection, `restart_run` requires a fix, `mark_diagnosed` disappears after submission.

## Reward Function

Per-step signal, separate from the grader. Hard cap at [-1.0, 1.0].

| Event | Reward | Condition |
|-------|--------|-----------|
| Any step | -0.01 | Flat, unconditional (encourages efficiency) |
| First-time inspection | +0.05 | Per inspection type, first time only |
| Correct diagnosis | +0.50 | `diagnosis == root_cause` |
| Wrong diagnosis | -0.30 | `diagnosis != root_cause` |
| Convergence after fix+restart | +0.40 | Fix applied, restarted, training recovers |
| Invalid action | -0.05 | Action not in `available_actions` |
| Wrong code fix | -0.10 | `fix_code` with incorrect line/replacement |
| **Context-gated penalty** | **-0.20** | `gradients_inspected AND gradients_were_normal AND action == add_callback` |

The step penalty is flat -0.01 (never multiplied by step count). Investigation bonuses fire once per type. The context-gated penalty requires the agent to have previously inspected gradients and found them normal — it cannot fire before inspection.

## Grading

Each task has a separate grader that evaluates the complete `EpisodeState` at episode end, returning a normalized 0.0-1.0 score. The grader is **not** a sum of step rewards — it's a holistic evaluation of whether the agent investigated correctly, applied the right fix, restarted training, and diagnosed accurately.

Example (Task 5 — BatchNorm Eval):

| Component | Points |
|-----------|--------|
| Inspected gradients | +0.05 |
| Inspected model modes (the revealing action) | +0.05 |
| Fixed model mode | +0.25 |
| Restarted training | +0.30 |
| Correct diagnosis | +0.40 |
| Fell for red herring (add_callback after normal gradients) | -0.20 |

An agent that chases the gradient spike red herring loses 0.20 points. An agent that goes straight to model modes and finds the real problem scores 1.0.

## Baseline Scores

### Heuristic vs LLM Comparison

| Task | Difficulty | Heuristic | Llama 3.3 70B | Llama 3.1 8B |
|------|-----------|-----------|---------------|--------------|
| `task_001` | Easy | **1.00** | 1.00 | 0.60 |
| `task_002` | Easy | **1.00** | 1.00 | 0.05 |
| `task_003` | Medium | **1.00** | 0.40 | 0.40 |
| `task_004` | Medium | **1.00** | 0.45 | 0.60 |
| `task_005` | Hard | **1.00** | 1.00 | 1.00 |
| `task_006` | Hard | **1.00** | — | 0.60-1.00 |
| `task_007` | Med-Hard | **1.00** | — | 0.60 |
| **Average** | | **1.00** | ~0.69* | 0.55 |

*Llama 3.3 70B results are partial (5/7 tasks before rate limit).

**What this tells you:**
- **Model size matters:** 70B scores ~25% higher than 8B. The environment scales with model capability.
- **8B struggles on multi-step tasks:** Task 2 score of 0.05 shows small models can't maintain investigation strategy across many steps.
- **The heuristic baseline is strong** because it was designed with knowledge of the task structure. An agent that doesn't know the structure has to figure it out from observations alone.

### Running Baselines

```bash
# Heuristic (deterministic, no API key, bit-exact reproducible)
python3 baseline_heuristic.py

# LLM (multi-provider support)
python3 baseline_inference.py                       # Groq — Llama 3.3 70B (free)
python3 baseline_inference.py --provider cerebras    # Cerebras — Llama 3.1 8B (free)
python3 baseline_inference.py --provider gemini      # Google Gemini 2.0 Flash
python3 baseline_inference.py --provider openai      # OpenAI GPT-4o

# Run all baselines with comparison table
python3 run_all_baselines.py
```

## API

### HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | `{"status": "ready", "tasks": 7}` |
| `/tasks` | GET | Task list with IDs, difficulties, action schema |
| `/grader` | POST | Score for last completed episode |
| `/baseline` | POST | Run heuristic on all tasks, return scores |
| `/dashboard` | GET | Live 4-panel diagnostic dashboard |
| `/validation-report` | GET | Simulation fidelity report |
| `/curriculum` | GET | Recommended task order (easy to hard, difficulty 1-5) |
| `/leaderboard` | GET | Sorted episode scores |
| `/replay/{id}` | GET | Full action/observation trace for an episode |
| `/schema` | GET | Action/observation JSON schemas |
| `/docs` | GET | Swagger UI |

### WebSocket (Primary Agent Interface)

The WebSocket endpoint at `/ws` maintains session state across a full episode. HTTP endpoints are stateless by framework design.

**Reset** (start an episode):
```json
{"type": "reset", "data": {"task_id": "task_003", "seed": 42}}
```

**Step** (take an action):
```json
{"type": "step", "data": {"action_type": "inspect_gradients"}}
{"type": "step", "data": {"action_type": "modify_config", "target": "learning_rate", "value": 0.001}}
{"type": "step", "data": {"action_type": "mark_diagnosed", "diagnosis": "lr_too_high"}}
```

**Response format:**
```json
{"type": "observation", "data": {"observation": {...}, "reward": 0.04, "done": false}}
```

## Dashboard

A live 4-panel diagnostic dashboard at `/dashboard`:

1. **Training Metrics** — loss/accuracy curves with Plotly.js
2. **Gradient & Weight Heatmap** — per-layer bars, color-coded (green=normal, red=exploding, blue=vanishing)
3. **Action Timeline & Rewards** — step-by-step bars showing reward per action and cumulative reward line
4. **Episode Summary** — state flags, available actions, code snippet (Task 6)

Select a task, click "Run Baseline", and watch the heuristic agent investigate, fix, and diagnose step by step. The charts update live over WebSocket.

## Setup

### Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openenv-core pydantic fastapi uvicorn
pip install pytest pytest-cov pytest-asyncio httpx websockets

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run tests (255 tests, 97% coverage)
pytest tests/ -v --cov=ml_training_debugger

# Run heuristic baseline
python3 baseline_heuristic.py
```

### Docker

```bash
docker build -t pytorch-debugger .
docker run -p 7860:7860 pytorch-debugger
curl http://localhost:7860/health
```

### Smoke Test

```bash
# Verify all critical paths
curl http://localhost:7860/health
curl http://localhost:7860/tasks | python3 -m json.tool
curl -X POST http://localhost:7860/baseline | python3 -m json.tool
curl -X POST http://localhost:7860/grader | python3 -m json.tool

# Reproducibility (must produce no diff)
python3 baseline_heuristic.py > run1.json
python3 baseline_heuristic.py > run2.json
diff run1.json run2.json
```

## Architecture

```
ml_training_debugger/
    models.py            — Pydantic data models (Action, Observation, EpisodeState)
    scenarios.py         — Task parameter sampling (7 tasks, deterministic per seed)
    pytorch_engine.py    — Real PyTorch models, fault injection, gradient/weight extraction
    simulation.py        — 20-epoch real training with parametric fallback
    reward_engine.py     — 7-component per-step reward with context gating
    graders.py           — Per-task holistic 0.0-1.0 scoring
    code_templates.py    — Task 6 bug variants + 4-strategy fix validation
    client.py            — Typed client extending EnvClient

server/
    environment.py       — MLTrainingEnvironment (reset/step/state)
    app.py               — FastAPI + custom endpoints
    dashboard.html       — Live Plotly.js diagnostic dashboard

tests/                   — 255 tests, 97% coverage
baseline_heuristic.py    — Rule-based agent (deterministic, no API key)
baseline_inference.py    — LLM agent (Groq/Cerebras/Gemini/OpenAI)
```

**Key design decisions:**
- **Grader is separate from reward function.** `reward_engine.py` returns a float per step for RL training signal. `graders.py` returns a holistic 0.0-1.0 score at episode end. They are different modules with different purposes.
- **Task IDs are opaque.** `task_001` through `task_007` — the agent cannot infer the diagnosis from the ID.
- **Task 6 diagnosis is always `code_bug`.** Regardless of which bug variant (eval_mode, detach_loss, zero_grad_missing, inplace_relu), the correct diagnosis is `code_bug`.
- **Dual model architectures.** SimpleCNN and SimpleMLP are randomly selected per episode, testing agent robustness to architecture variation.
- **Session isolation.** Each WebSocket connection gets its own environment instance with independent state.
- **`step()` never raises.** All invalid actions return a valid observation with -0.05 penalty and an error note.

### Technical Stack

- Python 3.12 · PyTorch 2.5.1 CPU-only · openenv-core v0.2.2
- `import torch` in every core module — zero numpy in core
- Typed Pydantic v2 models everywhere — no `Dict[str, Any]`
- Deterministic reproducibility via `torch.manual_seed()` at every reset
- Docker image: 885MB (multi-stage build, `strip --strip-unneeded`, transitive dep cleanup)

### Validation Suite

8/8 validation checks pass. Real PyTorch 20-epoch mini-training with fault injection. Each fault type is validated with behavioral checks (gradient detection, loss patterns, model mode, code fix acceptance). Both SimpleCNN and SimpleMLP architectures verified. Results served live at `GET /validation-report`.

## Walkthrough: Solving Task 5 (Hard)

This is the most interesting task because it has red herrings designed to mislead.

**What the agent sees on reset:**
- Loss oscillates between 2.1-2.5, never converging
- Val accuracy stuck at ~0.15
- Error log mentions GPU memory at 91%

**Step 1: Inspect gradients**
```
conv1: mean_norm=0.15, Normal
conv2: mean_norm=5.2, Normal
conv3: mean_norm=0.8, Normal
fc:    mean_norm=1.3, Normal (slight spike — the red herring)
```
All layers normal. `gradients_were_normal` is now True.

**Step 2: Inspect data** — class overlap 0.0, data is clean.

**Step 3: Inspect model modes**
```
conv1: "eval"  ← Problem found!
bn1:   "eval"
conv2: "eval"
bn2:   "eval"
fc:    "eval"
```
All layers stuck in eval mode. BatchNorm is using running statistics instead of batch statistics during training.

**Step 4: Fix model mode** — switches all layers to train mode.

**Step 5: Restart training** — convergence confirmed.

**Step 6: Diagnose `batchnorm_eval_mode`** — correct. Score: 1.0.

**What would have gone wrong:**
If the agent had seen the FC gradient spike and called `add_callback` (gradient clipping), it would have received -0.20 context-gated penalty — because it already knew gradients were normal. The penalty only fires when both `gradients_inspected=True` and `gradients_were_normal=True`. Before inspection, the same action would have no penalty.

## Research Summary

See [PAPER.md](PAPER.md) — "Context-Gated Reward Shaping for Evidence-Based ML Debugging"

Core claim: by conditioning penalties on the agent's accumulated information state (not just action outcomes), we create environments that reward systematic investigation over pattern-matching — a capability with direct transfer value to real-world MLOps debugging.

---

*Built for the Meta PyTorch OpenEnv Hackathon x Scaler School of Technology, 2026.*
