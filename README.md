# PyTorch Training Run Debugger

**OpenEnv RL Environment** — Meta PyTorch OpenEnv Hackathon x Scaler School of Technology, Round 1

An AI agent debugs broken PyTorch training runs by investigating gradients, model weights, data pipelines, and source code to diagnose and fix real ML failure patterns.

## What Is This?

This environment recreates the experience of an ML engineer facing a broken PyTorch training job. The agent receives a snapshot of a failing training run and must:

1. **Investigate** — inspect gradients, data batches, model weights, model modes, and code
2. **Diagnose** — identify the root cause from a closed set of known ML failures
3. **Fix** — apply the correct intervention (reduce LR, patch data, fix model mode, etc.)
4. **Verify** — restart training and confirm recovery before submitting diagnosis

### Key Differentiators

- **PyTorch-native internals** — Real `torch.nn.Module` models (~50K params), real `torch.autograd` gradients, real `state_dict()` weight snapshots
- **Context-gated reward shaping** — Penalty fires only when agent ignores evidence it already gathered; no penalty for reasonable priors
- **Progressive information reveal** — Gradient stats, weight stats, data batch stats only populated after corresponding inspection actions

## Environment Design

### Observation Space (`MLTrainingObservation`)

| Field | Type | Visibility |
|-------|------|-----------|
| `training_loss_history` | `list[float]` (20 epochs) | Always |
| `val_accuracy_history` | `list[float]` (20 epochs) | Always |
| `val_loss_history` | `list[float]` (20 epochs) | Always |
| `current_config` | `TrainingConfig` | Always |
| `error_log` | `Optional[str]` | Always |
| `gradient_stats` | `list[GradientStats]` | After `inspect_gradients` |
| `model_weight_stats` | `Optional[list[ModelWeightStats]]` | After `inspect_model_weights` |
| `data_batch_stats` | `Optional[DataBatchStats]` | After `inspect_data_batch` |
| `model_mode_info` | `Optional[dict[str, str]]` | After `inspect_model_modes` |
| `code_snippet` | `Optional[CodeSnippet]` | After `inspect_code` |
| `available_actions` | `list[str]` | Always (dynamic) |
| `episode_state` | `EpisodeState` | Always |

### Action Space (`MLTrainingAction`)

| Category | Actions |
|----------|---------|
| **Investigation** | `inspect_gradients`, `inspect_data_batch`, `inspect_model_modes`, `inspect_model_weights`, `inspect_code` |
| **Fix** | `modify_config`, `add_callback`, `replace_optimizer`, `patch_data_loader`, `fix_model_mode`, `fix_code` |
| **Terminal** | `restart_run`, `mark_diagnosed` |

Dynamic availability: `restart_run` requires a fix first; `fix_code` requires code inspection; `mark_diagnosed` disappears after submission.

### Diagnosis Enum

| Value | Description |
|-------|-------------|
| `lr_too_high` | Learning rate too large |
| `vanishing_gradients` | Gradients decay to near-zero |
| `data_leakage` | Validation samples in training |
| `overfitting` | Model memorizing, failing to generalize |
| `batchnorm_eval_mode` | Model in eval mode during training |
| `code_bug` | Bug in PyTorch training code |

### Reward Function

| Event | Reward | Gate |
|-------|--------|------|
| Any step | -0.01 | Flat, unconditional |
| First-time inspection | +0.05 | Per inspection type |
| `add_callback` after normal gradients | -0.20 | `gradients_inspected AND gradients_were_normal` |
| Invalid action | -0.05 | Action not in `available_actions` |
| Correct diagnosis | +0.50 | Equality check |
| Wrong diagnosis | -0.30 | Inequality check |
| Convergence after fix+restart | +0.40 | All gates met |

## Tasks

| ID | Difficulty | Root Cause | Description |
|----|-----------|------------|-------------|
| `task_001` | Easy | `lr_too_high` | Exploding gradients — all layers show `is_exploding: True`, NaN in error log |
| `task_002` | Easy | `vanishing_gradients` | Vanishing gradients — deeper layers show `is_vanishing: True`, flat loss curve |
| `task_003` | Medium | `data_leakage` | Silent data leakage — suspiciously high val accuracy, `class_overlap_score > 0.5` |
| `task_004` | Medium | `overfitting` | Train-val divergence — loss approaches 0 while val loss climbs |
| `task_005` | Hard | `batchnorm_eval_mode` | Model in eval mode with compound red herrings (FC gradient spike, GPU 91%, near-vanishing conv1) |
| `task_006` | Hard | `code_bug` | PyTorch code bug — agent must read and fix actual Python code (4 bug variants) |

## Baseline Scores

Rule-based heuristic baseline (deterministic, no API key, bit-exact reproducible):

| Task | Score | Notes |
|------|-------|-------|
| `task_001` | 1.00 | Direct signal: `is_exploding` on all layers |
| `task_002` | 1.00 | Direct signal: `is_vanishing` on deeper layers |
| `task_003` | 1.00 | `class_overlap_score > 0.5` triggers correct path |
| `task_004` | 1.00 | Detects train-val divergence + near-zero train loss |
| `task_005` | 0.35 | Fixed investigation order misses eval mode — hard task genuinely challenges agents |
| `task_006` | 1.00 | Pattern-matching catches 2 of 4 bug variants |

## Setup

### Local Development

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install openenv-core pydantic fastapi uvicorn

# Install dev tools
pip install pytest pytest-cov black ruff isort

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run tests
pytest tests/ -v --cov=ml_training_debugger

# Run baseline
python baseline_heuristic.py
```

### Docker

```bash
docker build -t pytorch-debugger .
docker run -p 7860:7860 pytorch-debugger
curl http://localhost:7860/health
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | `{"status": "ready", "tasks": 6}` |
| `/tasks` | GET | Task list with action schema |
| `/grader` | POST | Grader score for last completed episode |
| `/baseline` | POST | Run baseline, return scores for all 6 tasks |
| `/dashboard` | GET | Live diagnostic dashboard (Plotly.js, 4-panel) |
| `/ws` | WebSocket | Primary agent interface |
| `/reset` | POST | Reset environment (framework) |
| `/step` | POST | Execute action (framework) |
| `/state` | GET | Current state (framework) |
| `/schema` | GET | Action/observation schemas (framework) |
| `/docs` | GET | Swagger UI (framework) |

### WebSocket Message Format

The primary agent interface is the WebSocket endpoint at `/ws`. Messages use JSON:

**Reset** (start a new episode, optionally select task):
```json
{"type": "reset"}
{"type": "reset", "data": {"task_id": "task_003", "seed": 42}}
```
Without `data`, defaults to `task_001`. With `data`, selects the specified task.

Returns: `{"type": "observation", "data": {"observation": {...}, "reward": 0.0, "done": false}}`

**Step** (execute an action):
```json
{"type": "step", "data": {"action_type": "inspect_gradients"}}
```
```json
{"type": "step", "data": {"action_type": "modify_config", "target": "learning_rate", "value": 0.001}}
```
```json
{"type": "step", "data": {"action_type": "mark_diagnosed", "diagnosis": "lr_too_high"}}
```
Returns: `{"type": "observation", "data": {"observation": {...}, "reward": float, "done": bool}}`

### HTTP vs WebSocket

**WebSocket `/ws`** is the primary agent interface — it maintains a persistent session across reset/step/diagnose. Use this for full episodes.

**HTTP `POST /reset` and `POST /step`** are stateless per the OpenEnv framework design — each request creates a fresh environment instance. Use these for single-action queries or health checks, not full episodes.

**Custom endpoints** (`POST /baseline`, `POST /grader`, `GET /tasks`, `GET /health`) work independently of sessions.

## Validation Suite

A PyTorch validation suite proves simulation fidelity by comparing parametric curve generation against real training runs. Pre-computed fidelity reports are served at `GET /validation-report`.

**Methodology:** Real `torch.nn.Module` models are trained with each fault type, and the resulting loss/accuracy curves are compared against the parametric generators. All fault injection uses real `torch.autograd` gradients and `model.state_dict()` weights — not synthetic formulas.

**Coverage:** Exploding gradients, vanishing gradients, data leakage, overfitting, BatchNorm eval mode, and all 4 code bug variants.

## Architecture

- **Python 3.12** · PyTorch CPU-only · openenv-core
- Real `torch.nn.Module` models with real `torch.autograd` gradients
- Parametric curve generation for loss/accuracy histories (sub-ms latency)
- Typed Pydantic models everywhere — no `Dict[str, Any]`
- `import torch` in every core module — zero numpy in core
- Session isolation via per-session `EpisodeState`
- Deterministic reproducibility via `torch.manual_seed()`

### Docker Image Size

The Docker image is ~1.5GB. This is driven by `libtorch_cpu.so` (426MB) — the core PyTorch CPU binary required for real `torch.nn.Module`, `torch.autograd`, and `model.state_dict()` support. This is the intentional trade-off: real PyTorch gradient computation and weight inspection (not synthetic data) requires the full CPU runtime. Non-essential torch components (test suites, benchmark tools, CUDA stubs, type stubs) are stripped in the Dockerfile.
