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
| `task_003` | Medium | `data_leakage` | Silent data leakage — suspiciously high val accuracy, `class_overlap_score > 0.5` |
| `task_005` | Hard | `batchnorm_eval_mode` | Model in eval mode with compound red herrings (FC gradient spike, GPU 91%, near-vanishing conv1) |

## Baseline Scores

Rule-based heuristic baseline (deterministic, no API key):

| Task | Score |
|------|-------|
| `task_001` | 1.00 |
| `task_003` | 1.00 |
| `task_005` | 0.35 |

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
| `/health` | GET | `{"status": "ready", "tasks": 3}` |
| `/tasks` | GET | Task list with action schema |
| `/grader` | POST | Grader score for last completed episode |
| `/baseline` | POST | Run baseline, return scores |
| `/ws` | WebSocket | Primary agent interface |
| `/reset` | POST | Reset environment (framework) |
| `/step` | POST | Execute action (framework) |
| `/state` | GET | Current state (framework) |
| `/schema` | GET | Action/observation schemas (framework) |
| `/docs` | GET | Swagger UI (framework) |

## Architecture

- **Python 3.12** · PyTorch CPU-only · openenv-core
- Real `torch.nn.Module` models with real `torch.autograd` gradients
- Parametric curve generation for loss/accuracy histories (sub-ms latency)
- Typed Pydantic models everywhere — no `Dict[str, Any]`
- `import torch` in every core module — zero numpy in core
- Session isolation via per-session `EpisodeState`
- Deterministic reproducibility via `torch.manual_seed()`
