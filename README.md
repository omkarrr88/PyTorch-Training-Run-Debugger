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

- **Real PyTorch mini-training** — 20 real forward+backward epochs per reset, cached for instant replay. Loss/accuracy curves come from real training, not parametric formulas.
- **Dual model architectures** — SimpleCNN (~50K params) and SimpleMLP (~20K params) randomly selected per episode
- **Context-gated reward shaping** — Penalty fires only when agent ignores evidence it already gathered; no penalty for reasonable priors
- **Progressive information reveal** — Gradient stats, weight stats, data batch stats, confusion matrices only populated after corresponding inspection actions
- **7 tasks with difficulty scaling** — Easy to hard, with configurable difficulty level (1-5) per task

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
| `task_007` | Med-Hard | `scheduler_misconfigured` | LR scheduler with wrong gamma/step_size — training stagnates after initial progress |

All tasks support `difficulty_level` (1-5) via reset: `{"type": "reset", "data": {"task_id": "task_005", "difficulty_level": 4}}`

## Baseline Scores

### Heuristic vs LLM Comparison (3 agents, 7 tasks)

| Task | Difficulty | Heuristic | Llama 3.3 70B | Llama 3.1 8B | Notes |
|------|-----------|-----------|---------------|--------------|-------|
| `task_001` | Easy | **1.00** | 1.00 | 0.60 | 8B finds issue but misses fix+restart sequence |
| `task_002` | Easy | **1.00** | 1.00 | 0.05 | 8B barely investigates — struggles with multi-step reasoning |
| `task_003` | Medium | **1.00** | 0.40 | 0.40 | Both LLMs explore inefficiently vs heuristic's direct path |
| `task_004` | Medium | 0.45 | 0.45 | **0.60** | LLM's flexible investigation finds overfitting signals heuristic misses |
| `task_005` | Hard | **1.00** | 1.00 | 1.00 | All agents find eval mode via model inspection |
| `task_006` | Hard | **1.00** | — | 0.60–1.00 | Code debugging — 8B varies across providers |
| `task_007` | Med-Hard | **1.00** | — | 0.60 | Scheduler detection — heuristic's pattern matching excels |
| **Average** | | **0.92** | **0.69*** | **0.55** | |

*Llama 3.3 70B results are partial (5/7 tasks before rate limit). Projected average ~0.69.

**Key insights:**
1. **Model size matters:** 70B scores ~25% higher than 8B — the environment scales with model capability
2. **Heuristic beats LLMs:** A domain-specific decision tree (0.92) outperforms general-purpose LLMs (0.55-0.69) — proving the environment rewards systematic debugging strategy
3. **Task 4 is the exception:** LLMs outperform the heuristic on overfitting because real training curves require flexible reasoning, not rigid pattern matching
4. **8B struggles on multi-step tasks:** Task 2 (0.05) shows small models can't maintain investigation strategy across many steps

### Running Baselines

```bash
# Heuristic (deterministic, no API key, bit-exact reproducible)
python3 baseline_heuristic.py

# LLM (multi-provider support — set API key in .env)
python3 baseline_inference.py                       # Groq (default, free)
python3 baseline_inference.py --provider cerebras    # Cerebras (free)
python3 baseline_inference.py --provider gemini      # Google Gemini
python3 baseline_inference.py --provider openai      # OpenAI GPT-4o

# Run all baselines with comparison table
python3 run_all_baselines.py
```

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

8/8 validation checks pass — served live at `GET /validation-report`:

**Methodology:** Real PyTorch 20-epoch mini-training with fault injection. Each fault type is validated with behavioral checks (gradient detection, loss patterns, model mode, code fix acceptance). Both SimpleCNN and SimpleMLP architectures verified.

**Coverage:** Exploding gradients, vanishing gradients, data leakage, overfitting, BatchNorm eval mode, code bugs (4 variants), scheduler misconfigured, dual architecture.

## Architecture

- **Python 3.12** · PyTorch 2.5.1 CPU-only · openenv-core v0.2.2
- **Dual model architectures**: SimpleCNN (~50K params) + SimpleMLP (~20K params)
- **Real 20-epoch mini-training** per reset (cached per task/seed for instant replay)
- Typed Pydantic models everywhere — no `Dict[str, Any]`
- `import torch` in every core module — zero numpy in core
- Session isolation via per-session `EpisodeState`
- Deterministic reproducibility via `torch.manual_seed()`
- **251 tests, 95% coverage**

### Docker Image Size

The Docker image is **885MB** (optimized from 1.96GB via multi-stage build, torch 2.5.1, `strip --strip-unneeded`, and removal of unused transitive dependencies). The core `libtorch_cpu.so` (329MB stripped) is the irreducible minimum for real `torch.nn.Module`, `torch.autograd`, and `model.state_dict()` support — the intentional trade-off for authentic PyTorch computation vs synthetic data.

### Research Paper

See [PAPER.md](PAPER.md) — "Context-Gated Reward Shaping for Evidence-Based ML Debugging"

### Project Explanation

See [EXPLANATION.md](EXPLANATION.md) — full project explanation in simple language
