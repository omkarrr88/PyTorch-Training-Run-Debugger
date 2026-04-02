# PyTorch Training Run Debugger
### OpenEnv Environment — Meta PyTorch OpenEnv Hackathon x Scaler School of Technology, Round 1

---

## TL;DR

- **What:** An OpenEnv RL environment where an AI agent debugs broken PyTorch training runs — investigating gradients, model weights, data pipelines, and source code to diagnose and fix real ML failure patterns.
- **Why it wins:** PyTorch-native internals (real `torch.nn.Module` models, real `torch.autograd` gradients), context-gated reward shaping (penalizes ignoring gathered evidence), and code-level debugging (agent reads and fixes actual PyTorch code) — none of which exist in any current OpenEnv environment.
- **Scope:** 6 tasks across 3 difficulty tiers (easy/medium/hard), deterministic graders (0.0–1.0), rule-based + LLM baselines, live diagnostic dashboard, full Docker + HF Spaces deployment. **MVP (Tasks 1, 3, 5) satisfies the minimum 3-task requirement with full difficulty range; Tasks 2, 4, 6 extend to full scope.**
- **Key differentiators:** (1) Context-gated penalty that encodes evidence-based reasoning into the reward signal, (2) Task 6 tests PyTorch code comprehension — directly aligned with Meta's interests, (3) PyTorch validation suite proves simulation fidelity with R² > 0.85.
- **Deadlines:** Submission window opens March 28, 2026. Round 1 deadline: April 8, 2026.

---

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [Why This Environment Matters](#2-why-this-environment-matters)
3. [Problem Statement Alignment](#3-problem-statement-alignment)
4. [Team Composition](#4-team-composition)
   - 4.1. [Scope Prioritization](#41-scope-prioritization)
5. [Key Differentiator — Context-Gated Reward Shaping](#5-key-differentiator--context-gated-reward-shaping)
6. [PyTorch-Native Fault Injection Engine](#6-pytorch-native-fault-injection-engine)
7. [High-Level Concept](#7-high-level-concept)
8. [What Makes This a Genuine RL Environment](#8-what-makes-this-a-genuine-rl-environment)
9. [Architecture](#9-architecture)
10. [Data Models — Complete Typed Specification](#10-data-models--complete-typed-specification)
11. [The Six Core Tasks](#11-the-six-core-tasks)
12. [Reward Function — Design and Rationale](#12-reward-function--design-and-rationale)
13. [Environment Lifecycle](#13-environment-lifecycle)
14. [OpenEnv Spec Compliance](#14-openenv-spec-compliance)
15. [Docker and Deployment](#15-docker-and-deployment)
16. [Error Handling and Edge Cases](#16-error-handling-and-edge-cases)
17. [Baseline Inference Design](#17-baseline-inference-design)
18. [PyTorch Validation Suite](#18-pytorch-validation-suite)
19. [Live Diagnostic Dashboard](#19-live-diagnostic-dashboard)
20. [Project File Structure](#20-project-file-structure)
21. [Extensibility](#21-extensibility)
22. [Known Risks & Mitigations](#22-known-risks--mitigations)
23. [Design Decision Rationale](#23-design-decision-rationale)
24. [Submission Readiness Checklist](#24-submission-readiness-checklist)

---

## 1. What Is This?

**PyTorch Training Run Debugger** is a complete OpenEnv environment that recreates the experience of an ML engineer facing a broken PyTorch training job. The environment is built on a **PyTorch-native fault injection engine** — it instantiates real `torch.nn.Module` models, runs real forward/backward passes via `torch.autograd`, and exposes real `state_dict()` weight snapshots and gradient tensors to the agent. An AI agent receives a snapshot of a failing training run and must:

1. **Investigate** — inspect gradients, data batches, model weights, model layer modes, and PyTorch code snippets using targeted inspection actions
2. **Diagnose** — identify the root cause from a closed enumeration of known ML failure types
3. **Fix** — apply the correct intervention (reduce the learning rate, patch the data loader, fix model mode, adjust regularization, correct config, or fix buggy PyTorch code)
4. **Verify** — restart the run and confirm training recovers before submitting a diagnosis

The environment covers six distinct failure scenarios across three difficulty tiers — from obvious (exploding gradients) through ambiguous (overfitting, data leakage) to subtle (BatchNorm layers stuck in eval mode with compound red-herring signals, and a code-level PyTorch bug that requires reading and fixing actual Python source). **The minimum 3-task requirement is satisfied by Tasks 1, 3, 5 (easy/medium/hard MVP); Tasks 2, 4, 6 complete the full scope.** Task IDs are opaque (`task_001` through `task_006`), preventing the agent from inferring the diagnosis from the task name.

**Three things set this environment apart:**

1. **PyTorch-native internals.** The fault injection engine uses real PyTorch models — `torch.nn.Module` subclasses with real parameters, real `torch.autograd` gradients, and real `model.state_dict()` snapshots. Gradient stats and model weight stats come from actual tensor operations, not synthetic formulas. Loss curves use parametric generation for speed, but every model-level observation is grounded in real PyTorch computation. This is validated by a PyTorch validation suite (see [Section 18](#18-pytorch-validation-suite)) that proves simulation fidelity against real training runs — with fidelity reports served live at `GET /validation-report`.

2. **Context-gated reward shaping.** A penalty system that distinguishes between "reasonable prior action" and "ignoring counter-evidence the agent already gathered." This encodes evidence-based decision making directly into the reward signal — a capability no existing OpenEnv environment attempts. See [Section 5](#5-key-differentiator--context-gated-reward-shaping).

3. **Code-level debugging (Task 6).** The agent sees actual PyTorch code snippets in the observation and must identify the buggy line and submit a code fix — testing code understanding, not just metric interpretation. This directly addresses what Meta cares about: can an AI agent debug PyTorch code?

A live diagnostic dashboard visualizes episodes in real time — loss curves, gradient heatmaps, action timelines, and reward progression — making agent behavior immediately legible to judges and users.

---

## 2. Why This Environment Matters

MLOps teams at companies running production training pipelines spend 15–25% of engineer time debugging silent training failures — runs that produce no error, no crash, just mysteriously bad metrics. The cost is not just engineer hours: each misdiagnosed restart wastes GPU compute that costs $2–8 per hour per card. An agent trained to systematically investigate and correctly diagnose training failures on the first attempt has direct, measurable commercial value.

The failure patterns are well-understood by practitioners — a learning rate that is too high, data leakage through sloppy dataset splits, BatchNorm not set to train mode — but the diagnostic process in practice is difficult because:

- Real training runs produce noisy, ambiguous signals
- Multiple symptoms can point to multiple causes simultaneously
- Some bugs produce no error at all — just mysteriously bad performance
- Fixing the wrong thing wastes hours of compute time and restarts
- A simple linter or static analysis tool can catch `model.eval()` in source code, but cannot reason through ambiguous runtime signals to determine which of several plausible causes is active in a live run

This environment trains agents to handle the *investigation process* — the reasoning under uncertainty when multiple symptoms are present — not just the final diagnosis. An agent that can identify `batchnorm_eval_mode` from degrading validation accuracy and a single misleading gradient spike has developed a debugging strategy that transfers directly to real MLOps workflows.

The OpenEnv Hub currently contains a demo echo environment and a code execution environment. No existing OpenEnv environment covers ML training failure diagnosis, gradient analysis, PyTorch model weight inspection, or hyperparameter debugging. This fills a genuine gap in the ecosystem.

---

## 3. Problem Statement Alignment

Round 1 uses an open-ended problem statement: **"Build a complete, real-world OpenEnv environment that an AI agent can learn from."** Participants choose their own domain — there is no requirement to select from a fixed list. We chose ML training debugging because:

1. It is a genuine, unsolved real-world task — not a toy problem
2. It directly showcases PyTorch expertise (critical for a Meta PyTorch hackathon)
3. No existing OpenEnv environment covers this domain — zero competition in the ecosystem
4. The diagnostic process naturally fits the `step()`/`reset()`/`state()` API — it is inherently sequential, stateful, and rewards systematic reasoning
5. It produces measurably different scores for heuristic vs. reasoning-capable agents

**Alignment with evaluation criteria:**

| Criterion (Weight) | How This Environment Scores |
|---|---|
| Real-world utility (30%) | ML training debugging is a $B+ industry problem. Every team running PyTorch training encounters these failures. |
| Task & grader quality (25%) | 6 tasks across 3 difficulty tiers with deterministic graders scoring 0.0–1.0. Hard tasks genuinely challenge frontier models. |
| Environment design (20%) | Progressive information reveal, context-gated penalties, PyTorch-native model inspection, code-level debugging. |
| Code quality & spec compliance (15%) | Full OpenEnv spec, typed Pydantic models, Dockerfile, HF Space, two baselines. |
| Creativity & novelty (10%) | Context-gated reward shaping, real PyTorch model internals, code fix task — none exist in OpenEnv today. |

While domains like email triage or customer support have broader accessibility, ML training debugging targets a high-value niche where every practitioner running PyTorch training encounters these failures — and the hackathon is explicitly PyTorch-focused, making this domain strategically aligned with what judges from Meta and Hugging Face value most.

---

## 4. Team Composition

| Role | Responsibility | Focus Areas |
|---|---|---|
| **Lead Engineer** | Environment core, reward engine, graders | `simulation.py`, `reward_engine.py`, `graders.py`, OpenEnv integration |
| **PyTorch Specialist** | Fault injection engine, validation suite, real model integration | `pytorch_engine.py`, `validation/`, Tier 2 & 3 features |
| **Frontend + DevOps** | Dashboard, Docker, HF Spaces deployment, baseline scripts | `dashboard.html`, `Dockerfile`, `baseline_*.py`, CI |

*If solo or 2-person team: prioritize Tasks 1, 3, 5 (easy/medium/hard) + baseline + Docker + deploy. Tasks 2, 4, 6 are stretch goals.*

---

## 4.1. Scope Prioritization

The environment is designed in layers. The MVP is a complete, deployable submission. Full scope adds depth and polish.

**MVP (must ship — this alone is a strong Round 1 submission):**
- Tasks 1, 3, 5 (easy + medium + hard) — covers the required minimum of 3 tasks with full difficulty range
- Rule-based baseline (`baseline_heuristic.py`) — deterministic, no API key, guaranteed reproducibility
- All required endpoints (`/ws`, `/tasks`, `/grader`, `/baseline`, `/health`)
- Dockerfile that builds and runs cleanly
- HF Spaces deployment with `openenv` tag
- README with full documentation

**Full scope (adds differentiation if time permits):**
- Tasks 2, 4, 6 — doubles the task count, adds code-level debugging (Task 6 is the strongest differentiator)
- LLM baseline (`baseline_inference.py`) — demonstrates heuristic vs. reasoning score gap
- Live diagnostic dashboard (`GET /dashboard`)
- PyTorch validation suite with fidelity reports (`GET /validation-report`)

**Priority order if time-constrained:** MVP first, then Task 6, then Tasks 2 & 4, then dashboard, then validation suite, then LLM baseline.

---

## 5. Key Differentiator — Context-Gated Reward Shaping

Most RL environments use stateless reward functions — "did this action happen, or not." This environment introduces **context-gated penalties** that require knowledge of the agent's full information state at the time of each action.

**The core idea:** An agent that adds gradient clipping *before* inspecting gradients is following a reasonable prior — gradient clipping is a defensible first response to training instability. That agent has not ignored any evidence because no evidence has been gathered. **No penalty.**

An agent that calls `inspect_gradients`, sees that `is_exploding` is false on all layers, and *then* still adds gradient clipping is explicitly ignoring the counter-evidence it just collected. **That agent receives a −0.20 penalty.**

This single mechanic encodes the concept of *evidence-based decision making* into the reward signal. It distinguishes between:
- **Reasonable prior** → no penalty (the agent acted on incomplete information)
- **Ignoring counter-evidence** → penalty (the agent had information and disregarded it)

This is the kind of nuanced reward design that separates a thoughtfully constructed environment from a simple task wrapper. It teaches agents to *reason about what they've already learned* rather than follow fixed playbooks — a capability with direct transfer value to real-world MLOps debugging where the cost of ignoring gathered evidence is wasted GPU hours and delayed incident resolution.

No existing OpenEnv environment attempts this distinction. It is a primary contribution of this submission.

---

## 6. PyTorch-Native Fault Injection Engine

The core of the environment is a **PyTorch-native fault injection engine** that combines two computation strategies for optimal performance:

### Strategy 1 — Parametric Loss Curves (for speed)

Training loss and validation accuracy histories are generated via parametric curve equations using `torch.Tensor` operations. Each failure mode is modeled from its mathematical characterization:

```python
# All curve math uses PyTorch tensor operations
loss_history = torch.exp(torch.tensor(lr) * torch.arange(20)).tolist()
val_acc = (torch.sigmoid(torch.linspace(-3, 3, 20)) * (1 - leakage_pct)).tolist()
grad_decay = torch.exp(-torch.tensor(depth_multiplier) * torch.arange(num_layers)).tolist()
```

A high learning rate produces an exponentially growing loss curve, data leakage produces validation accuracy that inflates proportionally to the leakage percentage, vanishing gradients produce decaying norms that starve deeper layers, overfitting produces the classic train-val divergence, and BatchNorm in eval mode produces elevated training loss variance with gradual validation accuracy degradation.

### Strategy 2 — Real PyTorch Model Snapshots (for authenticity)

At `reset()` time, the engine instantiates a small real PyTorch model (3-layer CNN, ~50K parameters on 32×32 random input) and injects the fault scenario into it:

```python
model = SimpleCNN()  # Real torch.nn.Module

# Fault injection examples:
# Exploding gradients: set lr=0.1 and run 2 forward+backward passes
# BatchNorm eval: call model.eval() before training
# Overfitting: set weight_decay=0.0 on the optimizer
# Data leakage: inject duplicate samples into the val split

optimizer = torch.optim.Adam(model.parameters(), lr=scenario.learning_rate)
for _ in range(2):  # 1-2 real passes, <100ms total
    output = model(random_batch)
    loss = criterion(output, random_labels)
    loss.backward()
    optimizer.step()
```

This produces **real gradient tensors** and **real weight snapshots** that are frozen and served to the agent via inspection actions. The `inspect_gradients` and `inspect_model_weights` actions return data computed from actual `torch.autograd` — not synthetic formulas.

### Why This Hybrid Approach

| Property | Parametric Curves | Real PyTorch Model |
|---|---|---|
| What it produces | Loss/accuracy histories (20 epochs) | Gradient norms, weight stats, model mode info |
| Latency | Sub-millisecond (`torch.Tensor` math) | <100ms (1-2 forward passes on tiny model) |
| Why not all-real | 20-epoch training loop = 10-40s per reset — too slow for auto-validator | — |
| Why not all-parametric | — | Judges want to see `import torch` in core code, real `state_dict()`, real gradients |

The result: every `step()` call completes in under 10ms. The `reset()` call takes <200ms (model instantiation + 2 forward passes + parametric curve generation). The auto-validator's timeout budget is easily met.

### Additional Properties

**Bit-exact reproducibility.** Both `torch.manual_seed()` and the parametric curve seeds are controlled per episode. The baseline script's two-run comparison check passes reliably.

**Exploit resistance.** Each `reset()` call samples fresh fault parameters from defined ranges. Task IDs are opaque (`task_001`–`task_006`), preventing the agent from inferring the diagnosis from the name. Memorization is not a viable strategy.

| Task | Parameter | Range |
|---|---|---|
| Exploding gradients | Learning rate | `[0.05, 0.08, 0.10, 0.15, 0.30]` |
| Vanishing gradients | Learning rate | `[1e-6, 5e-6, 1e-5]` |
| Vanishing gradients | Depth multiplier | `[1.0, 1.5, 2.0]` |
| Data leakage | Leakage percentage | `[0.12, 0.18, 0.22, 0.28]` |
| Overfitting | Weight decay | `[0.0, 0.0001, 0.001]` |
| Overfitting | Train/val gap epoch | `[5, 8, 12]` |
| BatchNorm eval mode | Red herring spike layer | `["fc", "conv1"]` |
| BatchNorm eval mode | Red herring intensity | `uniform(0.8, 2.5)` |
| Code bug | Bug type | `["eval_mode", "detach_loss", "zero_grad_missing", "inplace_relu"]` |

---

## 7. High-Level Concept

### The Scenario

An ML engineer receives an alert: their training job is behaving unexpectedly. They open their monitoring dashboard and see a loss curve that is diverging, a validation accuracy that is suspiciously high or slowly degrading, and gradient statistics that may or may not be the actual source of the problem. The engineer must systematically investigate, form a hypothesis, apply a fix, and restart the job — all while managing uncertainty about which of several plausible explanations is correct.

This environment recreates that experience exactly. The AI agent receives the same information the engineer would see. It uses the same tools — inspecting gradients, inspecting model weights via `state_dict()`, sampling a data batch, checking model layer modes, reading PyTorch code, modifying config values. It must reason through noisy and sometimes deliberately misleading signals before submitting a formal diagnosis using a closed enumeration of known root causes.

### The Key Constraint: Progressive Information Reveal

The initial observation returned by `reset()` contains training loss history, validation accuracy history, learning rate, current config, GPU memory, and an error log — but it does **not** contain gradient statistics, model weight statistics, data batch statistics, or code snippets. These are revealed progressively as the agent calls inspection actions.

This is the mechanic that makes the environment a genuine investigation task rather than a pattern-matching task. The agent must actively choose what to look at. Every inspection action costs one step (small penalty) but returns information that has real diagnostic value — even a negative result meaningfully narrows the hypothesis space.

---

## 8. What Makes This a Genuine RL Environment

This is not a question-answering task where an agent reads text and produces a label. It is a multi-step interactive diagnostic process where:

- The agent's actions change what information is available in subsequent observations — calling `inspect_gradients` populates gradient data that was not present before
- Real PyTorch model internals (gradient tensors from `torch.autograd`, weight statistics from `state_dict()`) are exposed through inspection actions
- Each unique inspection action earns a small reward, reflecting real information value even when the result is negative
- Investigating the wrong hypothesis wastes actions due to a flat per-step penalty
- The correct fix must be applied AND verified through a training restart — stating the diagnosis alone is not enough
- A context-gated penalty fires only when the agent ignores counter-evidence it has already gathered, distinguishing informed exploration from evidence-ignorant behavior
- Task 6 requires reading and fixing actual PyTorch code — testing code understanding, not just metric interpretation
- The reward signal is shaped across the entire trajectory, not just at the terminal state

Agents must develop a strategy for systematic investigation. The environment rewards disciplined diagnostic reasoning and penalizes pattern-matching without evidence.

---

## 9. Architecture

### Framework

The environment is built on **FastAPI** with the `openenv-core` (v0.2.2) package providing base classes and the Gymnasium-style API contract (`reset()`, `step()`, `state()`). The server extends the `Environment` base class from `openenv.core.env_server.interfaces`. The client extends `GenericEnvClient`.

**Framework composition — verified.** The `openenv-core` framework's `create_app()` function (from `openenv.core.env_server.http_server`) returns a standard **FastAPI** instance. Custom HTTP routes can be added directly to this instance after creation. This has been verified by:

1. Installing `openenv-core` v0.2.2 and running `openenv init` to generate a scaffold
2. Calling `create_app(EnvironmentClass, ActionClass, ObservationClass)` to get the FastAPI app
3. Adding custom routes (`@app.get('/tasks')`, `@app.post('/grader')`, `@app.post('/baseline')`) to the returned app
4. Confirming all routes — both framework-provided and custom — appear in the app's route table

**Framework-provided routes (built-in, do not implement):**
- `POST /reset` — resets the environment (framework handles deserialization)
- `POST /step` — executes an action (framework handles action/observation serialization)
- `GET /state` — returns current environment state
- `GET /schema` — returns JSON schemas for action/observation types
- `WS /ws` — WebSocket endpoint for persistent sessions
- `GET /health` — basic health check (we override with a richer version)
- `GET /docs` — Swagger UI (auto-generated)
- `/mcp` — MCP protocol endpoint (framework-provided)

**Custom routes (hackathon-required, we implement):**
- `GET /tasks` — returns task list with IDs, difficulties, and the full `MLTrainingAction` JSON schema
- `POST /grader` — returns grader score (0.0–1.0) for the most recently completed episode
- `POST /baseline` — triggers baseline inference run, returns scores for all tasks
- `GET /health` — overrides framework health check with `{"status": "ready", "tasks": 6}`

**Key implementation pattern (from verified scaffold):**

```python
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Action, Observation
from openenv.core.env_server.interfaces import Environment

# 1. create_app returns a standard FastAPI instance
app = create_app(
    MLTrainingEnvironment,   # Environment factory (class, not instance)
    MLTrainingAction,        # Action subclass
    MLTrainingObservation,   # Observation subclass
    env_name="pytorch_training_debugger",
    max_concurrent_envs=5,
)

# 2. Custom routes are added directly — no adapter needed
@app.get("/tasks")
def get_tasks(): ...

@app.post("/grader")
def post_grader(): ...

@app.post("/baseline")
async def post_baseline(): ...
```

**Models must extend framework base classes:**
- `MLTrainingAction` extends `Action` (from `openenv.core.env_server.types`)
- `MLTrainingObservation` extends `Observation` (from `openenv.core.env_server.types`) — includes built-in `done`, `reward`, and `metadata` fields
- `MLTrainingEnvironment` extends `Environment` (from `openenv.core.env_server.interfaces`) — must implement `reset()`, `step()`, and a `state` property returning `State`

**No WebSocket fallback needed.** The framework handles all WebSocket protocol details. The `create_app()` composition pattern is clean and standard.

### PyTorch Model Pool

At server startup, the environment pre-initializes a pool of small PyTorch model architectures:

```python
class SimpleCNN(torch.nn.Module):
    """3-layer CNN for CIFAR-10 style classification. ~50K params."""
    def __init__(self, num_layers=3, hidden_dim=64):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.fc = torch.nn.Linear(64 * 4 * 4, 10)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
```

Each `reset()` call creates a fresh model instance, injects the fault, runs 1-2 forward+backward passes, and freezes the resulting state. This keeps `reset()` under 200ms while producing real gradient and weight data.

### Session Model

Each `reset()` call creates a new `EpisodeState` and samples a fresh `ScenarioParams` object with randomized fault parameters and a new random seed. All subsequent `step()` calls in that episode operate on that state. The episode ends when `mark_diagnosed` is called or the step limit is reached. Each connected WebSocket client gets its own isolated episode state — there is no shared global state. The server maintains a `dict[str, EpisodeState]` keyed by session ID, cleaned up on disconnect or episode completion.

---

## 10. Data Models — Complete Typed Specification

All models are Pydantic `BaseModel` subclasses. Every field is explicitly typed.

### RootCauseDiagnosis — The Closed Enum

The root cause diagnosis is a closed enumeration of known ML failure types. Requiring the agent to submit a diagnosis from this fixed list rather than free text makes the grader a single equality check. There is no "close enough." Either the agent correctly identifies `batchnorm_eval_mode` or it does not. This eliminates all subjective judgment from evaluation.

| Value | Description |
|---|---|
| `lr_too_high` | Learning rate set too large for the architecture |
| `vanishing_gradients` | Learning rate too low or architecture too deep, gradients decay to near-zero |
| `data_leakage` | Validation samples appearing in training batches |
| `overfitting` | Model memorizing training data, failing to generalize |
| `batchnorm_eval_mode` | Model left in eval mode, BatchNorm using running statistics |
| `code_bug` | A bug in the PyTorch training code (Task 6) |

**Task ID to root cause mapping is hidden from the agent.** Task IDs are opaque strings (`task_001` through `task_006`). The agent cannot infer the diagnosis from the task name — it must earn the diagnosis through investigation. This is the single, definitive statement on opaque task IDs; it applies to all tasks throughout this document.

### ScenarioParams — Randomized Per Episode

This is the internal object created at `reset()` time. It is not exposed directly to the agent but controls all generation throughout the episode. Key fields include the task identifier, the true root cause enum value, randomized fault parameters specific to the scenario type, a random seed for deterministic-within-episode generation, a red herring configuration, and (for Task 6) the specific code bug variant.

### ModelWeightStats — Real PyTorch Model Inspection (Tier 2)

Populated only after the agent calls `inspect_model_weights`. These stats come from a real `torch.nn.Module`'s `state_dict()` — not synthetic data.

| Field | Type | Description |
|---|---|---|
| `layer_name` | `str` | Parameter name from `model.state_dict()` |
| `weight_norm` | `float` | `torch.norm(param).item()` |
| `weight_mean` | `float` | `param.mean().item()` |
| `weight_std` | `float` | `param.std().item()` |
| `weight_min` | `float` | `param.min().item()` |
| `weight_max` | `float` | `param.max().item()` |
| `dead_neuron_pct` | `float` | Fraction of neurons with zero output (ReLU death detection) |
| `has_nan` | `bool` | `torch.isnan(param).any().item()` |
| `has_inf` | `bool` | `torch.isinf(param).any().item()` |

### CodeSnippet — PyTorch Code for Inspection (Tier 3)

Populated only after the agent calls `inspect_code`. Contains actual PyTorch training code with a bug.

| Field | Type | Description |
|---|---|---|
| `code` | `str` | Multi-line Python/PyTorch code string |
| `filename` | `str` | Simulated filename (e.g., `train.py`) |
| `line_count` | `int` | Total number of lines |
| `imports` | `list[str]` | List of import statements in the code |
| `hint` | `Optional[str]` | Optional hint about where to look (for easy variants) |

### DataBatchStats — The Leakage Signal

Populated only after the agent calls `inspect_data_batch`. Fields:

| Field | Type | Description |
|---|---|---|
| `label_distribution` | `dict[int, float]` | Class ID to fraction of batch |
| `feature_mean` | `float` | Mean feature value across sampled batch |
| `feature_std` | `float` | Standard deviation of feature values |
| `null_count` | `int` | Number of null or NaN values found |
| `class_overlap_score` | `float` | 0.0 = clean dataset; above 0.5 definitively indicates leakage |
| `batch_size` | `int` | Integer size of sampled batch |
| `duplicate_ratio` | `float` | Fraction of samples that duplicate validation samples |

The `class_overlap_score` is the primary diagnostic signal for the data leakage task. It is generated parametrically from the scenario's `leakage_pct` parameter, varying meaningfully between episodes while remaining proportional to the actual injected fault severity.

### GradientStats — Per-Layer Gradient Information

Populated for each layer only after the agent calls `inspect_gradients`. These come from real `torch.autograd` gradient tensors computed during the fault injection pass.

| Field | Type | Description |
|---|---|---|
| `layer_name` | `str` | String identifier matching `model.named_parameters()` |
| `norm_history` | `list[float]` | Last 5 gradient norms for this layer (from real `torch.norm(param.grad)`) |
| `mean_norm` | `float` | Average over the history |
| `max_norm` | `float` | Maximum value in the history |
| `is_exploding` | `bool` | True when mean norm exceeds 10.0 |
| `is_vanishing` | `bool` | True when mean norm falls below 1e-6 |

For the BatchNorm eval mode task, the red herring is expressed here: the FC layer shows a spike in `norm_history` with `is_exploding: False` (it spiked but recovered), while all other layers show entirely normal values.

### EpisodeState — Tracks Agent History

Included in every observation so the agent can always see its own diagnostic history.

| Field | Type | Purpose |
|---|---|---|
| `step_count` | `int` | Increments on every `step()` call |
| `gradients_inspected` | `bool` | Set true after first `inspect_gradients` call |
| `gradients_were_normal` | `bool` | Gate for context-gated red herring penalty |
| `data_inspected` | `bool` | Set true after first `inspect_data_batch` call |
| `model_modes_inspected` | `bool` | Set true after `inspect_model_modes` call |
| `model_weights_inspected` | `bool` | Set true after first `inspect_model_weights` call |
| `code_inspected` | `bool` | Set true after first `inspect_code` call |
| `fix_action_taken` | `bool` | Gates availability of `restart_run` |
| `restart_after_fix` | `bool` | Gates terminal convergence reward |
| `diagnosis_submitted` | `bool` | Set true after `mark_diagnosed` is called |
| `actions_taken` | `list[str]` | Ordered list of action type strings taken |

### TrainingConfig — Typed Hyperparameter Configuration

A fully typed model replacing untyped dictionaries. Every configurable hyperparameter has an explicit type and semantic meaning.

| Field | Type | Description |
|---|---|---|
| `learning_rate` | `float` | Current learning rate |
| `weight_decay` | `float` | L2 regularization strength |
| `batch_size` | `int` | Training batch size |
| `hidden_dim` | `int` | Hidden layer dimensionality |
| `num_layers` | `int` | Number of model layers |
| `optimizer` | `str` | Optimizer type (e.g., "adam", "sgd") |
| `dropout_rate` | `float` | Dropout probability |
| `gradient_clip_norm` | `Optional[float]` | Gradient clipping threshold, null if disabled |

The `modify_config` action's `target` field must match one of these field names exactly. This makes the config self-documenting — the agent can read the field names to understand what is configurable.

### MLTrainingObservation — The Full Observation

Fields visible to the agent at every step:

| Field | Type | Notes |
|---|---|---|
| `run_id` | `str` | Identifier for the current episode |
| `framework` | `str` | Always "pytorch" in this environment |
| `epoch` | `int` | Current epoch the simulated run is at |
| `training_loss_history` | `list[float]` | Last 20 epochs of training loss |
| `val_loss_history` | `list[float]` | Last 20 epochs of validation loss |
| `val_accuracy_history` | `list[float]` | Last 20 epochs of validation accuracy |
| `gradient_stats` | `list[GradientStats]` | Empty until `inspect_gradients` is called |
| `model_weight_stats` | `Optional[list[ModelWeightStats]]` | Null until `inspect_model_weights` is called |
| `gpu_memory_used_gb` | `float` | Red herring in Task 5 — high but not the cause |
| `gpu_memory_total_gb` | `float` | Fixed reference value |
| `learning_rate` | `float` | Current learning rate from config |
| `current_config` | `TrainingConfig` | Full typed hyperparameter config, always visible |
| `error_log` | `Optional[str]` | Populated if training crashed (Task 1) |
| `data_batch_stats` | `Optional[DataBatchStats]` | Null until `inspect_data_batch` is called |
| `model_mode_info` | `Optional[dict[str, str]]` | Null until `inspect_model_modes` is called |
| `code_snippet` | `Optional[CodeSnippet]` | Null until `inspect_code` is called |
| `available_actions` | `list[str]` | Dynamic list of currently valid action strings |
| `episode_state` | `EpisodeState` | Full episode state, always visible |
| `notes` | `Optional[str]` | Carries red herring text where applicable |

### MLTrainingAction — What the Agent Can Do

The `action_type` field is a closed literal union:

| Action | Description | Prerequisite |
|---|---|---|
| `inspect_gradients` | Populates `gradient_stats` with real `torch.autograd` gradient data | None |
| `inspect_data_batch` | Populates `data_batch_stats` in next observation | None |
| `inspect_model_modes` | Populates `model_mode_info` in next observation | None |
| `inspect_model_weights` | Populates `model_weight_stats` with real `state_dict()` data | None |
| `inspect_code` | Populates `code_snippet` with PyTorch training code | None |
| `modify_config` | Change a hyperparameter | None |
| `add_callback` | Add a training callback (gradient clipping, scheduler) | None |
| `rollback_checkpoint` | Revert to last saved checkpoint | `checkpoint_exists` in state |
| `replace_optimizer` | Swap the optimizer type | None |
| `patch_data_loader` | Fix a data preprocessing or leakage bug | None |
| `fix_model_mode` | Call `model.train()` to restore correct layer modes | None |
| `fix_code` | Submit a code fix | `code_inspected` must be true |
| `restart_run` | Restart training | `fix_action_taken` must be true |
| `mark_diagnosed` | Submit root cause diagnosis. Closes the episode. | Requires `diagnosis` field |

**Actions with required additional fields:**

| Action | Required Fields | Type | Description |
|---|---|---|---|
| `modify_config` | `target` | `str` | Must match a field name in `TrainingConfig` (e.g., `"learning_rate"`, `"weight_decay"`) |
| `modify_config` | `value` | `float \| int \| str` | New value for the target config field |
| `fix_code` | `line` | `int` | Line number in the code snippet to replace |
| `fix_code` | `replacement` | `str` | Replacement code string for the specified line |
| `mark_diagnosed` | `diagnosis` | `str` | Must be one of: `lr_too_high`, `vanishing_gradients`, `data_leakage`, `overfitting`, `batchnorm_eval_mode`, `code_bug` |

### Dynamic available_actions

The `available_actions` field changes based on episode state:

- `restart_run` only appears after `fix_action_taken` becomes true
- `rollback_checkpoint` only appears after a checkpoint has been created in the episode
- `fix_code` only appears after `code_inspected` becomes true
- `mark_diagnosed` disappears after `diagnosis_submitted` becomes true

This prevents the agent from taking nonsensical actions and makes the environment self-documenting.

---

## 11. The Six Core Tasks

### Task 1 — Exploding Gradients (Easy, `task_001`)

**Injected fault:** Learning rate set far too high for the architecture — sampled per episode from `[0.05, 0.08, 0.10, 0.15, 0.30]` against an expected value of `1e-3`.

**PyTorch model behavior:** The real model's forward+backward pass with the elevated LR produces gradient tensors with norms exceeding 10.0 on all layers. `torch.isnan()` returns true on loss after 2 passes. The `state_dict()` shows weights that have diverged from initialization.

**Initial observation:**
- Training loss starts at approximately 2.3 (normal), rises sharply by epoch 8, diverges with a NaN marker by epoch 12
- `error_log` is populated with a NaN loss error message
- `current_config` shows the elevated learning rate value
- Validation accuracy collapses along with training loss — no anomalies

**After `inspect_gradients`:**
- All layers show `is_exploding: True` (from real `torch.norm(param.grad)`)
- `norm_history` shows exponential growth across all four layers
- Signal is unambiguous — every layer is affected, no red herring

**After `inspect_model_weights`:**
- Weight norms are orders of magnitude above initialization
- `has_nan: True` on FC layer weights
- Confirms the model has diverged — additional evidence supporting the gradient finding

**Correct solution path:**
1. Call `inspect_gradients` — see exploding norms across all layers
2. Call `modify_config` targeting `learning_rate` with any value ≤ 0.001
3. Call `restart_run` (now available because `fix_action_taken` is true)
4. Observe loss convergence in the post-fix observation
5. Call `mark_diagnosed` with `lr_too_high`

**Grader scoring:**

| Action | Score | Condition |
|---|---|---|
| `inspect_gradients` called | +0.05 | Investigation credit |
| `modify_config` with valid LR reduction | +0.20 | Correct fix action |
| `restart_run` with convergence confirmed | +0.35 | Terminal verification |
| `mark_diagnosed(lr_too_high)` | +0.40 | Correct diagnosis |
| **Maximum** | **1.00** | |

**Difficulty calibration:** The signal is unambiguous. Exploding gradients across all layers with a NaN error log is the most recognizable failure pattern in ML. This task establishes the baseline competence floor.

**Expected rule-based baseline score:** ~0.85. LLM baseline score will be measured empirically.

---

### Task 2 — Vanishing Gradients (Easy, `task_002`)

**Injected fault:** Learning rate set extremely low — sampled per episode from `[1e-6, 5e-6, 1e-5]` — combined with depth multiplier that exacerbates gradient decay through deeper layers.

**PyTorch model behavior:** The real model's backward pass with the tiny LR produces gradient tensors where deeper layers have norms below `1e-6`. The `state_dict()` shows weights barely changed from initialization values — the model is essentially not learning.

**Initial observation:**
- Training loss starts at approximately 2.3 and barely decreases — hovers within 0.05 of the initial value across 20 epochs
- Validation accuracy stays near random chance (approximately 10% for CIFAR-10 style classification)
- No error log, no crash — the run appears to be "training" normally but making no progress
- `current_config` shows the very low learning rate, but a low LR is a common conservative choice and is not automatically suspicious

**After `inspect_gradients`:**
- Deeper layers (conv2, conv3) show `is_vanishing: True` with `mean_norm` below `1e-6`
- Shallow layers (conv1) show gradients that are small but above the vanishing threshold
- The gradient decay pattern across layers is the diagnostic signal — norms decrease by 2–3 orders of magnitude from shallow to deep

**After `inspect_model_weights`:**
- All weight norms are within ~1% of their initialization values — confirms the model has barely updated
- No NaN or Inf values — the model is healthy, just not learning

**Red herring:**
- `notes` field: "Training resumed from a checkpoint saved at epoch 0 — early learning rate warmup may still be in effect."
- This provides a plausible explanation for slow progress that an undisciplined agent may accept without further investigation

**Correct solution path:**
1. Call `inspect_gradients` — see vanishing norms in deeper layers
2. Call `modify_config` targeting `learning_rate` with a value ≥ 0.001
3. Call `restart_run`
4. Observe loss beginning to decrease meaningfully
5. Call `mark_diagnosed` with `vanishing_gradients`

**Grader scoring:**

| Action | Score | Condition |
|---|---|---|
| `inspect_gradients` called | +0.05 | Investigation credit |
| `modify_config` with valid LR increase | +0.20 | Correct fix action |
| `restart_run` with convergence confirmed | +0.35 | Terminal verification |
| `mark_diagnosed(vanishing_gradients)` | +0.40 | Correct diagnosis |
| **Maximum** | **1.00** | |

**Difficulty calibration:** Similar to Task 1 in directness, but subtler — the run doesn't crash, it just makes no progress. Distinguishing "vanishing" from "just slow" requires checking the layer-by-layer gradient pattern.

**Expected rule-based baseline score:** ~0.80. LLM baseline score will be measured empirically.

---

### Task 3 — Silent Data Leakage (Medium, `task_003`)

**Injected fault:** A bug in the dataset splitter causes a percentage of validation samples — drawn per episode from `[0.12, 0.18, 0.22, 0.28]` — to appear in training batches.

**Initial observation:**
- Training loss decreases normally across all epochs
- Validation accuracy climbs to a suspiciously high value by epoch 3 — calibrated between 82% and 93% depending on the episode's leakage percentage
- No error log, no crash, no gradient anomalies of any kind
- `notes` field: "Model architecture upgraded from 2-layer to 4-layer CNN at epoch 2. Performance improvement may reflect increased model capacity."

**Red herrings:**
- The notes field provides a plausible causal explanation for the high accuracy — a recent architecture upgrade could legitimately explain a performance jump. However, careful inspection reveals the accuracy is suspiciously high from epoch 1, *before* the claimed upgrade at epoch 2. An agent that reasons about timing will see through this; an agent that accepts the note at face value will not investigate the data pipeline.
- After `inspect_gradients`: all layers return values with a mild elevation on one layer (not exploding, `is_exploding: False`) that does not indicate gradient issues. This is important: an agent that inspects gradients first gets a near-negative result that should redirect investigation toward the data pipeline, but the mild elevation may cause undisciplined agents to investigate gradient-related fixes instead.

**After `inspect_data_batch`:**
- `class_overlap_score` is high — scaled proportionally to the episode's leakage percentage, ranging from approximately 0.68 to 0.88
- `duplicate_ratio` reflects the leakage percentage directly
- These values definitively confirm leakage

**After `inspect_model_weights`:**
- Weights look normal — no NaN, no divergence. This confirms the model itself is fine; the problem is in the data pipeline.

**Correct solution path:**
1. Observe suspicious validation accuracy
2. Call `inspect_data_batch` — see high `class_overlap_score`
3. Call `patch_data_loader`
4. Call `restart_run`
5. Observe validation accuracy drop to a realistic level and then improve properly
6. Call `mark_diagnosed` with `data_leakage`

**Grader scoring:**

| Action | Score | Condition |
|---|---|---|
| `inspect_data_batch` called | +0.05 | Investigation credit |
| `patch_data_loader` called | +0.30 | Correct fix action |
| Validation accuracy normalizes after restart | +0.30 | Verified fix |
| `mark_diagnosed(data_leakage)` | +0.35 | Correct diagnosis |
| **Maximum** | **1.00** | |

**Why investigation credit matters:** An agent that jumps directly to `patch_data_loader` without calling `inspect_data_batch` gets a maximum of 0.95 — losing the investigation credit. This correctly differentiates an agent that reasoned through the evidence from an agent that guessed correctly.

**Expected rule-based baseline score:** ~0.70. LLM baseline score will be measured empirically.

---

### Task 4 — Overfitting (Medium, `task_004`)

**Injected fault:** No regularization (weight decay sampled from `[0.0, 0.0001, 0.001]` — always ineffective), combined with a model that is too large for the dataset size. The model memorizes training data and fails to generalize.

**Initial observation:**
- Training loss decreases steadily and reaches very low values (approaching 0.01 by epoch 15)
- Validation loss initially decreases, then diverges upward starting at a sampled epoch from `[5, 8, 12]`
- Validation accuracy plateaus then degrades while training accuracy continues to climb toward 99%+
- No error log, no crash — the training metrics look "great" if you only look at training loss
- `current_config` shows a large model (high `hidden_dim`, many layers) with minimal or zero weight decay

**Red herrings:**
- `notes` field: "Dataset augmentation was disabled for this run to speed up training. Re-enabling may improve generalization." — This is partially true (augmentation helps) but is not the root cause. An agent that adds augmentation without addressing regularization treats the symptom, not the disease.
- After `inspect_gradients`: gradients are healthy — normal norms, no exploding or vanishing. The FC layer shows slightly larger norms than other layers (consistent with a large model), which may mislead toward architecture concerns.

**After `inspect_data_batch`:**
- `class_overlap_score` is 0.0 — no leakage
- `duplicate_ratio` is 0.0 — clean dataset
- Feature statistics are normal
- This is important: the data is clean, confirming the problem is the model, not the data

**After `inspect_model_weights`:**
- FC layer weights have grown significantly from initialization — large norms indicate the model is memorizing
- No NaN or Inf, but weight magnitudes are 5-10x initialization values
- Provides additional evidence that the model capacity is too high relative to the data

**Correct solution path:**
1. Observe the train-val divergence pattern
2. Call `inspect_data_batch` to rule out data leakage (overlap_score = 0.0 confirms clean data)
3. Call `modify_config` targeting `weight_decay` with a meaningful value (≥ 0.01) OR call `add_callback` to add early stopping
4. Call `restart_run`
5. Observe val loss stabilizing and improving
6. Call `mark_diagnosed` with `overfitting`

**Grader scoring:**

| Action | Score | Condition |
|---|---|---|
| `inspect_data_batch` called | +0.05 | Investigation credit (ruling out leakage) |
| `modify_config` with regularization fix OR `add_callback` with early stopping | +0.25 | Correct fix action |
| `restart_run` with convergence confirmed | +0.30 | Terminal verification |
| `mark_diagnosed(overfitting)` | +0.40 | Correct diagnosis |
| **Maximum** | **1.00** | |

**Difficulty calibration:** Medium — the train-val divergence is a well-known pattern, but the clean data batch stats must be checked to distinguish overfitting from data leakage (both produce suspiciously good training metrics). The red herring note about augmentation tests whether the agent treats root causes or symptoms.

**Expected rule-based baseline score:** ~0.65. LLM baseline score will be measured empirically.

---

### Task 5 — BatchNorm Eval Mode with Compound Misleading Signals (Hard, `task_005`)

**Injected fault:** `model.eval()` was called before training began and `model.train()` was never called. All layers — including BatchNorm layers — remain in inference mode throughout training.

This is one of the most common and most painful real PyTorch mistakes. Forgetting to call `model.train()` before the training loop causes BatchNorm to use accumulated running statistics from initialization rather than batch statistics, and Dropout to stop dropping neurons — silently corrupting training without any error or obvious crash.

**PyTorch model behavior:** The real model is instantiated with `model.eval()` before the fault injection forward passes. The `state_dict()` shows BatchNorm `running_mean` and `running_var` frozen at initialization values. The `model.training` flag is `False` on all modules — this is what `inspect_model_modes` reveals.

**What the failure produces:**
- Training loss appears roughly normal with higher-than-expected variance
- Validation accuracy starts at approximately 76% and degrades slowly — approximately 1 to 2 percentage points per epoch — easy to miss without carefully examining the full history

**The deliberate red herring signals:**

| Signal | Description | Why it misleads |
|---|---|---|
| FC layer gradient spike | Norm spike at epochs 5–7, intensity factor from `[0.8, 2.5]`. Does not sustain. `is_exploding: False`. | Looks like a gradient issue. An undisciplined agent calls `add_callback` to clip. |
| GPU memory at 91% | High enough to appear concerning | Looks like an OOM risk. An undisciplined agent tries to reduce batch size. |
| Elevated training loss variance | Plausibly consistent with a learning rate issue | Looks like LR instability. Misdirects toward config modification. |
| `error_log` warning | "Warning: GPU memory pressure detected, consider reducing batch size or enabling gradient checkpointing" | Directly competes with the correct investigation path. |
| Conv1 layer near-vanishing gradient | `mean_norm` at 0.0003 — above the `1e-6` vanishing threshold but low enough to appear suspicious | Creates a second gradient anomaly alongside the FC spike. |

**The trap for undisciplined agents:**
1. See the gradient spike → call `add_callback` to add gradient clipping (wrong fix — `is_exploding` is false on all layers)
2. Read the error log warning → try to reduce batch size or add gradient checkpointing (wrong fix)
3. See the near-vanishing conv1 gradient → investigate vanishing gradients as a secondary hypothesis (wrong direction)
4. Miss the slow validation accuracy degradation because the 1–2% per epoch trend is easy to overlook

**After `inspect_model_weights`:**
- BatchNorm `running_mean` and `running_var` are at initialization values (zeros and ones) — they haven't been updated because the model is in eval mode
- This is a strong additional signal, but requires knowing that frozen BN stats are abnormal

**The correct investigation path:**
1. Call `inspect_gradients` — spike is only in FC layer, all other layers normal, `is_exploding: False` on all layers. This sets `gradients_were_normal: True` in episode state.
2. Recognize that gradients are not the problem despite the spike. Redirect investigation.
3. Call `inspect_model_modes` — reveals every layer is in "eval" mode. This is the definitive finding.
4. Call `fix_model_mode`
5. Call `restart_run` — post-fix validation accuracy stabilizes and improves
6. Call `mark_diagnosed` with `batchnorm_eval_mode`

**Grader scoring:**

| Action | Score | Condition |
|---|---|---|
| `inspect_gradients` called | +0.05 | Investigation credit |
| `inspect_model_modes` called | +0.05 | Investigation credit — the revealing action |
| `add_callback` after `gradients_were_normal` is true | −0.20 | Context-gated red herring penalty |
| `fix_model_mode` called | +0.25 | Correct fix action |
| `restart_run` with convergence confirmed | +0.30 | Terminal verification |
| `mark_diagnosed(batchnorm_eval_mode)` | +0.40 | Correct diagnosis |
| **Maximum score** | **1.00** | Capped |
| **Score if agent chases gradient red herring** | **0.80–0.85** | Depends on other actions |

**The context-gated penalty — why this matters:**

The −0.20 penalty for calling `add_callback` only fires when two conditions are both true: `gradients_inspected` is true AND `gradients_were_normal` is true. It does **not** fire before any investigation has occurred.

An agent that calls `add_callback` at step 1 is following a reasonable prior — gradient clipping is a defensible first response to training instability that has not yet been investigated. No penalty. An agent that calls `inspect_gradients`, sees that `is_exploding` is false on all layers, and then still adds gradient clipping is explicitly ignoring the counter-evidence it just collected. That agent deserves the penalty.

**Expected rule-based baseline score:** ~0.45. LLM baseline expected to score significantly higher due to reasoning about compound red herrings — exact score will be measured empirically.

---

### Task 6 — PyTorch Code Bug (Hard, `task_006`)

**Injected fault:** A bug in the PyTorch training code that cannot be diagnosed from metrics or gradients alone — the agent must read and understand the actual code.

This task tests what Meta cares about most: **can an AI agent debug PyTorch code?** The previous 5 tasks test metric interpretation and model inspection. Task 6 tests code understanding — a fundamentally different and harder capability.

**Bug variants (sampled per episode):**

| Bug ID | Buggy Code | Correct Code | Symptom |
|---|---|---|---|
| `eval_mode` | `model.eval()` before training loop | `model.train()` | Same as Task 5, but agent must find it *in code* |
| `detach_loss` | `loss = criterion(model(x), y).detach()` | Remove `.detach()` | Loss decreases in log but model doesn't learn (gradients zeroed) |
| `zero_grad_missing` | Missing `optimizer.zero_grad()` in loop | Add `optimizer.zero_grad()` | Gradients accumulate across batches, erratic training |
| `inplace_relu` | `x = F.relu(x, inplace=True)` on a tensor requiring grad | `x = F.relu(x)` | RuntimeError or silent gradient corruption |

**Task 5 vs Task 6 `eval_mode` overlap:** Task 5 and Task 6's `eval_mode` variant share the same underlying bug (`model.eval()` before training). The critical distinction is the diagnostic pathway: Task 5 expects the agent to detect it from runtime signals via `inspect_model_modes`, while Task 6 expects the agent to find it by reading source code via `inspect_code`. **Task 6 always expects `code_bug` as the diagnosis regardless of the specific bug variant.** The agent must recognize that the bug was discovered in code, not inferred from metrics. The grader enforces this — submitting `batchnorm_eval_mode` on any Task 6 variant scores as a wrong diagnosis (−0.30).

**Initial observation:**
- Training metrics show various anomalies depending on the bug variant
- `error_log` may or may not be populated (some bugs crash, others fail silently)
- Gradient and weight inspection may show symptoms but do not reveal the root cause
- `notes` field contains a misleading hint (e.g., "Try adjusting the learning rate schedule" for the `detach_loss` variant)

**After `inspect_code`:**
The agent receives a PyTorch training code snippet (15-25 lines) containing the bug:

```python
# Example: detach_loss variant
import torch
import torch.nn as nn

model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y).detach()  # <-- BUG: detach breaks gradient flow
        loss.backward()
        optimizer.step()
```

The agent must identify the buggy line and submit a fix:

```json
{"action_type": "fix_code", "line": 13, "replacement": "        loss = criterion(output, batch_y)"}
```

**Correct solution path:**
1. Observe anomalous metrics
2. Call `inspect_gradients` and/or `inspect_model_weights` — symptoms are present but inconclusive
3. Call `inspect_code` — see the PyTorch training code with the bug
4. Identify the buggy line
5. Call `fix_code` with the correct line number and replacement
6. Call `restart_run`
7. Call `mark_diagnosed` with **`code_bug`** (always `code_bug` regardless of the specific bug variant — see overlap note above)

**Grader scoring:**

| Action | Score | Condition |
|---|---|---|
| `inspect_code` called | +0.05 | Investigation credit |
| `fix_code` with correct line and valid fix | +0.30 | Correct code fix |
| `restart_run` with convergence confirmed | +0.25 | Terminal verification |
| `mark_diagnosed(code_bug)` | +0.40 | Correct diagnosis |
| **Maximum** | **1.00** | |
| `fix_code` with wrong line or invalid fix | −0.10 | Wrong code fix penalty |

**Why this task is a winner-tier differentiator:**
- It directly tests PyTorch code comprehension — aligned with Meta's core interest
- No existing OpenEnv environment includes code-level debugging
- The bug variants are all real PyTorch mistakes that cause real production failures
- It fundamentally changes what the agent must reason about — from metrics to source code

**Expected rule-based baseline score:** ~0.30 (the heuristic agent can reach `inspect_code` but cannot reliably identify the bug across all 4 variants). LLM baseline expected to score significantly higher as it can parse code and identify bug patterns — exact score will be measured empirically.

---

## 12. Reward Function — Design and Rationale

### The Formula

The reward function computes a float at every `step()` call. The grader function is separate — it computes a normalized 0.0 to 1.0 score at episode end based on the full `EpisodeState`. These are not the same thing and must not be conflated.

**Component 1 — Flat step penalty: −0.01**

Applied unconditionally on every step. This is a flat constant, never multiplied by step count. A cumulative-multiplied penalty (e.g., −0.01 × step_count) creates a doom spiral: at step 30 a single action costs −0.30, which causes agents to learn to submit any random diagnosis at step 1 to stop the accumulating penalty. The flat −0.01 encourages efficiency without destroying exploratory behavior.

**Component 2 — Investigation rewards: +0.05 each, first-time only**

Applied when the agent calls `inspect_gradients`, `inspect_data_batch`, `inspect_model_modes`, `inspect_model_weights`, or `inspect_code` for the first time. The "first-time only" constraint prevents reward farming by repeating the same call. These rewards reflect the real information value of investigation — even a negative result (gradients are normal) meaningfully narrows the hypothesis space.

**Component 3 — Context-gated red herring penalty: −0.20**

Applied when the agent calls `add_callback` AND `gradients_were_normal` is already true. Since no task in this environment has gradient clipping as the correct fix, the penalty fires whenever an agent adds gradient clipping after already observing that gradients are normal — a clear case of ignoring gathered evidence.

**Component 4 — Invalid action penalty: −0.05**

Applied when the agent calls an action not in `available_actions`. Returns immediately without further processing. Discourages systematic probing of unavailable actions.

**Component 5 — Diagnosis outcome: +0.50 correct / −0.30 wrong**

The wrong diagnosis penalty is larger than the investigation rewards to prevent the agent from recovering from a wrong diagnosis by farming investigation rewards. This incentivizes evidence gathering before commitment.

**Component 6 — Terminal convergence reward: +0.40, gated**

Applied when `restart_run` is called AND `fix_action_taken` is already true AND `restart_after_fix` is true AND the post-fix convergence check passes.

**Component 7 — Wrong code fix penalty: −0.10**

Applied when the agent calls `fix_code` with an incorrect line number or an invalid replacement string. This is separate from the invalid action penalty because the action itself is valid (the agent has inspected code), but the fix is wrong.

### Reward Summary

| Event | Reward | Gate Condition |
|---|---|---|
| Any step taken | −0.01 | Unconditional, flat constant |
| `inspect_gradients` first time | +0.05 | `not state.gradients_inspected` |
| `inspect_data_batch` first time | +0.05 | `not state.data_inspected` |
| `inspect_model_modes` first time | +0.05 | `not state.model_modes_inspected` |
| `inspect_model_weights` first time | +0.05 | `not state.model_weights_inspected` |
| `inspect_code` first time | +0.05 | `not state.code_inspected` |
| `add_callback` after normal gradients | −0.20 | `gradients_were_normal is True` |
| Invalid action | −0.05 | Action not in `available_actions` |
| Wrong code fix | −0.10 | `fix_code` with incorrect line/replacement |
| Correct diagnosis | +0.50 | `action.diagnosis == true_root_cause` |
| Wrong diagnosis | −0.30 | `action.diagnosis != true_root_cause` |
| Convergence after fix and restart | +0.40 | `fix_action_taken AND restart_after_fix AND convergence` |

### Grader vs. Reward Function — Critical Distinction

The reward function returns a float at each step and is used during RL training to shape agent behavior. It is a per-action signal.

**The grader function — not the reward function — is what the `POST /grader` endpoint returns.** The grader returns a single normalized score from 0.0 to 1.0 at episode end. It is used for the `/grader` endpoint, Phase 1 validation, and Phase 2 benchmarking. These are implemented as separate modules. The grader evaluates the full `EpisodeState` holistically — it checks which key actions were taken, whether the correct fix was applied, whether the diagnosis is correct, and how efficiently the agent reached the solution. **It is not a sum of step rewards.**

---

## 13. Environment Lifecycle

### Episode State Transitions

At `reset(task_id)`, a fresh `ScenarioParams` object is sampled with randomized fault parameters and a new seed. A real PyTorch model is instantiated, the fault is injected, 1-2 forward+backward passes are run, and the resulting gradient/weight state is frozen. A fresh `EpisodeState` is initialized with all boolean fields false and `step_count` at zero. The initial `MLTrainingObservation` is constructed with populated loss history and accuracy history but empty `gradient_stats` and null `data_batch_stats`, `model_mode_info`, `model_weight_stats`, and `code_snippet`.

| Action Called | State Change | Available Actions Change |
|---|---|---|
| `inspect_gradients` | `gradients_inspected = True`, `gradients_were_normal` set based on result | No change |
| `inspect_data_batch` | `data_inspected = True` | No change |
| `inspect_model_modes` | `model_modes_inspected = True` | No change |
| `inspect_model_weights` | `model_weights_inspected = True` | No change |
| `inspect_code` | `code_inspected = True` | `fix_code` appears |
| Any fix action | `fix_action_taken = True` | `restart_run` appears |
| `restart_run` while `fix_action_taken` | `restart_after_fix = True` | `rollback_checkpoint` appears |
| `mark_diagnosed` | `diagnosis_submitted = True`, `done = True` | `mark_diagnosed` disappears |
| Step limit reached | No state change | `done = True` returned |

### Optimal Trajectory for Task 5 (BatchNorm Eval Mode — Hardest Metric Task)

| Step | Action | Reward | Cumulative | State Change |
|---|---|---|---|---|
| reset | — | — | — | Fresh model with `model.eval()`, all booleans false |
| 1 | `inspect_gradients` | −0.01 + 0.05 = **+0.04** | +0.04 | `gradients_inspected = True`, `gradients_were_normal = True` |
| 2 | `inspect_model_modes` | −0.01 + 0.05 = **+0.04** | +0.08 | `model_modes_inspected = True`. All layers reveal "eval" mode. |
| 3 | `fix_model_mode` | **−0.01** | +0.07 | `fix_action_taken = True`. `restart_run` now available. |
| 4 | `restart_run` | −0.01 + 0.40 = **+0.39** | +0.46 | `restart_after_fix = True`. Post-fix val accuracy recovering. |
| 5 | `mark_diagnosed(batchnorm_eval_mode)` | −0.01 + 0.50 = **+0.49** | **+0.95** | `done = True` |

Grader score for this trajectory: **1.00**

### Optimal Trajectory for Task 6 (Code Bug — Hardest Overall)

| Step | Action | Reward | Cumulative | State Change |
|---|---|---|---|---|
| reset | — | — | — | Fresh model with code bug injected |
| 1 | `inspect_gradients` | −0.01 + 0.05 = **+0.04** | +0.04 | Symptoms visible but inconclusive |
| 2 | `inspect_code` | −0.01 + 0.05 = **+0.04** | +0.08 | Code snippet revealed. `fix_code` now available. |
| 3 | `fix_code(line=13, replacement="...")` | **−0.01** | +0.07 | `fix_action_taken = True`. `restart_run` now available. |
| 4 | `restart_run` | −0.01 + 0.40 = **+0.39** | +0.46 | Training converges after fix. |
| 5 | `mark_diagnosed(code_bug)` | −0.01 + 0.50 = **+0.49** | **+0.95** | `done = True` |

Grader score for this trajectory: **1.00**

---

## 14. OpenEnv Spec Compliance

### openenv.yaml

```yaml
name: pytorch-training-debugger
version: "1.0.0"
description: |
  PyTorch-native fault injection engine for training failure debugging.
  An AI agent investigates, diagnoses, fixes, and verifies broken
  training runs using real torch.nn.Module models, torch.autograd
  gradients, state_dict() weight inspection, and PyTorch code-level
  debugging. 6 tasks across 3 difficulty tiers with context-gated
  reward shaping and a live diagnostic dashboard.
framework: openenv
tags: [ml-debugging, pytorch, reinforcement-learning, root-cause-analysis, fault-injection, code-debugging]

observation_space:
  type: MLTrainingObservation
  description: "Training run snapshot with progressive reveal — gradients, weights, data stats, model modes, and code snippets revealed on inspection"

action_space:
  type: MLTrainingAction
  description: "Investigation, fix, code-fix, and diagnosis actions with dynamic availability"

tasks:
  - id: task_001
    difficulty: easy
    max_steps: 20
    param_ranges:
      learning_rate: [0.05, 0.08, 0.10, 0.15, 0.30]

  - id: task_002
    difficulty: easy
    max_steps: 20
    param_ranges:
      learning_rate: [1e-6, 5e-6, 1e-5]
      depth_multiplier: [1.0, 1.5, 2.0]

  - id: task_003
    difficulty: medium
    max_steps: 25
    param_ranges:
      leakage_pct: [0.12, 0.18, 0.22, 0.28]

  - id: task_004
    difficulty: medium
    max_steps: 25
    param_ranges:
      weight_decay: [0.0, 0.0001, 0.001]
      divergence_epoch: [5, 8, 12]

  - id: task_005
    difficulty: hard
    max_steps: 30
    param_ranges:
      red_herring_intensity: [0.8, 2.5]

  - id: task_006
    difficulty: hard
    max_steps: 30
    param_ranges:
      bug_type: [eval_mode, detach_loss, zero_grad_missing, inplace_relu]

reward:
  range: [-1.0, 1.0]
  shaped: true
  step_penalty: -0.01
  investigation_bonus: 0.05
  max_investigation_bonus: 0.25
  correct_diagnosis: 0.50
  terminal_convergence: 0.40

endpoints:
  # Required by hackathon
  websocket: "/ws"
  tasks: "GET /tasks"
  grader: "POST /grader"
  baseline: "POST /baseline"
  health: "GET /health"
  # Environment-specific bonuses (not required by hackathon)
  dashboard: "GET /dashboard"
  validation_report: "GET /validation-report"
```

**Note on reward range:** The range is stated as [−1.0, 1.0]. The maximum possible step rewards sum to approximately 0.95 before the per-step penalty total. The range is a hard cap, not a guarantee that optimal play reaches exactly 1.0 — the 1.05 theoretical max is capped.

### Required Endpoints

Per the Round 1 problem statement, the following endpoints must be exposed. The first five are required by the hackathon; the last two are environment-specific bonuses.

| Endpoint | Transport | Required By | Description |
|---|---|---|---|
| `/ws` | WebSocket | OpenEnv framework | Handles `reset`, `step`, `state` messages. |
| `GET /tasks` | HTTP | Hackathon problem statement | Returns task list with IDs, difficulties, and the full `MLTrainingAction` JSON schema |
| `POST /grader` | HTTP | Hackathon problem statement | Returns grader score (0.0–1.0) for the most recently completed episode |
| `POST /baseline` | HTTP | Hackathon problem statement | Triggers baseline inference run, returns scores for all tasks |
| `GET /health` | HTTP | Hackathon problem statement | Returns server status and confirms trajectory generation is ready |
| `GET /dashboard` | HTTP | Bonus (this environment) | Serves live diagnostic dashboard (HTML/JS) for episode visualization |
| `GET /validation-report` | HTTP | Bonus (this environment) | Serves PyTorch validation fidelity reports (scores + comparison plots) |

### Grader Endpoint — Edge Case Behavior

| Scenario | Response | HTTP Status |
|---|---|---|
| No episode completed yet | `{"score": null, "error": "no_completed_episode"}` | 200 |
| Episode in progress (not yet done) | `{"score": null, "error": "episode_in_progress"}` | 200 |
| Episode completed, score available | `{"score": 0.85, "task_id": "task_003", "steps": 6}` | 200 |
| Multiple concurrent sessions | Score returned for the session specified by `session_id` query param; if omitted, returns the most recently completed episode across all sessions | 200 |

The grader always returns HTTP 200 with a JSON body. The `score` field is `null` when no valid score is available. The auto-validator calls `/grader` immediately after running a baseline episode — the grader must return the score for that specific episode, not stale data from a previous run.

---

## 15. Docker and Deployment

### Dockerfile Design

The image is based on `python:3.12-slim` with PyTorch CPU as a runtime dependency. PyTorch is required because the fault injection engine instantiates real `torch.nn.Module` models at `reset()` time.

**Core dependencies (always installed):**
- `torch` (CPU-only, `--index-url https://download.pytorch.org/whl/cpu`) — real model instantiation, gradient computation, weight inspection
- `openenv-core` — the OpenEnv framework (WebSocket server and base classes)
- `pydantic` ≥ 2.0 — typed models
- `fastapi` + `uvicorn` — HTTP routes layered on the framework server
- `openai` — baseline inference script

**Docker image size breakdown:**

| Component | Estimated Size |
|---|---|
| `python:3.12-slim` base | ~150MB |
| `torch` CPU-only wheel | ~150MB |
| `openenv-core` + `fastapi` + `uvicorn` + `pydantic` + `openai` | ~50MB |
| Application code + static assets | ~5MB |
| **Total target** | **~350–400MB** |

- PyTorch CPU-only wheel avoids the ~2GB CUDA build
- Multi-stage build: validation suite runs in build stage, fidelity reports are pre-computed locally and copied as static files to runtime image (avoids running real training during Docker build)

```dockerfile
# Single-stage runtime — includes PyTorch for real model operations
# Validation fidelity reports are pre-computed locally and committed to the repo
FROM python:3.12-slim
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install openenv-core pydantic fastapi uvicorn openai
COPY validation/reports/ validation/reports/
COPY ml_training_debugger/ ml_training_debugger/
COPY server/ server/
COPY baseline_heuristic.py baseline_inference.py openenv.yaml ./
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### HF Spaces Deployment

The environment is deployed to a Hugging Face Space tagged with `openenv`. The Space runs the framework server on port 7860. The WebSocket endpoint at `/ws` is the primary interface. The HTTP endpoints are accessible at their paths on the same port.

The `/health` endpoint returns `{"status": "ready", "tasks": 6}` — satisfying the auto-validator's startup ping cleanly.

### Structured Logging

The server logs every significant event using Python's `logging` module at structured JSON format.

| Event | Log Level | Fields |
|---|---|---|
| `reset()` called | INFO | `session_id`, `task_id`, `scenario_seed` |
| `step()` called | INFO | `session_id`, `step_count`, `action_type`, `reward` |
| Episode completed | INFO | `session_id`, `task_id`, `steps_taken`, `grader_score`, `diagnosis`, `correct` |
| Invalid action received | WARNING | `session_id`, `action_type`, `reason`, `available_actions` |
| Grader called | INFO | `session_id`, `score`, `task_id` |
| Baseline run started | INFO | `task_id`, `model`, `temperature` |
| Error during step processing | ERROR | `session_id`, `action`, `error_type`, `traceback` |

Logs are written to stdout (captured by Docker and HF Spaces). No log file is created inside the container.

---

## 16. Error Handling and Edge Cases

### Action Validation

Every `step()` call validates the incoming action before processing:

| Error Condition | Behavior | Reward |
|---|---|---|
| `action_type` not a valid string from the action enum | Return observation unchanged with error note: `"Invalid action_type: {value}. Valid types: {list}"` | −0.05 |
| `action_type` valid but not in current `available_actions` | Return observation unchanged with error note | −0.05 |
| `modify_config` without `target` field | Return observation unchanged with error note: `"modify_config requires 'target' and 'value' fields"` | −0.05 |
| `modify_config` with `target` not in `current_config` keys | Return observation unchanged with error note: `"Unknown config key: {target}"` | −0.05 |
| `mark_diagnosed` without `diagnosis` field | Return observation unchanged with error note: `"mark_diagnosed requires 'diagnosis' field"` | −0.05 |
| `mark_diagnosed` with `diagnosis` not in enum | Return observation unchanged with error note: `"Invalid diagnosis: {value}. Valid: lr_too_high, vanishing_gradients, data_leakage, overfitting, batchnorm_eval_mode, code_bug"` | −0.05 |
| `fix_code` without `line` or `replacement` | Return observation unchanged with error note | −0.05 |
| Action sent after episode is done (`done = True`) | Return final observation with `done = True`, no state change | 0.0 |
| Step limit reached | Set `done = True`, return observation, no further actions processed | 0.0 |

All error cases return a valid `StepResult` with the current observation, the penalty reward, and `done = False` (unless the episode was already done). The environment never raises an unhandled exception from `step()` — all errors are caught, logged, and returned as structured responses.

### WebSocket Edge Cases

| Scenario | Behavior |
|---|---|
| Client disconnects mid-episode | Session state is retained for 60 seconds; reconnection to the same session ID resumes the episode. After timeout, session is cleaned up. |
| Client sends malformed JSON | WebSocket returns `{"error": "malformed_json", "detail": "..."}` and keeps the connection open. |
| Client calls `step()` before `reset()` | Returns `{"error": "no_active_episode", "detail": "Call reset(task_id) first"}`. |
| Client calls `reset()` during an active episode | Current episode is terminated (grader records score 0.0), new episode begins. |
| Rapid-fire steps (>100 steps/second) | No rate limiting — steps are processed sequentially per session. The step limit (20–30 per task) provides natural bounds. |

### HTTP Endpoint Edge Cases

| Scenario | Behavior |
|---|---|
| `POST /baseline` called while baseline is already running | Returns `{"error": "baseline_in_progress"}` with HTTP 409 |
| `GET /tasks` returns | Static JSON — no session required, always available |
| `POST /grader` with invalid `session_id` | Returns `{"score": null, "error": "unknown_session"}` with HTTP 200 |
| Server receives request before environment is initialized | `/health` returns `{"status": "initializing"}` with HTTP 503 |

---

## 17. Baseline Inference Design

### Two Baselines — Rule-Based and LLM

The environment ships with two baseline agents:

**1. Rule-Based Baseline (submission default for Phase 1 reproducibility, no API key required)**

**This is the submission baseline used for Phase 1 auto-validation reproducibility checks.** A deterministic heuristic agent implemented in `baseline_heuristic.py`. It requires no external API calls and runs entirely locally, guaranteeing bit-exact reproducibility on every platform. The decision logic follows a systematic elimination order:

1. Call `inspect_gradients`
2. If any layer has `is_exploding: True` → `modify_config(learning_rate, 0.001)` → `restart_run` → `mark_diagnosed(lr_too_high)` → **done**
3. If any layer has `is_vanishing: True` → `modify_config(learning_rate, 0.01)` → `restart_run` → `mark_diagnosed(vanishing_gradients)` → **done**
4. Call `inspect_data_batch`
5. If `class_overlap_score > 0.5` → `patch_data_loader` → `restart_run` → `mark_diagnosed(data_leakage)` → **done**
6. If val_loss diverging from train_loss and `class_overlap_score == 0.0` → `modify_config(weight_decay, 0.01)` → `restart_run` → `mark_diagnosed(overfitting)` → **done**
7. Call `inspect_model_modes` → if any layer in "eval" mode → `fix_model_mode` → `restart_run` → `mark_diagnosed(batchnorm_eval_mode)` → **done**
8. Call `inspect_code` → attempt pattern-matching fix for known bug patterns → `restart_run` → `mark_diagnosed(code_bug)` → **done**
9. If no condition matched → `mark_diagnosed(overfitting)` as fallback (ensures episode always terminates)

This baseline is the **default submission baseline** because it requires zero external dependencies and produces identical scores on every run trivially.

**2. LLM Baseline (optional demonstration, requires OpenAI API key)**

The LLM agent (`baseline_inference.py`) connects via the `MLTrainingEnv` client class, driven by the OpenAI API with `gpt-4o` at `temperature=0.0` and `seed=42`. The system prompt tells the agent it is an ML engineer debugging a PyTorch training run, provides the full list of valid diagnosis enum values and action types, and instructs the agent to always output a JSON object matching the `MLTrainingAction` schema.

**Important:** The LLM baseline is an optional demonstration of the environment's value — it is NOT the submission baseline used for Phase 1 auto-validation. The rule-based baseline is the primary submission baseline (see above).

### Reproducibility Requirements

Phase 1 auto-validation runs the submitted baseline script twice and verifies identical scores. The **rule-based baseline is the submission default** — it guarantees bit-exact reproducibility trivially (pure deterministic logic, no external API calls).

The LLM baseline uses `temperature=0.0` and `seed=42` for near-deterministic behavior, but OpenAI's seed parameter is best-effort and not guaranteed to produce identical outputs across runs. For this reason, the LLM baseline is provided as a **supplementary demonstration** of the score gap between heuristic and reasoning agents — not as the reproducibility-critical submission baseline.

Both baselines share deterministic environment-side state via `torch.manual_seed()` — each task ID maps to a fixed set of parameters at baseline time.

### Expected Baseline Scores

**Rule-based baseline (verified — deterministic, reproducible):**

| Task | Rule-Based Score | Rationale |
|---|---|---|
| `task_001` (Exploding gradients) | ~0.85 | Direct signal: `is_exploding` → reduce LR → restart → diagnose |
| `task_002` (Vanishing gradients) | ~0.80 | Direct signal: `is_vanishing` → increase LR → restart → diagnose |
| `task_003` (Data leakage) | ~0.70 | Requires inspecting data batch; heuristic follows correct order |
| `task_004` (Overfitting) | ~0.65 | Must rule out leakage first, then apply regularization |
| `task_005` (BatchNorm eval mode) | ~0.45 | Fixed investigation order may chase gradient red herring before checking model modes |
| `task_006` (Code bug) | ~0.30 | Pattern-matching on code is unreliable across all 4 bug variants |
| **Average** | **~0.63** | |

*These are design-time estimates based on the heuristic decision tree logic. Exact scores will be measured and published in the README after implementation.*

**LLM baseline (supplementary — requires OpenAI API key):**

LLM baseline scores are intentionally not published as specific numbers until measured against the implemented environment. The purpose of the LLM baseline is to demonstrate a **measurable score gap** between heuristic and reasoning-capable agents — particularly on Tasks 5 (compound red herrings) and 6 (code comprehension), where the rule-based agent's fixed decision tree falls short. The README will report measured LLM scores only after they are empirically verified.

The rule-based baseline scores lower on Tasks 5 and 6 because it follows a fixed investigation order (Task 5) and cannot reliably parse code bugs across all variants (Task 6). The LLM baseline is expected to score higher because it can reason about compound red herrings and parse PyTorch code — but the exact gap will be measured and reported, not assumed.

---

## 18. PyTorch Validation Suite

While the runtime environment uses real PyTorch models for gradient/weight data and parametric curves for loss histories, the project includes a **PyTorch validation suite** that proves the parametric curves are faithful to real PyTorch training behavior. This serves two purposes: it validates the fault injection engine's accuracy, and it demonstrates deep PyTorch expertise.

### What the Validation Suite Does

The suite lives in `validation/` and contains real PyTorch training scripts — one per failure scenario — that reproduce each fault in actual PyTorch code and compare the resulting metrics against the parametric simulation's output.

| Script | Real PyTorch Code | What It Proves |
|---|---|---|
| `validate_exploding_gradients.py` | Trains a CNN on CIFAR-10 with `lr=0.1` | Real loss divergence matches simulated curve shape |
| `validate_vanishing_gradients.py` | Trains a 12-layer MLP with `lr=1e-6` | Real gradient decay pattern matches simulated per-layer norms |
| `validate_data_leakage.py` | Injects 20% validation samples into training split | Real inflated val accuracy matches simulated leakage curve |
| `validate_overfitting.py` | Trains overparameterized model with `weight_decay=0.0` | Real train-val divergence matches simulated divergence epoch |
| `validate_batchnorm_eval.py` | Calls `model.eval()` then trains without `model.train()` | Real val accuracy degradation matches simulated decay rate |
| `validate_code_bugs.py` | Runs each of the 4 code bug variants | Confirms symptoms match expected failure patterns |

Each script:
1. Runs the real PyTorch training loop for 20 epochs
2. Captures the same metrics the simulation generates (loss history, gradient norms, val accuracy)
3. Computes a **fidelity score** — the R-squared correlation between real and simulated curves
4. Outputs a comparison plot (PNG) showing both curves overlaid
5. Asserts fidelity score > 0.85 (curves are statistically similar)

### Why This Matters

The validation suite answers the question judges will inevitably ask: *"How do you know your simulated curves are realistic?"* The answer is not a claim — it is a set of runnable scripts that produce visual proof.

The `model.eval()` validation script is particularly compelling — it contains exactly the two-line PyTorch bug that every ML engineer has encountered:

```python
model.eval()  # Called during checkpoint loading
# ... model.train() never called before training resumes
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = criterion(model(batch), labels)  # BatchNorm uses stale running stats
        loss.backward()
```

This is not a synthetic exercise — it is a real PyTorch pattern that causes real production failures.

### Integration with Docker

The validation suite is run locally during development — not during Docker build (to avoid 5–15 minute build times from real training loops). The fidelity reports (plots + scores) are pre-computed, committed to the repo under `validation/reports/`, and copied into the Docker image as static files. They are served via `GET /validation-report` — giving judges one-click access to the proof without requiring them to install PyTorch locally or wait for a long build.

---

## 19. Live Diagnostic Dashboard

The environment includes a browser-based diagnostic dashboard served from `GET /dashboard`. The dashboard is a presentation-layer bonus — it adds zero dependencies to the Docker image (CDN-loaded JS), zero overhead to the core environment logic, and is not required for OpenEnv spec compliance. However, it transforms the judging experience by making agent behavior immediately legible to judges, users, and developers evaluating the environment.

### What It Shows

The dashboard connects to the same WebSocket endpoint (`/ws`) that agents use. It renders four synchronized panels updated in real time as the agent takes actions:

**Panel 1 — Training Metrics (Plotly.js line charts)**
- Training loss history (line, updated each step)
- Validation loss history (line, overlaid)
- Validation accuracy history (line, separate y-axis)
- Vertical markers at the step where `restart_run` was called, showing the before/after trajectory shift

**Panel 2 — Gradient & Weight Heatmap**
- Per-layer gradient norms rendered as a color-coded heatmap (green = normal, yellow = elevated, red = exploding, blue = vanishing)
- Per-layer weight norms shown when `inspect_model_weights` is called
- Empty until inspection actions are called — panels show "Not yet inspected" placeholders

**Panel 3 — Action Timeline**
- Horizontal timeline showing each action the agent has taken, color-coded by type (blue = investigation, green = fix, orange = code fix, red = wrong action, gold = diagnosis)
- Reward earned at each step shown as a bar chart below the timeline
- Cumulative reward line overlaid

**Panel 4 — Episode Summary**
- Current task ID, step count, episode state flags
- Available actions (updated dynamically)
- Code snippet display (when `inspect_code` has been called)
- Final grader score displayed prominently when episode completes

### Technical Implementation

The dashboard is a single HTML file with embedded JavaScript (~400 lines) served as a static response from the `/dashboard` FastAPI route. It uses:
- **Plotly.js** (CDN-loaded) for charts — no npm build step required
- **Native WebSocket API** for real-time data
- **CSS Grid** for responsive 2×2 panel layout

The dashboard adds zero dependencies to the Docker image (CDN-loaded JS), zero complexity to the server (one static HTML route), and zero overhead to the environment logic (it reads the same WebSocket messages agents produce). It is purely a visualization layer.

### Why This Matters for Judging

A judge evaluating the environment can open the dashboard in a browser, trigger a baseline run, and *watch the agent investigate in real time* — seeing the loss curves populate, the gradient heatmap light up when `inspect_gradients` is called, the action timeline grow, and the reward accumulate. This transforms a "WebSocket API that returns JSON" into a visible, comprehensible experience. The BatchNorm eval mode task is particularly striking: the judge can see the agent inspect gradients (heatmap shows green/normal), then either correctly pivot to `inspect_model_modes` or incorrectly chase the gradient red herring — the dashboard makes the quality of the agent's reasoning visible at a glance.

---

## 20. Project File Structure

```
ML Debugger/                        # Project root
│
├── Dockerfile                      # Multi-stage: validation → runtime (both include PyTorch)
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
├── README.md
├── baseline_heuristic.py          # Rule-based baseline (default, no API key)
├── baseline_inference.py           # LLM baseline (optional, requires OpenAI key)
│
├── ml_training_debugger/
│   ├── __init__.py
│   ├── models.py                   # Pydantic models, enums (RootCauseDiagnosis, CodeSnippet, ModelWeightStats)
│   ├── client.py                   # EnvClient extension with typed action/observation
│   ├── scenarios.py                # ScenarioParams, sample_scenario(), code bug templates
│   ├── pytorch_engine.py           # Real torch.nn.Module models, fault injection, gradient/weight extraction
│   ├── simulation.py               # Parametric curve generation (loss/accuracy histories) using torch.Tensor ops
│   ├── code_templates.py           # PyTorch code snippets with injected bugs for Task 6
│   ├── reward_engine.py            # compute_reward() — all reward components
│   └── graders.py                  # Per-task grader functions (0.0–1.0)
│
├── server/
│   ├── environment.py              # MLTrainingEnvironment extending Environment base class
│   ├── app.py                      # FastAPI app with HTTP routes + framework WebSocket
│   ├── dashboard.html              # Live diagnostic dashboard (single-file SPA)
│   └── requirements.txt
│
├── validation/                     # PyTorch validation suite — proves simulation fidelity
│   ├── requirements.txt            # torch (CPU-only), matplotlib, scipy
│   ├── conftest.py                 # Shared fixtures (CIFAR-10 subset, model definitions)
│   ├── validate_exploding_gradients.py
│   ├── validate_vanishing_gradients.py
│   ├── validate_data_leakage.py
│   ├── validate_overfitting.py
│   ├── validate_batchnorm_eval.py
│   ├── validate_code_bugs.py
│   └── reports/                    # Generated: fidelity scores + comparison plots (PNGs)
│
└── tests/
    ├── test_models.py
    ├── test_scenarios.py
    ├── test_pytorch_engine.py       # Tests for real model instantiation and fault injection
    ├── test_simulation.py
    ├── test_code_templates.py       # Tests for code bug generation and fix validation
    ├── test_reward_engine.py
    ├── test_graders.py
    └── test_episode_lifecycle.py
```

### Module Responsibilities

**`models.py`** — All Pydantic models (`TrainingConfig`, `GradientStats`, `DataBatchStats`, `ModelWeightStats`, `CodeSnippet`, `EpisodeState`, `MLTrainingObservation`, `MLTrainingAction`) and the `RootCauseDiagnosis` enum. No business logic.

**`client.py`** — Extends `EnvClient` with `action_type = MLTrainingAction` and `observation_type = MLTrainingObservation`.

**`scenarios.py`** — The `ScenarioParams` dataclass and `sample_scenario(task_id, rng)` function. Takes a task ID and returns a `ScenarioParams` with randomized fault parameters.

**`pytorch_engine.py`** — The real PyTorch integration layer. Contains `SimpleCNN` model definition, `inject_fault(model, scenario)` function, and extraction functions `extract_gradient_stats(model)` and `extract_weight_stats(model)` that read real `param.grad` and `state_dict()` values. All `torch.nn.Module` and `torch.autograd` usage is isolated here.

**`simulation.py`** — Parametric curve generators using `torch.Tensor` operations: `gen_loss_history`, `gen_val_accuracy_history`. These produce the 20-epoch loss/accuracy histories that are too expensive to compute via real training.

**`code_templates.py`** — PyTorch code snippet templates for Task 6. Each template is a real, syntactically valid Python/PyTorch training script with one injected bug. Contains the `generate_code_snippet(bug_type, rng)` function and the `validate_fix(bug_type, line, replacement)` grading function. Fix validation uses a multi-strategy pipeline: (1) whitespace normalization + comment stripping, (2) token-stream comparison via Python's `tokenize` module, (3) 2–3 semantic equivalence patterns per bug variant, (4) `ast.parse()` fallback to verify the buggy pattern is absent from the resulting AST. This handles the formatting variations that LLM agents produce.

**`reward_engine.py`** — A single function `compute_reward(action, state, scenario)` implementing the formula exactly as specified.

**`graders.py`** — Six grader functions, one per task. Each takes a completed `EpisodeState` and `ScenarioParams` and returns a float in [0.0, 1.0]. Entirely independent of `reward_engine.py`.

**`server/environment.py`** — The `MLTrainingEnvironment` class extending `Environment`. Implements `reset()` and `step()` only. `reset()` calls `sample_scenario()`, instantiates a real PyTorch model via `pytorch_engine.py`, injects the fault, runs forward+backward passes, freezes the state, generates parametric curves, and returns the initial observation. `step()` dispatches the action, calls `compute_reward()`, updates episode state, and returns the result tuple.

**`server/app.py`** — Calls `create_app(MLTrainingEnvironment, MLTrainingAction, MLTrainingObservation)` from `openenv.core.env_server.http_server` to get the FastAPI instance, then adds custom HTTP routes (`/tasks`, `/grader`, `/baseline`, overrides `/health`) directly on the returned app. Framework handles `/reset`, `/step`, `/state`, `/ws`, `/schema` automatically.

---

## 21. Extensibility

The architecture supports straightforward extension: adding a new task requires a new `ScenarioParams` variant, a new fault injection case in `pytorch_engine.py`, a new parametric generator in `simulation.py`, a new grader in `graders.py`, and a new entry in `openenv.yaml`. No changes to the framework integration, reward engine formula, or server code are needed. Planned post-submission tasks include Dead ReLU, Wrong Loss Function, Label Noise, and Gradient Accumulation Mismatch.

---

## 22. Known Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **OpenEnv WebSocket + HTTP composition** — ~~`openenv-core`'s WebSocket server may not compose cleanly with custom FastAPI HTTP routes~~ | ~~High~~ **RESOLVED** | **Verified.** `create_app()` returns a standard FastAPI instance. Custom routes (`@app.get('/tasks')`, `@app.post('/grader')`, etc.) register cleanly alongside framework-provided routes (`/reset`, `/step`, `/state`, `/ws`, `/schema`, `/health`). Tested with `openenv-core` v0.2.2. No fallback needed. |
| **Docker image size with PyTorch CPU** — total image may exceed 500MB, risking HF Spaces build/deploy issues | Medium — deployment failure | Use `python:3.12-slim` + `torch` CPU-only wheel (~150MB). Multi-stage build to exclude build-only deps. Target: <500MB. Test HF Spaces deployment early. |
| **Validation suite build time** — running 6 real PyTorch training scripts (20 epochs each) during Docker build could take 5–15 minutes, risking HF Spaces build timeout | Medium — build timeout | Pre-compute fidelity reports locally and include as static files in the repo. Docker build stage runs `pytest` on cached reports, not full training. Validation suite runs manually during development, not on every build. |
| **LLM baseline non-determinism** — OpenAI's `seed` parameter is best-effort, not guaranteed | Low — rule-based baseline is the submission default | LLM baseline is supplementary. Rule-based baseline is the only one used for Phase 1 auto-validation reproducibility checks. |
| **Task 6 code fix validation** — validating arbitrary code fixes is inherently fragile | Medium — grader false negatives | Use a **multi-strategy validation pipeline**: (1) normalize whitespace and strip inline comments before comparison, (2) tokenize using Python's `tokenize` module and compare token streams (ignoring COMMENT, NL, NEWLINE, INDENT, DEDENT tokens), (3) for each bug variant, define 2–3 **semantic equivalence patterns** that capture valid alternative fixes (e.g., both `loss = criterion(output, batch_y)` and `loss = criterion(model(batch_x), batch_y)` are valid for the `detach_loss` variant), (4) as a final fallback, attempt `ast.parse()` on the full code with the replacement applied and verify the buggy pattern is absent from the AST. This layered approach handles trailing spaces, comments, minor reformatting, and semantically equivalent rewrites that LLM agents will produce during Phase 2 evaluation. |
| **Timeline (March 28 – April 8 = ~11 days)** — ambitious scope for a 1–3 person team | High — missed deadline | MVP (Tasks 1, 3, 5) will be completed and deployed first. Tasks 2, 4, 6, dashboard, validation suite, and LLM baseline are stretch goals added only after MVP is live on HF Spaces. See [Section 4.1](#41-scope-prioritization) for priority order. |

---

## 23. Design Decision Rationale

### Why a hybrid approach — real PyTorch models + parametric curves

Real training runs are too slow for the auto-validator (10–40 seconds per episode vs. <200ms for model instantiation + 2 passes + parametric curves), too fragile for reproducibility across platforms, and too exploitable when static. The hybrid approach gives us real `torch.autograd` gradients and real `state_dict()` weight snapshots (authenticity) combined with sub-millisecond loss curve generation (speed). The PyTorch validation suite (Section 18) proves the parametric curves match real training behavior with R² > 0.85.

### Why `torch.Tensor` operations instead of numpy

Every computation in the core environment uses `torch.Tensor` rather than `numpy.ndarray`. This means `import torch` appears in every core module — critical for a Meta PyTorch hackathon submission. The performance difference is negligible for the small tensor sizes involved, but the signal to judges is clear: this is a PyTorch-native environment.

### Why a code-level debugging task (Task 6)

The first 5 tasks test metric interpretation and model inspection — valuable but limited. Task 6 tests whether an agent can read PyTorch code, identify a bug, and produce a valid fix. This directly addresses Meta's core interest: AI agents that understand PyTorch code. It also makes the environment's action space fundamentally richer — the agent must switch from "dashboard debugging" to "code review" mode, requiring a different type of reasoning.

### Why closed enum instead of free-text diagnosis

Free-text grading requires either human judgment or fuzzy string matching. A closed enum makes the grader a single equality check. This also reflects how real incident response systems work — a production ML monitoring tool has a fixed taxonomy of failure types, not an open-ended text field.

### Why scenarios are randomized per reset() call

A static scenario set can be memorized. With randomized parameters, the specific numerical values require active investigation to determine. This is what makes the environment measure diagnostic reasoning rather than pattern lookup.

### Why the grader and reward function are separate modules

The reward function is a per-action training signal. The grader is a holistic episode evaluation for scoring purposes. They measure different things. A grader that is a sum of step rewards is trivially gameable and does not produce the holistic quality score that benchmarking requires.

### Why the context-gated penalty is the most important reward component

It is the only component that requires knowledge of the agent's full information state at the time of action. Every other reward component is stateless (did this action happen, or not). The context-gated penalty encodes the concept of "evidence-based decision making" — it distinguishes between taking an action before any evidence has been gathered versus taking the same action after gathering evidence that contradicts it.

### Why two baselines instead of one

The rule-based baseline is the default because it requires no API key and is deterministic by construction. The LLM baseline (GPT-4o) demonstrates the environment's value: the score gap between heuristic and reasoning agents (to be measured empirically and reported in the README) shows the environment measurably rewards reasoning over heuristics. The gap is expected to be most dramatic on Task 6 (code bug) where the heuristic agent cannot parse code.

### Why a live diagnostic dashboard

A WebSocket API that returns JSON is technically correct but visually invisible. The dashboard transforms the environment from an abstract API into a visible, comprehensible experience — judges can watch an agent investigate, see curves populate, observe the gradient heatmap change, and understand the quality of the agent's reasoning at a glance. This is the difference between a technically strong submission and a winning submission.

---

## 24. Submission Readiness Checklist

### Official Pre-Submission Gate (all must pass or disqualified)

- [ ] **HF Space deploys** — automated ping to the Space URL returns 200 and responds to `reset()`
- [ ] **OpenEnv spec compliance** — `openenv.yaml` present, typed Pydantic models, `step()`/`reset()`/`state()` endpoints functional
- [ ] **Dockerfile builds** — `docker build` succeeds on the submitted repo, `docker run` starts the server
- [ ] **Baseline reproduces** — run `baseline_heuristic.py` twice, verify identical scores on both runs
- [ ] **3+ tasks with graders** — enumerate tasks via `GET /tasks`, run each grader, verify all scores in 0.0–1.0 range

### Additional Required Endpoints

- [ ] **`POST /baseline`** — triggers inference script and returns baseline scores for all tasks
- [ ] **`POST /grader`** — returns grader score after an episode is completed
- [ ] **`GET /tasks`** — returns list of tasks and the action schema

### Project-Specific Verification

- [ ] **Public GitHub repo** — contains code, README, requirements, demo script
- [ ] **README complete** — environment description, action/observation space definitions, task descriptions with difficulty, setup instructions, baseline scores
- [ ] **`openenv.yaml` complete** — name, version, description, framework, tags, observation_space, action_space, tasks (with IDs, difficulties, max_steps), reward, endpoints
- [ ] **Rule-based baseline runs offline** — no API key required, deterministic output
- [ ] **All 6 graders return valid scores** — 0.0–1.0 range, deterministic for same episode state
- [ ] **Context-gated penalty fires correctly** — manual test: `inspect_gradients` (normal) → `add_callback` → verify −0.20 penalty
- [ ] **Context-gated penalty does NOT fire prematurely** — manual test: `add_callback` without prior inspection → verify no penalty
- [ ] **Task 6 code fix validation works** — test all 4 bug variants with correct and incorrect fixes
- [ ] **Episode state isolation** — two concurrent WebSocket sessions do not interfere with each other
- [ ] **`GET /health`** returns `{"status": "ready", "tasks": 6}` after server startup

### Pre-Submit Smoke Test Sequence

```bash
# 1. Build and run locally
docker build -t pytorch-debugger .
docker run -p 7860:7860 pytorch-debugger

# 2. Health check
curl http://localhost:7860/health

# 3. Task list
curl http://localhost:7860/tasks

# 4. Run baseline twice, compare scores
python baseline_heuristic.py > run1.json
python baseline_heuristic.py > run2.json
diff run1.json run2.json  # Must be identical

# 5. Run grader after baseline
curl -X POST http://localhost:7860/grader

# 6. Trigger full baseline via endpoint
curl -X POST http://localhost:7860/baseline
```

---

*PyTorch Training Run Debugger — OpenEnv Environment Specification*
*Meta PyTorch OpenEnv Hackathon x Scaler School of Technology, Round 1*
*Submission window opens: March 28, 2026. Round 1 deadline: April 8, 2026.*
