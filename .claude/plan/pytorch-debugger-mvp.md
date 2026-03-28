# Implementation Plan: PyTorch Training Run Debugger — OpenEnv Environment

**Generated:** 2026-03-28
**King File:** `ml-training-debugger-spec.md` — single source of truth for all conflicts
**Runtime:** Python 3.12 · PyTorch CPU-only · openenv-core (installed in .venv)
**MVP Scope:** Tasks 1, 3, 5 + rule-based baseline + all required endpoints + Docker + HF Spaces

---

## Markdown Files Confirmed Read

| File | Lines | Role |
|------|-------|------|
| `ml-training-debugger-spec.md` | 1549 | **KING FILE** — final authority on all design decisions |
| `CLAUDE.md` | ~280 | Coding standards, non-negotiable rules, reward constants |
| `PRD.md` | ~368 | Product requirements, success metrics, timeline |
| `ROADMAP.md` | ~442 | Phased roadmap with acceptance criteria |

All four files read in full. The spec is the definitive authority.

---

## Complete Project Structure (Final State)

```
ML Debugger/                           # Project root
├── .claude/
│   └── plan/
│       └── pytorch-debugger-mvp.md    # This plan
├── .dockerignore
├── .gitignore
├── .python-version                    # "3.12"
├── CLAUDE.md                          # Already exists
├── Dockerfile
├── PRD.md                             # Already exists
├── README.md
├── ROADMAP.md                         # Already exists
├── baseline_heuristic.py              # Rule-based baseline (no API key)
├── baseline_inference.py              # LLM baseline (optional, requires OPENAI_API_KEY)
├── deploy.sh                          # One-command build+test+validate script
├── ml-training-debugger-spec.md       # Already exists (king file)
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
│
├── ml_training_debugger/
│   ├── __init__.py
│   ├── models.py                      # All Pydantic models + RootCauseDiagnosis enum
│   ├── client.py                      # EnvClient extension with typed action/observation
│   ├── scenarios.py                   # ScenarioParams + sample_scenario()
│   ├── pytorch_engine.py              # SimpleCNN, fault injection, gradient/weight extraction
│   ├── simulation.py                  # Parametric curve generation (torch.Tensor ops)
│   ├── code_templates.py              # Task 6: code snippets with bugs + validate_fix()
│   ├── reward_engine.py               # compute_reward() — all 7 components
│   └── graders.py                     # Per-task grader functions (0.0–1.0)
│
├── server/
│   ├── __init__.py
│   ├── environment.py                 # MLTrainingEnvironment(Environment)
│   ├── app.py                         # create_app() + custom routes
│   └── dashboard.html                 # Live diagnostic dashboard (Phase 3)
│
├── validation/                        # PyTorch validation suite (Phase 3)
│   ├── requirements.txt
│   ├── conftest.py
│   ├── validate_exploding_gradients.py
│   ├── validate_vanishing_gradients.py
│   ├── validate_data_leakage.py
│   ├── validate_overfitting.py
│   ├── validate_batchnorm_eval.py
│   ├── validate_code_bugs.py
│   └── reports/                       # Pre-computed fidelity plots
│
└── tests/
    ├── __init__.py
    ├── conftest.py                    # Shared fixtures
    ├── test_models.py
    ├── test_scenarios.py
    ├── test_pytorch_engine.py
    ├── test_simulation.py
    ├── test_code_templates.py
    ├── test_reward_engine.py
    ├── test_graders.py
    ├── test_episode_lifecycle.py
    ├── test_endpoints.py
    └── test_baseline_reproducibility.py
```

---

## Phase 0: Project Initialization & Validation Setup

### Goal
A running skeleton server that proves the toolchain works end-to-end. Zero business logic — just plumbing.

### Files to Create

**Step 0.1 — Project config files:**

1. **`.python-version`** — content: `3.12`

2. **`.gitignore`**:
```
.venv/
__pycache__/
*.pyc
*.pyo
.env
run*.json
.pytest_cache/
htmlcov/
*.egg-info/
dist/
build/
validation/reports/*.png
.mypy_cache/
```

3. **`.dockerignore`**:
```
.venv/
__pycache__/
.git/
.pytest_cache/
tests/
validation/
*.md
!README.md
.claude/
run*.json
htmlcov/
```

4. **`pyproject.toml`**:
```toml
[project]
name = "pytorch-training-debugger"
version = "1.0.0"
description = "OpenEnv RL environment for PyTorch training failure debugging"
requires-python = ">=3.12"
dependencies = [
    "torch",
    "openenv-core",
    "pydantic>=2.0",
    "fastapi",
    "uvicorn",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "pytest-asyncio",
    "black",
    "ruff",
    "isort",
    "httpx",
    "websockets",
]
llm = [
    "openai",
]

[tool.black]
line-length = 88

[tool.isort]
profile = "black"

[tool.ruff]
line-length = 88
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

5. **`requirements.txt`** (for Docker — flat list, no dev deps):
```
torch
openenv-core
pydantic>=2.0
fastapi
uvicorn
openai
```

**Step 0.2 — Package stubs:**

6. **`ml_training_debugger/__init__.py`**:
```python
"""PyTorch Training Run Debugger — OpenEnv Environment."""

__version__ = "1.0.0"
```

7. **`ml_training_debugger/models.py`** — STUB with all Pydantic models:
```python
"""All Pydantic models, enums, and typed data structures.

No business logic. Pure data definitions.
"""

from __future__ import annotations

import enum
from typing import Literal, Optional

import torch
from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class RootCauseDiagnosis(str, enum.Enum):
    """Closed enumeration of ML failure root causes."""
    LR_TOO_HIGH = "lr_too_high"
    VANISHING_GRADIENTS = "vanishing_gradients"
    DATA_LEAKAGE = "data_leakage"
    OVERFITTING = "overfitting"
    BATCHNORM_EVAL_MODE = "batchnorm_eval_mode"
    CODE_BUG = "code_bug"


class TrainingConfig(BaseModel):
    """Typed hyperparameter configuration."""
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 64
    hidden_dim: int = 64
    num_layers: int = 3
    optimizer: str = "adam"
    dropout_rate: float = 0.0
    gradient_clip_norm: Optional[float] = None


class GradientStats(BaseModel):
    """Per-layer gradient information from real torch.autograd."""
    layer_name: str
    norm_history: list[float]
    mean_norm: float
    max_norm: float
    is_exploding: bool
    is_vanishing: bool


class ModelWeightStats(BaseModel):
    """Per-layer weight statistics from real state_dict()."""
    layer_name: str
    weight_norm: float
    weight_mean: float
    weight_std: float
    weight_min: float
    weight_max: float
    dead_neuron_pct: float = 0.0
    has_nan: bool = False
    has_inf: bool = False


class DataBatchStats(BaseModel):
    """Data batch inspection results."""
    label_distribution: dict[int, float]
    feature_mean: float
    feature_std: float
    null_count: int = 0
    class_overlap_score: float
    batch_size: int
    duplicate_ratio: float = 0.0


class CodeSnippet(BaseModel):
    """PyTorch code for Task 6 inspection."""
    code: str
    filename: str = "train.py"
    line_count: int
    imports: list[str]
    hint: Optional[str] = None


class EpisodeState(BaseModel):
    """Tracks agent history within an episode."""
    step_count: int = 0
    gradients_inspected: bool = False
    gradients_were_normal: bool = False
    data_inspected: bool = False
    model_modes_inspected: bool = False
    model_weights_inspected: bool = False
    code_inspected: bool = False
    fix_action_taken: bool = False
    restart_after_fix: bool = False
    diagnosis_submitted: bool = False
    actions_taken: list[str] = Field(default_factory=list)

    def compute_available_actions(self) -> list[str]:
        """Dynamically compute available actions based on current state."""
        actions = [
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
        ]
        if self.code_inspected:
            actions.append("fix_code")
        if self.fix_action_taken:
            actions.append("restart_run")
        if self.restart_after_fix:
            actions.append("rollback_checkpoint")
        if not self.diagnosis_submitted:
            actions.append("mark_diagnosed")
        return actions


ACTION_TYPES = Literal[
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
    "fix_code",
    "restart_run",
    "mark_diagnosed",
    "rollback_checkpoint",
]


class MLTrainingAction(Action):
    """What the agent can do — extends openenv Action."""
    action_type: str
    target: Optional[str] = None
    value: Optional[float | int | str] = None
    diagnosis: Optional[str] = None
    line: Optional[int] = None
    replacement: Optional[str] = None


class MLTrainingObservation(Observation):
    """Full observation — extends openenv Observation (has done, reward, metadata)."""
    run_id: str = ""
    framework: str = "pytorch"
    epoch: int = 20
    training_loss_history: list[float] = Field(default_factory=list)
    val_loss_history: list[float] = Field(default_factory=list)
    val_accuracy_history: list[float] = Field(default_factory=list)
    gradient_stats: list[GradientStats] = Field(default_factory=list)
    model_weight_stats: Optional[list[ModelWeightStats]] = None
    gpu_memory_used_gb: float = 6.2
    gpu_memory_total_gb: float = 16.0
    learning_rate: float = 0.001
    current_config: TrainingConfig = Field(default_factory=TrainingConfig)
    error_log: Optional[str] = None
    data_batch_stats: Optional[DataBatchStats] = None
    model_mode_info: Optional[dict[str, str]] = None
    code_snippet: Optional[CodeSnippet] = None
    available_actions: list[str] = Field(default_factory=list)
    episode_state: EpisodeState = Field(default_factory=EpisodeState)
    notes: Optional[str] = None
```

8. **`ml_training_debugger/client.py`** — STUB:
```python
"""Typed EnvClient for baseline scripts."""

from openenv.core.env_client import EnvClient

from ml_training_debugger.models import MLTrainingAction, MLTrainingObservation


class MLTrainingEnvClient(EnvClient[MLTrainingAction, MLTrainingObservation, dict]):
    """Typed client for the PyTorch Training Debugger environment."""

    def _step_payload(self, action: MLTrainingAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_observation(self, data: dict) -> MLTrainingObservation:
        return MLTrainingObservation.model_validate(data)
```

9. **`server/__init__.py`** — empty file

10. **`server/environment.py`** — STUB:
```python
"""MLTrainingEnvironment — extends openenv Environment."""

from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from ml_training_debugger.models import (
    EpisodeState,
    MLTrainingAction,
    MLTrainingObservation,
    TrainingConfig,
)


class MLTrainingEnvironment(
    Environment[MLTrainingAction, MLTrainingObservation, dict]
):
    """OpenEnv environment for PyTorch training run debugging."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MLTrainingObservation:
        """Reset environment, return initial observation."""
        state = EpisodeState()
        obs = MLTrainingObservation(
            run_id=episode_id or "episode_001",
            training_loss_history=[2.3] * 20,
            val_loss_history=[2.3] * 20,
            val_accuracy_history=[0.1] * 20,
            current_config=TrainingConfig(),
            available_actions=state.compute_available_actions(),
            episode_state=state,
            done=False,
            reward=0.0,
        )
        return obs

    def step(
        self,
        action: MLTrainingAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MLTrainingObservation:
        """Process one agent action."""
        state = EpisodeState()
        obs = MLTrainingObservation(
            run_id="episode_001",
            training_loss_history=[2.3] * 20,
            val_loss_history=[2.3] * 20,
            val_accuracy_history=[0.1] * 20,
            current_config=TrainingConfig(),
            available_actions=state.compute_available_actions(),
            episode_state=state,
            done=False,
            reward=-0.01,
        )
        return obs

    @property
    def state(self) -> dict:
        """Return current environment state."""
        return {"status": "active"}
```

11. **`server/app.py`** — STUB with all endpoints:
```python
"""FastAPI app — openenv create_app() + custom routes."""

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core.env_server.http_server import create_app

from ml_training_debugger.models import MLTrainingAction, MLTrainingObservation
from server.environment import MLTrainingEnvironment

logger = logging.getLogger(__name__)

# create_app takes the class (factory), not an instance
app: FastAPI = create_app(
    MLTrainingEnvironment,
    MLTrainingAction,
    MLTrainingObservation,
    env_name="pytorch_training_debugger",
    max_concurrent_envs=5,
)


@app.get("/health")
def health_check() -> dict:
    """Health check — required by hackathon auto-validator."""
    return {"status": "ready", "tasks": 3}


@app.get("/tasks")
def get_tasks() -> list[dict]:
    """Return task list with IDs, difficulties, and action schema."""
    schema = MLTrainingAction.model_json_schema()
    return [
        {"id": "task_001", "difficulty": "easy", "max_steps": 20, "action_schema": schema},
        {"id": "task_003", "difficulty": "medium", "max_steps": 25, "action_schema": schema},
        {"id": "task_005", "difficulty": "hard", "max_steps": 30, "action_schema": schema},
    ]


@app.post("/grader")
def post_grader() -> dict:
    """Return grader score for most recently completed episode."""
    return {"score": None, "error": "no_completed_episode"}


@app.post("/baseline")
async def post_baseline() -> dict:
    """Trigger baseline run, return scores."""
    return {"scores": {"task_001": 0.0, "task_003": 0.0, "task_005": 0.0}}
```

12. **`openenv.yaml`**:
```yaml
spec_version: 1
name: pytorch-training-debugger
type: space
runtime: fastapi
app: server.app:app
port: 7860

# Extended metadata
version: "1.0.0"
description: |
  PyTorch-native fault injection engine for training failure debugging.
  An AI agent investigates, diagnoses, fixes, and verifies broken
  training runs using real torch.nn.Module models, torch.autograd
  gradients, state_dict() weight inspection, and PyTorch code-level
  debugging.
framework: openenv
tags: [ml-debugging, pytorch, reinforcement-learning, root-cause-analysis, fault-injection]

observation_space:
  type: MLTrainingObservation
  description: "Training run snapshot with progressive reveal"

action_space:
  type: MLTrainingAction
  description: "Investigation, fix, and diagnosis actions with dynamic availability"

tasks:
  - id: task_001
    difficulty: easy
    max_steps: 20
  - id: task_003
    difficulty: medium
    max_steps: 25
  - id: task_005
    difficulty: hard
    max_steps: 30

reward:
  range: [-1.0, 1.0]
  shaped: true
  step_penalty: -0.01
  investigation_bonus: 0.05
  correct_diagnosis: 0.50
  terminal_convergence: 0.40

endpoints:
  websocket: "/ws"
  tasks: "GET /tasks"
  grader: "POST /grader"
  baseline: "POST /baseline"
  health: "GET /health"
```

13. **`Dockerfile`**:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install PyTorch CPU-only first (largest layer, cached)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ml_training_debugger/ ml_training_debugger/
COPY server/ server/
COPY openenv.yaml .
COPY baseline_heuristic.py .

# Copy pre-computed validation reports if they exist
COPY validation/reports/ validation/reports/ 2>/dev/null || true

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

14. **`tests/__init__.py`** — empty file

15. **`tests/conftest.py`**:
```python
"""Shared test fixtures."""

import pytest

from ml_training_debugger.models import (
    EpisodeState,
    MLTrainingAction,
    MLTrainingObservation,
    TrainingConfig,
)


@pytest.fixture
def fresh_episode_state() -> EpisodeState:
    return EpisodeState()


@pytest.fixture
def sample_config() -> TrainingConfig:
    return TrainingConfig(learning_rate=0.001)


@pytest.fixture
def sample_observation() -> MLTrainingObservation:
    state = EpisodeState()
    return MLTrainingObservation(
        run_id="test_episode",
        training_loss_history=[2.3 - i * 0.1 for i in range(20)],
        val_loss_history=[2.3 - i * 0.08 for i in range(20)],
        val_accuracy_history=[0.1 + i * 0.04 for i in range(20)],
        current_config=TrainingConfig(),
        available_actions=state.compute_available_actions(),
        episode_state=state,
        done=False,
        reward=0.0,
    )
```

16. **`tests/test_models.py`**:
```python
"""Test all Pydantic models instantiate and serialize correctly."""

import json
import pytest
from ml_training_debugger.models import (
    CodeSnippet,
    DataBatchStats,
    EpisodeState,
    GradientStats,
    MLTrainingAction,
    MLTrainingObservation,
    ModelWeightStats,
    RootCauseDiagnosis,
    TrainingConfig,
)


class TestRootCauseDiagnosis:
    def test_all_six_values_exist(self):
        assert len(RootCauseDiagnosis) == 6

    def test_values_are_strings(self):
        for d in RootCauseDiagnosis:
            assert isinstance(d.value, str)


class TestTrainingConfig:
    def test_default_instantiation(self):
        config = TrainingConfig()
        assert config.learning_rate == 0.001

    def test_json_roundtrip(self):
        config = TrainingConfig(learning_rate=0.01)
        data = json.loads(config.model_dump_json())
        restored = TrainingConfig.model_validate(data)
        assert restored.learning_rate == 0.01


class TestEpisodeState:
    def test_fresh_state(self):
        state = EpisodeState()
        assert state.step_count == 0
        assert not state.gradients_inspected
        assert not state.diagnosis_submitted

    def test_available_actions_initial(self):
        state = EpisodeState()
        actions = state.compute_available_actions()
        assert "inspect_gradients" in actions
        assert "mark_diagnosed" in actions
        assert "fix_code" not in actions
        assert "restart_run" not in actions

    def test_fix_code_available_after_code_inspected(self):
        state = EpisodeState(code_inspected=True)
        actions = state.compute_available_actions()
        assert "fix_code" in actions

    def test_restart_run_available_after_fix(self):
        state = EpisodeState(fix_action_taken=True)
        actions = state.compute_available_actions()
        assert "restart_run" in actions

    def test_mark_diagnosed_disappears_after_submission(self):
        state = EpisodeState(diagnosis_submitted=True)
        actions = state.compute_available_actions()
        assert "mark_diagnosed" not in actions


class TestMLTrainingObservation:
    def test_extends_observation(self):
        from openenv.core.env_server.types import Observation
        assert issubclass(MLTrainingObservation, Observation)

    def test_has_done_and_reward(self):
        obs = MLTrainingObservation(done=True, reward=0.5)
        assert obs.done is True
        assert obs.reward == 0.5

    def test_json_serialization(self):
        obs = MLTrainingObservation(
            run_id="test",
            training_loss_history=[1.0, 2.0],
            val_accuracy_history=[0.5],
        )
        data = json.loads(obs.model_dump_json())
        assert data["run_id"] == "test"


class TestMLTrainingAction:
    def test_extends_action(self):
        from openenv.core.env_server.types import Action
        assert issubclass(MLTrainingAction, Action)

    def test_basic_action(self):
        action = MLTrainingAction(action_type="inspect_gradients")
        assert action.action_type == "inspect_gradients"

    def test_modify_config_action(self):
        action = MLTrainingAction(
            action_type="modify_config",
            target="learning_rate",
            value=0.001,
        )
        assert action.target == "learning_rate"

    def test_mark_diagnosed_action(self):
        action = MLTrainingAction(
            action_type="mark_diagnosed",
            diagnosis="lr_too_high",
        )
        assert action.diagnosis == "lr_too_high"

    def test_fix_code_action(self):
        action = MLTrainingAction(
            action_type="fix_code",
            line=13,
            replacement="loss = criterion(output, batch_y)",
        )
        assert action.line == 13
```

**Step 0.3 — Validation Commands:**

```bash
# In project root with venv activated
source .venv/bin/activate

# 1. Verify imports
python -c "from ml_training_debugger.models import MLTrainingAction, MLTrainingObservation; print('models OK')"
python -c "from ml_training_debugger.client import MLTrainingEnvClient; print('client OK')"
python -c "from server.app import app; print('app OK')"

# 2. Run tests
pytest tests/test_models.py -v

# 3. Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860 &
sleep 3
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl http://localhost:7860/docs
kill %1

# 4. Formatting
black ml_training_debugger/ server/ tests/ --check
ruff check ml_training_debugger/ server/ tests/
isort ml_training_debugger/ server/ tests/ --check --profile black
```

### Acceptance Criteria — Phase 0

- [ ] All Pydantic models instantiate without error and serialize to valid JSON
- [ ] `MLTrainingObservation` extends `Observation` (has `done`, `reward`, `metadata`)
- [ ] `MLTrainingAction` extends `Action` (has `metadata`)
- [ ] `EpisodeState.compute_available_actions()` returns correct dynamic action lists
- [ ] Server starts on port 7860 and responds to `/health` with `{"status": "ready", "tasks": 3}`
- [ ] `/tasks` returns 3 tasks with action schema
- [ ] `pytest tests/test_models.py` passes all tests
- [ ] `client.py` imports without error
- [ ] `black --check`, `ruff check`, `isort --check` all pass

---

## Phase 1: Core Data Models & Pydantic Types

### Goal
Finalize all model fields to match the spec exactly. No business logic yet — just data shapes.

### Files to Edit

**`ml_training_debugger/models.py`** — Already created in Phase 0. Verify:
- All fields match spec Section 10 exactly
- `GradientStats.is_exploding` threshold: `mean_norm > 10.0`
- `GradientStats.is_vanishing` threshold: `mean_norm < 1e-6`
- `TrainingConfig` field names match `modify_config` target options
- `EpisodeState.compute_available_actions()` logic matches spec Section 10 dynamic rules

### Tests (write BEFORE implementation — TDD)

All tests already written in `tests/test_models.py` from Phase 0. Extend with:

```python
class TestGradientStats:
    def test_exploding_threshold(self):
        stats = GradientStats(
            layer_name="fc", norm_history=[15.0], mean_norm=15.0, max_norm=15.0,
            is_exploding=True, is_vanishing=False,
        )
        assert stats.is_exploding is True

    def test_vanishing_threshold(self):
        stats = GradientStats(
            layer_name="conv1", norm_history=[1e-7], mean_norm=1e-7, max_norm=1e-7,
            is_exploding=False, is_vanishing=True,
        )
        assert stats.is_vanishing is True

    def test_normal_gradients(self):
        stats = GradientStats(
            layer_name="conv1", norm_history=[0.5], mean_norm=0.5, max_norm=0.5,
            is_exploding=False, is_vanishing=False,
        )
        assert not stats.is_exploding
        assert not stats.is_vanishing
```

### Acceptance Criteria — Phase 1

- [ ] Every field in every model matches the spec Section 10 types exactly
- [ ] No `Dict[str, Any]` in any public model (typed Pydantic everywhere)
- [ ] `import torch` appears in `models.py`
- [ ] All model tests pass

---

## Phase 2: PyTorch-Native Fault Injection Engine + Simulation

### Goal
Real PyTorch models with real gradients + parametric curve generators. This is the technical heart.

### Files to Create

**Step 2.1 — `ml_training_debugger/scenarios.py`** (~120 lines):

```python
"""ScenarioParams and scenario sampling."""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from ml_training_debugger.models import RootCauseDiagnosis


@dataclasses.dataclass(frozen=True)
class ScenarioParams:
    """Internal scenario parameters — not exposed to agent."""
    task_id: str
    root_cause: RootCauseDiagnosis
    seed: int
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    leakage_pct: float = 0.0
    depth_multiplier: float = 1.0
    divergence_epoch: int = 5
    red_herring_intensity: float = 1.0
    red_herring_spike_layer: str = "fc"
    bug_type: Optional[str] = None
    notes: Optional[str] = None
    error_log: Optional[str] = None
    gpu_memory_used_gb: float = 6.2
    max_steps: int = 20


def sample_scenario(task_id: str, seed: int) -> ScenarioParams:
    """Sample a ScenarioParams for the given task."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Use torch for random selection
    def choose(options: list) -> any:
        idx = int(torch.randint(0, len(options), (1,), generator=rng).item())
        return options[idx]

    if task_id == "task_001":
        lr = choose([0.05, 0.08, 0.10, 0.15, 0.30])
        return ScenarioParams(
            task_id=task_id,
            root_cause=RootCauseDiagnosis.LR_TOO_HIGH,
            seed=seed,
            learning_rate=lr,
            error_log=f"RuntimeError: Loss is NaN at epoch 12 (lr={lr})",
            max_steps=20,
        )

    elif task_id == "task_003":
        leakage = choose([0.12, 0.18, 0.22, 0.28])
        return ScenarioParams(
            task_id=task_id,
            root_cause=RootCauseDiagnosis.DATA_LEAKAGE,
            seed=seed,
            leakage_pct=leakage,
            notes="Model architecture upgraded from 2-layer to 4-layer CNN at epoch 2. Performance improvement may reflect increased model capacity.",
            max_steps=25,
        )

    elif task_id == "task_005":
        intensity = (
            torch.empty(1).uniform_(0.8, 2.5, generator=rng).item()
        )
        spike_layer = choose(["fc", "conv1"])
        return ScenarioParams(
            task_id=task_id,
            root_cause=RootCauseDiagnosis.BATCHNORM_EVAL_MODE,
            seed=seed,
            red_herring_intensity=intensity,
            red_herring_spike_layer=spike_layer,
            gpu_memory_used_gb=14.56,  # 91% of 16GB
            error_log="Warning: GPU memory pressure detected, consider reducing batch size or enabling gradient checkpointing",
            max_steps=30,
        )

    raise ValueError(f"Unknown task_id: {task_id}")
```

**Step 2.2 — `ml_training_debugger/pytorch_engine.py`** (~250 lines):

Key functions:
- `SimpleCNN(torch.nn.Module)` — 3-layer CNN, ~50K params
- `create_model_and_inject_fault(scenario: ScenarioParams) -> tuple[torch.nn.Module, dict]`
- `extract_gradient_stats(model: torch.nn.Module) -> list[GradientStats]`
- `extract_weight_stats(model: torch.nn.Module) -> list[ModelWeightStats]`
- `extract_model_modes(model: torch.nn.Module) -> dict[str, str]`

Implementation notes:
- `torch.manual_seed(scenario.seed)` at the start of every call
- For Task 1: set lr high, run 2 forward+backward passes → gradients explode
- For Task 3: normal model, no gradient anomaly
- For Task 5: call `model.eval()` before training → BatchNorm frozen
- All gradient stats come from real `param.grad` tensors
- All weight stats come from real `model.state_dict()`

**Step 2.3 — `ml_training_debugger/simulation.py`** (~180 lines):

Key functions:
- `gen_loss_history(scenario: ScenarioParams) -> list[float]` — all torch.Tensor ops
- `gen_val_accuracy_history(scenario: ScenarioParams) -> list[float]`
- `gen_val_loss_history(scenario: ScenarioParams) -> list[float]`

Per-task parametric curves from spec Section 6:
- Task 1: `loss = torch.exp(torch.tensor(lr) * torch.arange(20))`
- Task 3: `val_acc = torch.sigmoid(torch.linspace(-3, 3, 20)) * (1 - leakage_pct)`
- Task 5: Normal loss + elevated variance, slow val_acc degradation

### Tests to Create FIRST (TDD)

**`tests/test_scenarios.py`**:
- `sample_scenario("task_001", seed=42)` returns `root_cause == LR_TOO_HIGH`
- `sample_scenario("task_003", seed=42)` returns `root_cause == DATA_LEAKAGE`
- `sample_scenario("task_005", seed=42)` returns `root_cause == BATCHNORM_EVAL_MODE`
- Different seeds produce different parameters (but same root cause per task)
- Unknown task_id raises ValueError

**`tests/test_pytorch_engine.py`**:
- `SimpleCNN` is a real `torch.nn.Module` with ~50K params
- Task 1 fault injection: `is_exploding=True` on all layers
- Task 5 fault injection: `is_exploding=False` on all layers, `model.training==False`
- `extract_gradient_stats` returns `list[GradientStats]` with real float norms
- `extract_weight_stats` returns `list[ModelWeightStats]` from real state_dict
- `extract_model_modes` returns dict mapping layer names to "train"/"eval"
- **CRITICAL**: `import torch` in pytorch_engine.py, zero `import numpy`

**`tests/test_simulation.py`**:
- All outputs are `list[float]` of length 20
- Task 1 (exploding): loss diverges (last value >> first value)
- Task 3 (leakage): val_acc suspiciously high from early epochs
- Task 5 (batchnorm): slow val_acc degradation (~1-2% per epoch)
- All computation uses torch (no numpy)

### Acceptance Criteria — Phase 2

- [ ] `SimpleCNN` is a real `torch.nn.Module` with ~50K parameters
- [ ] `create_model_and_inject_fault` for Task 1 produces exploding gradients (`is_exploding=True` all layers)
- [ ] `create_model_and_inject_fault` for Task 5 produces `model.training==False` on all layers
- [ ] `extract_gradient_stats` returns real floats from `torch.norm(param.grad)`
- [ ] `extract_weight_stats` returns real floats from `state_dict()`
- [ ] Parametric curves produce 20-element lists with correct shapes per task
- [ ] `import torch` in `pytorch_engine.py` and `simulation.py` — zero `import numpy`
- [ ] `torch.manual_seed(seed)` ensures reproducibility
- [ ] All Phase 2 tests pass

---

## Phase 3: MVP Tasks (1, 3, 5) + Reward Engine + Graders

### Goal
All reward logic and graders implemented. The environment can score episodes.

### Files to Create

**Step 3.1 — `ml_training_debugger/reward_engine.py`** (~100 lines):

```python
def compute_reward(
    action: MLTrainingAction,
    episode_state: EpisodeState,
    scenario: ScenarioParams,
    is_valid_action: bool,
    is_correct_fix: bool | None = None,
    convergence_confirmed: bool = False,
) -> float:
```

All 7 components per spec Section 12:
1. Step penalty: -0.01 (flat, unconditional)
2. Investigation bonus: +0.05 (first-time per type)
3. Context-gated penalty: -0.20 (ONLY when `gradients_inspected AND gradients_were_normal`)
4. Invalid action: -0.05
5. Wrong code fix: -0.10
6. Correct diagnosis: +0.50 / Wrong diagnosis: -0.30
7. Terminal convergence: +0.40 (gated on `fix_action_taken AND restart_after_fix`)

Hard cap at [-1.0, 1.0].

**Step 3.2 — `ml_training_debugger/graders.py`** (~150 lines):

One function per task. Each returns float in [0.0, 1.0]:
- `grade_task_001(state: EpisodeState, scenario: ScenarioParams) -> float`
- `grade_task_003(state: EpisodeState, scenario: ScenarioParams) -> float`
- `grade_task_005(state: EpisodeState, scenario: ScenarioParams) -> float`

Grader scoring per spec Section 11:
- Task 1: inspect_gradients(+0.05), correct LR fix(+0.20), restart+converge(+0.35), correct diagnosis(+0.40) = 1.0
- Task 3: inspect_data(+0.05), patch_data_loader(+0.30), restart+converge(+0.30), correct diagnosis(+0.35) = 1.0
- Task 5: inspect_gradients(+0.05), inspect_model_modes(+0.05), fix_model_mode(+0.25), restart+converge(+0.30), correct diagnosis(+0.40) = 1.05 → capped at 1.0. Penalty: add_callback after normal gradients = -0.20.

**CRITICAL — Grader is NOT a sum of step rewards.** It evaluates EpisodeState holistically.

### Tests to Create FIRST (TDD)

**`tests/test_reward_engine.py`** — THE MOST CRITICAL TEST FILE:

```python
class TestContextGatedPenalty:
    """The project's primary innovation — must be exact."""

    def test_no_penalty_before_inspection(self):
        """add_callback at step 1 (no prior inspection) -> NO penalty."""
        state = EpisodeState()  # gradients_inspected=False
        action = MLTrainingAction(action_type="add_callback")
        reward = compute_reward(action, state, scenario, is_valid_action=True)
        # Should be just step penalty: -0.01
        assert reward == pytest.approx(-0.01)

    def test_penalty_after_normal_gradients(self):
        """inspect_gradients (normal) then add_callback -> -0.20 penalty."""
        state = EpisodeState(gradients_inspected=True, gradients_were_normal=True)
        action = MLTrainingAction(action_type="add_callback")
        reward = compute_reward(action, state, scenario, is_valid_action=True)
        # Step penalty + context-gated penalty: -0.01 + -0.20 = -0.21
        assert reward == pytest.approx(-0.21)

    def test_no_penalty_after_abnormal_gradients(self):
        """inspect_gradients (exploding) then add_callback -> no context penalty."""
        state = EpisodeState(gradients_inspected=True, gradients_were_normal=False)
        action = MLTrainingAction(action_type="add_callback")
        reward = compute_reward(action, state, scenario, is_valid_action=True)
        assert reward == pytest.approx(-0.01)
```

Also test:
- Step penalty is flat -0.01 (NOT multiplied by step_count)
- Investigation bonus +0.05 first-time only
- Investigation bonus NOT awarded on repeat
- Correct diagnosis: +0.50
- Wrong diagnosis: -0.30
- Terminal convergence: +0.40 when all gates met
- Invalid action: -0.05
- Wrong code fix: -0.10
- Reward capped at [-1.0, 1.0]

**`tests/test_graders.py`**:
- Each grader returns float in [0.0, 1.0]
- Perfect Task 1 path scores 1.0
- Wrong diagnosis on Task 1 scores < 0.5
- Task 5: agent that chases red herring scores 0.80-0.85
- Task 5: optimal path scores 1.0
- Grader is deterministic (same state → same score)

### Acceptance Criteria — Phase 3

- [ ] `compute_reward` implements all 7 components exactly per spec Section 12
- [ ] Context-gated penalty fires ONLY when `gradients_inspected=True AND gradients_were_normal=True`
- [ ] Context-gated penalty does NOT fire before `inspect_gradients` has been called
- [ ] Step penalty is flat -0.01 (never multiplied by step_count)
- [ ] All 3 graders return [0.0, 1.0] with meaningful variance
- [ ] Grader != reward function (separate modules, separate logic)
- [ ] All Phase 3 tests pass

---

## Phase 4: Environment Lifecycle, EpisodeState, and Action Handling

### Goal
Full `reset()` and `step()` implementations in `environment.py`. The environment is functionally complete.

### Files to Edit

**`server/environment.py`** — Full implementation:

`reset(task_id)`:
1. Parse `task_id` from `kwargs` (framework passes it via kwargs or episode_id)
2. Derive deterministic seed from task_id
3. Call `sample_scenario(task_id, seed)`
4. Call `torch.manual_seed(scenario.seed)`
5. Call `create_model_and_inject_fault(scenario)` → get real model
6. Generate parametric curves via `simulation.py`
7. Create fresh `EpisodeState`
8. Store `(scenario, model, state)` keyed by session/episode ID
9. Return `MLTrainingObservation` with populated loss/acc histories, config, error_log, available_actions — but empty gradient_stats, null data_batch_stats, null model_mode_info, null code_snippet

`step(action)`:
1. Validate action (see spec Section 16 error handling matrix)
2. Increment `step_count`
3. Dispatch by `action.action_type`:
   - **`inspect_gradients`**: Extract real gradient stats, set `gradients_inspected=True`, compute `gradients_were_normal` (all layers `is_exploding==False`)
   - **`inspect_data_batch`**: Generate data batch stats, set `data_inspected=True`
   - **`inspect_model_modes`**: Extract model modes, set `model_modes_inspected=True`
   - **`inspect_model_weights`**: Extract real weight stats, set `model_weights_inspected=True`
   - **`inspect_code`**: Generate code snippet (if task supports it), set `code_inspected=True`
   - **`modify_config`**: Validate target/value, apply change, set `fix_action_taken=True`
   - **`add_callback`**: Apply callback, set `fix_action_taken=True`
   - **`replace_optimizer`**: Apply, set `fix_action_taken=True`
   - **`patch_data_loader`**: Apply, set `fix_action_taken=True`
   - **`fix_model_mode`**: Apply, set `fix_action_taken=True`
   - **`fix_code`**: Validate fix via `validate_fix()`, set `fix_action_taken=True`
   - **`restart_run`**: Requires `fix_action_taken`, set `restart_after_fix=True`, check convergence
   - **`mark_diagnosed`**: Set `diagnosis_submitted=True`, `done=True`
   - **`rollback_checkpoint`**: Requires `restart_after_fix`
4. Call `compute_reward(action, state, scenario, ...)`
5. Check step limit → set `done=True` if reached
6. Update `available_actions` via `state.compute_available_actions()`
7. Return `MLTrainingObservation` with all updated fields

**Session isolation**:
- Store per-session state in `self._sessions: dict[str, SessionData]`
- Session ID comes from the framework (via `episode_id` or WebSocket session)
- Clean up on episode completion or disconnect

### Error Handling (spec Section 16 — ALL cases):

| Error | Behavior | Reward |
|-------|----------|--------|
| Invalid action_type | Return obs unchanged + error note | -0.05 |
| Action not in available_actions | Return obs unchanged + error note | -0.05 |
| modify_config missing target/value | Return obs unchanged + error note | -0.05 |
| modify_config with unknown target | Return obs unchanged + error note | -0.05 |
| mark_diagnosed missing diagnosis | Return obs unchanged + error note | -0.05 |
| mark_diagnosed with invalid diagnosis | Return obs unchanged + error note | -0.05 |
| fix_code missing line/replacement | Return obs unchanged + error note | -0.05 |
| Action after done=True | Return final obs, no state change | 0.0 |
| Step limit reached | Set done=True, return obs | 0.0 |

**CRITICAL**: `step()` must NEVER raise an unhandled exception.

### Tests to Create FIRST (TDD)

**`tests/test_episode_lifecycle.py`**:
- Full reset→inspect→fix→restart→diagnose flow for Task 1
- Full flow for Task 3
- Full flow for Task 5
- `available_actions` updates correctly at each step
- `done=True` after `mark_diagnosed`
- Step limit triggers `done=True`
- Action after done returns final obs with no state change
- Invalid action returns -0.05 penalty
- `restart_run` not available before `fix_action_taken`
- `fix_code` not available before `code_inspected`
- Session isolation: two episodes don't interfere

### Acceptance Criteria — Phase 4

- [ ] `reset(task_id)` for tasks 001/003/005 returns valid `MLTrainingObservation` with correct initial state
- [ ] `step()` dispatches all 14 action types correctly
- [ ] Task 1: `inspect_gradients` → `is_exploding=True` all layers (real torch.autograd)
- [ ] Task 5: `inspect_gradients` → `is_exploding=False` all layers, `gradients_were_normal=True`
- [ ] Task 3: `inspect_data_batch` → `class_overlap_score > 0.5`
- [ ] Task 5: `inspect_model_modes` → all layers in "eval" mode
- [ ] All error conditions from spec Section 16 handled (never raises)
- [ ] Progressive information reveal works (gradient_stats empty until inspected)
- [ ] All Phase 4 tests pass

---

## Phase 5: Server (FastAPI + openenv-core) + All Required Endpoints

### Goal
Wire the real environment into the server. All hackathon-required endpoints return real data.

### Files to Edit

**`server/app.py`** — Full implementation:

```python
# Store reference to last completed episode for /grader
_last_completed: dict[str, dict] = {}  # session_id -> {score, task_id, steps}
_baseline_running: bool = False

@app.get("/health")
def health_check():
    return {"status": "ready", "tasks": 3}

@app.get("/tasks")
def get_tasks():
    schema = MLTrainingAction.model_json_schema()
    return [
        {"id": "task_001", "difficulty": "easy", "max_steps": 20, "action_schema": schema},
        {"id": "task_003", "difficulty": "medium", "max_steps": 25, "action_schema": schema},
        {"id": "task_005", "difficulty": "hard", "max_steps": 30, "action_schema": schema},
    ]

@app.post("/grader")
def post_grader(session_id: str | None = None):
    # Return score for most recently completed episode
    # Edge cases per spec Section 14

@app.post("/baseline")
async def post_baseline():
    # Run baseline_heuristic logic internally
    # Return {"scores": {"task_001": float, ...}}
    # Return 409 if already running
```

**Grader endpoint edge cases** (spec Section 14):
- No episode completed → `{"score": null, "error": "no_completed_episode"}`
- Episode in progress → `{"score": null, "error": "episode_in_progress"}`
- Episode completed → `{"score": 0.85, "task_id": "task_003", "steps": 6}`
- Always HTTP 200 with JSON body

### Tests to Create FIRST (TDD)

**`tests/test_endpoints.py`**:
- `GET /health` returns `{"status": "ready", "tasks": 3}` with 200
- `GET /tasks` returns 3 tasks with action schema
- `POST /grader` returns `{"score": null, "error": "no_completed_episode"}` initially
- `POST /baseline` returns scores for all tasks
- `POST /baseline` while running returns 409
- Integration: reset→step→grader returns valid score

### Acceptance Criteria — Phase 5

- [ ] `GET /health` returns `{"status": "ready", "tasks": 3}` (200)
- [ ] `GET /tasks` returns 3 tasks with IDs, difficulties, action schema
- [ ] `POST /grader` handles all edge cases per spec Section 14
- [ ] `POST /baseline` runs baseline and returns scores
- [ ] Framework auto-provides: `/reset`, `/step`, `/state`, `/ws`, `/schema`, `/docs`
- [ ] All Phase 5 tests pass

---

## Phase 6: Rule-Based Baseline + Reproducibility Guarantees

### Goal
Deterministic baseline that produces bit-exact identical scores on two runs.

### Files to Create

**`baseline_heuristic.py`** (~150 lines):

Decision tree from spec Section 17:
```
1. reset(task_id)
2. inspect_gradients
3. IF any layer is_exploding → modify_config(lr=0.001) → restart → diagnose lr_too_high
4. IF any layer is_vanishing → modify_config(lr=0.01) → restart → diagnose vanishing_gradients
5. inspect_data_batch
6. IF class_overlap_score > 0.5 → patch_data_loader → restart → diagnose data_leakage
7. IF val_loss diverging → modify_config(weight_decay=0.01) → restart → diagnose overfitting
8. inspect_model_modes → IF any eval → fix_model_mode → restart → diagnose batchnorm_eval_mode
9. inspect_code → attempt fix → restart → diagnose code_bug
10. FALLBACK: diagnose overfitting
```

Uses `MLTrainingEnvClient` or `GenericEnvClient` to connect via WebSocket.

**Reproducibility requirements:**
- `torch.manual_seed(seed)` at every `reset()` with deterministic seed per task
- No floating-point non-determinism in parametric curves
- Heuristic is pure logic with no randomness
- Two runs must produce identical JSON output

### Tests to Create FIRST (TDD)

**`tests/test_baseline_reproducibility.py`**:
- Run baseline twice → `diff run1.json run2.json` is empty
- All scores in [0.0, 1.0]
- Expected approximate scores: task_001 ~0.85, task_003 ~0.70, task_005 ~0.45

### Acceptance Criteria — Phase 6

- [ ] `baseline_heuristic.py` runs all 3 MVP tasks without error
- [ ] Two consecutive runs produce bit-exact identical JSON output
- [ ] No API key required
- [ ] All scores in [0.0, 1.0] with meaningful variance
- [ ] Decision tree follows spec Section 17 exactly

---

## Phase 7: Docker, HF Spaces, Logging, Error Handling & Edge Cases

### Goal
Production-ready container that deploys cleanly.

### Files to Edit

**`Dockerfile`** — Finalize:
- Base: `python:3.12-slim`
- PyTorch CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Target: <500MB
- `EXPOSE 7860`
- `CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]`

**Note on Dockerfile COPY**: Cannot use `COPY ... 2>/dev/null || true` in Dockerfile. Instead, ensure all files exist or use multi-stage approach.

**Logging** — Add to `server/app.py` and `server/environment.py`:
- JSON structured logging to stdout
- Log every `reset()`, `step()`, episode completion, errors

**WebSocket edge cases** (spec Section 16):
- Client disconnects mid-episode → retain state 60s
- Malformed JSON → return error, keep connection
- step() before reset() → return "no_active_episode" error
- reset() during active episode → terminate current, start new

### Acceptance Criteria — Phase 7

- [ ] `docker build -t pytorch-debugger .` succeeds
- [ ] Docker image <500MB
- [ ] `docker run -p 7860:7860 pytorch-debugger` starts and serves in <60s
- [ ] `curl http://localhost:7860/health` returns `{"status": "ready", "tasks": 3}`
- [ ] All WebSocket edge cases handled per spec Section 16
- [ ] Structured JSON logging on all significant events

---

## Phase 8: Full Testing Suite + Pre-Submission Smoke Tests

### Goal
>80% test coverage, all edge cases covered.

### Files to Create/Extend

All test files listed above, plus:
- Fill coverage gaps identified by `pytest --cov`
- Add edge case tests for every error in spec Section 16
- Add test for `step()` after `done=True`
- Add test for step limit termination

### Commands

```bash
pytest tests/ -v --cov=ml_training_debugger --cov=server --cov-report=term-missing
```

### Acceptance Criteria — Phase 8

- [ ] `pytest --cov` shows >80% coverage on all modules
- [ ] Every error condition from spec Section 16 has a test
- [ ] Context-gated penalty tests pass (both paths)
- [ ] Dynamic available_actions tests pass
- [ ] All 3 graders tested with multiple scenarios
- [ ] Zero test failures

---

## Phase 9: Final Polish & Submission Readiness

### Goal
README complete, all endpoints verified, `openenv validate` passes, deploy to HF Spaces.

### Files to Create

**`README.md`** (~200 lines):
- Environment description and motivation
- Action/observation space definitions
- Task descriptions with difficulty
- Setup instructions
- Baseline scores table

**`deploy.sh`**:
```bash
#!/bin/bash
set -euo pipefail

echo "=== Building Docker image ==="
docker build -t pytorch-debugger .

echo "=== Starting container ==="
docker run -d -p 7860:7860 --name smoke-test pytorch-debugger
sleep 10

echo "=== Health check ==="
curl -f http://localhost:7860/health || { echo "FAIL: health"; exit 1; }

echo "=== Tasks endpoint ==="
curl -f http://localhost:7860/tasks | python3 -m json.tool || { echo "FAIL: tasks"; exit 1; }

echo "=== Baseline reproducibility ==="
python3 baseline_heuristic.py > run1.json 2>/dev/null
python3 baseline_heuristic.py > run2.json 2>/dev/null
diff run1.json run2.json && echo "PASS: reproducible" || { echo "FAIL: non-reproducible"; exit 1; }

echo "=== Baseline via endpoint ==="
curl -f -X POST http://localhost:7860/baseline | python3 -m json.tool || { echo "FAIL: baseline endpoint"; exit 1; }

echo "=== Grader via endpoint ==="
curl -f -X POST http://localhost:7860/grader | python3 -m json.tool || { echo "FAIL: grader endpoint"; exit 1; }

echo "=== Tests ==="
pytest tests/ -v --cov=ml_training_debugger --cov-report=term-missing

echo "=== Cleanup ==="
docker stop smoke-test && docker rm smoke-test
rm -f run1.json run2.json

echo "=== ALL CHECKS PASSED ==="
```

### Acceptance Criteria — Phase 9

- [ ] `openenv validate` passes
- [ ] `deploy.sh` runs end-to-end with zero failures
- [ ] README is complete per hackathon requirements
- [ ] Docker image <500MB, starts <60s
- [ ] Baseline bit-exact reproducible
- [ ] 3+ tasks with graders returning [0.0, 1.0] with meaningful variance
- [ ] HF Space deployed, tagged `openenv`, responds to `reset()`
- [ ] All typed Pydantic models — no `Dict[str, Any]`
- [ ] `import torch` in every core module — zero numpy in core
- [ ] Context-gated penalty fires correctly and does not fire prematurely
- [ ] Test suite passes with >80% coverage

---

## Technical Risk Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **WebSocket + HTTP composition** | ~~High~~ RESOLVED | `create_app()` returns standard FastAPI. Custom routes add cleanly. Verified in Phase 0. |
| **Docker image size** | Medium | `python:3.12-slim` + torch CPU-only (~150MB). Target <500MB. Test early in Phase 7. |
| **Task 6 fix validation fragility** | Medium | Multi-strategy pipeline: normalize → tokenize → semantic patterns → AST fallback. Test 5+ whitespace variations. (Post-MVP Phase 2 stretch) |
| **Red-herring penalty gating** | HIGH | `gradients_were_normal` set inside `inspect_gradients` handler when ALL layers have `is_exploding=False`. Threshold: `mean_norm > 10.0`. Test BOTH paths explicitly. |
| **Session isolation** | Medium | `dict[str, SessionData]` keyed by session ID. Framework provides session management. |
| **Baseline reproducibility** | HIGH | `torch.manual_seed(seed)` at every `reset()`. Seed derived deterministically from task_id. Heuristic is pure logic. Test with `diff run1.json run2.json`. |
| **Dockerfile build time** | Low | No real training during build. Validation reports pre-computed locally. |
| **openenv.yaml format** | Medium | Template uses `spec_version: 1`, `type: space`, `runtime: fastapi`, `app: server.app:app`. Extended fields (tasks, reward, etc.) are additive. Test with `openenv validate` early. |
| **Port mismatch** | Low | Spec says 7860 (HF Spaces default). openenv template says 8000. Use 7860 everywhere. |

---

## Exact openenv.yaml (Final)

```yaml
spec_version: 1
name: pytorch-training-debugger
type: space
runtime: fastapi
app: server.app:app
port: 7860

version: "1.0.0"
description: |
  PyTorch-native fault injection engine for training failure debugging.
  An AI agent investigates, diagnoses, fixes, and verifies broken
  training runs using real torch.nn.Module models, torch.autograd
  gradients, state_dict() weight inspection, and PyTorch code-level
  debugging. 3 tasks across 3 difficulty tiers with context-gated
  reward shaping.
framework: openenv
tags: [ml-debugging, pytorch, reinforcement-learning, root-cause-analysis, fault-injection, openenv]

observation_space:
  type: MLTrainingObservation
  description: "Training run snapshot with progressive reveal — gradients, weights, data stats, model modes revealed on inspection"

action_space:
  type: MLTrainingAction
  description: "Investigation, fix, and diagnosis actions with dynamic availability"

tasks:
  - id: task_001
    difficulty: easy
    max_steps: 20
  - id: task_003
    difficulty: medium
    max_steps: 25
  - id: task_005
    difficulty: hard
    max_steps: 30

reward:
  range: [-1.0, 1.0]
  shaped: true
  step_penalty: -0.01
  investigation_bonus: 0.05
  max_investigation_bonus: 0.25
  correct_diagnosis: 0.50
  terminal_convergence: 0.40

endpoints:
  websocket: "/ws"
  tasks: "GET /tasks"
  grader: "POST /grader"
  baseline: "POST /baseline"
  health: "GET /health"
```

---

## Exact Dockerfile (Final)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install PyTorch CPU-only first (largest layer, cached)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ml_training_debugger/ ml_training_debugger/
COPY server/ server/
COPY openenv.yaml .
COPY baseline_heuristic.py .
COPY README.md .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## Pre-Submission Smoke Test Sequence

```bash
# 1. Clean build
docker build --no-cache -t pytorch-debugger .

# 2. Start container
docker run -d -p 7860:7860 --name smoke-test pytorch-debugger
sleep 10

# 3. Health check
curl -f http://localhost:7860/health

# 4. Tasks endpoint
curl -f http://localhost:7860/tasks | python3 -m json.tool

# 5. Baseline reproducibility
python3 baseline_heuristic.py > run1.json 2>/dev/null
python3 baseline_heuristic.py > run2.json 2>/dev/null
diff run1.json run2.json && echo "PASS: reproducible" || echo "FAIL"

# 6. Baseline via endpoint
curl -f -X POST http://localhost:7860/baseline | python3 -m json.tool

# 7. Grader via endpoint
curl -f -X POST http://localhost:7860/grader | python3 -m json.tool

# 8. OpenEnv validation
openenv validate

# 9. Test suite
pytest tests/ -v --cov=ml_training_debugger --cov-report=term-missing

# 10. Cleanup
docker stop smoke-test && docker rm smoke-test
rm -f run1.json run2.json

echo "=== All checks passed ==="
```

---

## Post-MVP Stretch (Phase 2 from ROADMAP)

**Only after MVP is 100% deployed and passing all auto-validation:**

1. **Task 6** (code debugging) — highest impact differentiator
   - Create `ml_training_debugger/code_templates.py`
   - 4 bug variants: eval_mode, detach_loss, zero_grad_missing, inplace_relu
   - Multi-strategy fix validation: normalize → tokenize → semantic → AST
   - Diagnosis is ALWAYS `code_bug` regardless of variant

2. **Tasks 2 & 4** — fill out to 6 tasks
   - Task 2: vanishing gradients (easy, mirror of Task 1)
   - Task 4: overfitting (medium, train-val divergence)

3. **Dashboard** — `server/dashboard.html`, Plotly.js via CDN

4. **Validation Suite** — `validation/*.py`, R² > 0.85

5. **LLM Baseline** — `baseline_inference.py`, GPT-4o

Update `openenv.yaml`, `/tasks`, `/health` task count as tasks are added.

---

## SESSION_ID

- CODEX_SESSION: N/A (codeagent-wrapper not available)
- GEMINI_SESSION: N/A (codeagent-wrapper not available)

Plan generated by Claude Opus 4.6 via deep analysis of all 4 project markdown files + openenv-core framework API inspection.
