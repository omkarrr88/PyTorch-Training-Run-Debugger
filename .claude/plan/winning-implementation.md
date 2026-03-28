# Implementation Plan: All 13 Improvements for #1 Finish

## Task Type
- [x] Backend (Python/PyTorch/FastAPI)

## Current State (Verified 2026-03-28)
- 187 tests pass, 97% coverage
- 6 tasks, all endpoints working, WS task selection works
- Docker 1.48GB, baseline reproducible, openenv validates
- Missing: real training curves, LLM scores, 2nd architecture, Task 7, Docker optimization

---

## Phase 0: Repo Cleanup (5 min)

**Files**: None to create
**What**: Verify clean state, ensure no stale files
**Acceptance**: `pytest` passes, `openenv validate` passes

---

## Phase 1: Add SimpleMLP Architecture (Tier 1, Item 3)

**Files to create**: None (add to `pytorch_engine.py`)
**Files to edit**: `ml_training_debugger/pytorch_engine.py`, `ml_training_debugger/scenarios.py`

**What**:
- Add `SimpleMLP(nn.Module)` class — 3 hidden layers, ~20K params, BatchNorm, ReLU
- Add `model_type` field to `ScenarioParams` (Literal["cnn", "mlp"])
- Use torch.Generator to randomly pick CNN or MLP at `sample_scenario()` time
- Update `create_model_and_inject_fault()` to use selected model type
- Update `extract_gradient_stats()` layer names for MLP

**Pseudo-code**:
```python
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=128, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)
```

**Tests**: New tests in `test_pytorch_engine.py` for SimpleMLP
**Acceptance**: Both CNN and MLP instantiate, fault injection works on both, gradient extraction works

---

## Phase 2: Replace Parametric Curves with Real Mini-Training (Tier 1, Item 2)

**Files to edit**: `ml_training_debugger/simulation.py`, `ml_training_debugger/pytorch_engine.py`

**What**:
- Add `run_real_training(model, scenario, epochs=20) -> dict` to `pytorch_engine.py`
- Returns `{"loss_history": [...], "val_acc_history": [...], "val_loss_history": [...]}`
- Use real forward+backward on random CIFAR-10 style data
- Cache results in module-level `_TRAINING_CACHE: dict[tuple[str, int], dict]` keyed by (task_id, seed)
- Update `simulation.py` to call real training instead of parametric formulas
- Keep `torch.manual_seed(seed)` for reproducibility
- Fallback to parametric if cache miss and training too slow (>3s)

**Key constraints**:
- 20 epochs on SimpleCNN with batch_size=16 takes ~0.5-1s on CPU
- Cache means second reset() with same task/seed is instant
- Must still be deterministic (torch.manual_seed)

**Tests**: Verify loss histories come from real training, are reproducible across runs
**Acceptance**: `baseline_heuristic.py` produces identical scores on two runs with real curves

---

## Phase 3: Add Task 7 — LR Scheduler Bug (Tier 1, Item 4)

**Files to edit**: `models.py`, `scenarios.py`, `simulation.py`, `pytorch_engine.py`, `graders.py`, `reward_engine.py`, `server/app.py`, `openenv.yaml`, `baseline_heuristic.py`, `README.md`

**What**:
- Add `SCHEDULER_MISCONFIGURED = "scheduler_misconfigured"` to `RootCauseDiagnosis`
- Add `task_007` to `sample_scenario()` — medium-hard difficulty, max_steps=25
- Scenario: training starts OK for first N epochs, then LR scheduler kicks in with wrong gamma/step_size, causing performance degradation
- Agent must inspect config + loss curve inflection point
- New grader: `grade_task_007()` — rewards inspecting config, identifying scheduler issue, fixing it
- Add `fix_scheduler` to action space (or reuse `modify_config` with target `lr_scheduler_gamma`)
- Update `/health` to return `"tasks": 7`
- Update `/tasks` to include task_007
- Update heuristic baseline to handle task_007
- Add to openenv.yaml

**Pseudo-scenario**:
```python
if task_id == "task_007":
    gamma = _choose([0.01, 0.001, 0.0001], rng)  # way too aggressive
    step_size = _choose([2, 3, 5], rng)
    return ScenarioParams(
        task_id=task_id,
        root_cause=RootCauseDiagnosis.SCHEDULER_MISCONFIGURED,
        seed=effective_seed,
        scheduler_gamma=gamma,
        scheduler_step_size=step_size,
        max_steps=25,
        notes="LR scheduler was recently added to improve convergence.",
    )
```

**Tests**: Full lifecycle test for task_007, grader test
**Acceptance**: task_007 works end-to-end, heuristic baseline handles it

---

## Phase 4: Add Difficulty Scaling (Tier 2, Item 6)

**Files to edit**: `scenarios.py`, `server/environment.py`

**What**:
- Add `difficulty_level: int = 3` to `ScenarioParams` (1-5)
- Accept `difficulty_level` in `reset()` kwargs
- Scale noise, red herring intensity, and ambiguity based on level:
  - Level 1: obvious signals, no noise, no red herrings
  - Level 3: default (current behavior)
  - Level 5: max noise, multiple red herrings, ambiguous signals
- Affects: noise amplitude in curves, red herring intensity, number of misleading notes

**Acceptance**: `reset(task_id="task_005", difficulty_level=1)` produces clearer signals than level 5

---

## Phase 5: Add Curriculum, Leaderboard, Replay Endpoints (Tier 2 + Tier 3)

**Files to edit**: `server/app.py`

**What**:
- `GET /curriculum` — returns ordered task list for training:
  ```json
  {"curriculum": [
    {"task_id": "task_001", "difficulty_level": 1},
    {"task_id": "task_001", "difficulty_level": 3},
    ...
    {"task_id": "task_005", "difficulty_level": 5}
  ]}
  ```
- `GET /leaderboard` — returns sorted episode scores from `_baseline_results`
- `GET /replay/{episode_id}` — returns full action/observation trace for an episode
- For replay: store action/observation history in `SessionData`

**Acceptance**: All 3 endpoints return valid JSON

---

## Phase 6: Add Confusion Matrix to Data Batch Stats (Tier 3, Item 10)

**Files to edit**: `models.py`, `simulation.py`

**What**:
- Add `confusion_matrix: Optional[list[list[float]]]` to `DataBatchStats`
- Generate 10x10 confusion matrix in `gen_data_batch_stats()`
- For data leakage: high diagonal, some off-diagonal leakage
- For overfitting: perfect diagonal for train, scattered for val
- For normal: moderate diagonal with realistic confusion

**Acceptance**: `inspect_data_batch` returns confusion_matrix field

---

## Phase 7: Exploit Resistance Proof (Tier 2, Item 8)

**Files to create**: `tests/test_exploit_resistance.py`
**Files to edit**: `README.md`

**What**:
- Test that runs all 7 tasks with seeds 1-100
- Records score variance per task
- Asserts no single strategy works across all seeds (std > 0 for hard tasks)
- Add results table to README

**Acceptance**: Test passes, README shows variance table

---

## Phase 8: PAPER.md (Tier 3, Item 13)

**Files to create**: `PAPER.md`

**What**: 1-page research summary:
- Title: "Context-Gated Reward Shaping for Evidence-Based ML Debugging"
- Abstract, motivation, method (context-gated penalty), environment design, results, conclusion
- Include baseline comparison table
- ~500-800 words

**Acceptance**: PAPER.md exists and reads well

---

## Phase 9: LLM Baseline (Tier 1, Item 1)

**Files to edit**: `baseline_inference.py`, `README.md`

**What**:
- This requires OPENAI_API_KEY from the user
- Run `python baseline_inference.py` with real API key
- Record scores for all 7 tasks
- Update README with comparison table
- If no API key available: document expected behavior and add placeholder scores

**Acceptance**: README has heuristic vs LLM comparison table

---

## Phase 10: Final Polish + Docker + README + Smoke Test

**Files to edit**: `Dockerfile`, `README.md`, `deploy-hf.sh`

**What**:
- Docker: Already at 1.48GB — document the trade-off (libtorch_cpu.so is 426MB minimum)
- Create `deploy-hf.sh` script
- Update README with all new features (Task 7, difficulty scaling, curriculum, leaderboard, replay, confusion matrix)
- Final smoke test: all tests pass, all endpoints work, baseline reproducible

**Acceptance**: Everything green, ready to submit

---

## Key Files to Create/Edit

| File | Operation | Phase | Description |
|------|-----------|-------|-------------|
| `ml_training_debugger/pytorch_engine.py` | Modify | 1,2 | Add SimpleMLP, real training, caching |
| `ml_training_debugger/models.py` | Modify | 3,6 | Add scheduler_misconfigured enum, confusion_matrix |
| `ml_training_debugger/scenarios.py` | Modify | 1,3,4 | Add model_type, task_007, difficulty_level |
| `ml_training_debugger/simulation.py` | Modify | 2,6 | Real training curves, confusion matrix |
| `ml_training_debugger/graders.py` | Modify | 3 | Add grade_task_007 |
| `server/app.py` | Modify | 3,5 | Task 7, curriculum, leaderboard, replay endpoints |
| `server/environment.py` | Modify | 4,5 | Difficulty scaling, replay storage |
| `openenv.yaml` | Modify | 3 | Add task_007 |
| `baseline_heuristic.py` | Modify | 3 | Handle task_007 |
| `README.md` | Modify | 7,9,10 | Exploit resistance, LLM scores, new features |
| `PAPER.md` | Create | 8 | Research summary |
| `deploy-hf.sh` | Create | 10 | HF deployment script |
| `tests/test_exploit_resistance.py` | Create | 7 | 100-seed variance test |

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Real training slows reset() beyond 3s | Cache per (task_id, seed); MLP is faster than CNN |
| Task 7 breaks existing tests | Run full suite after each phase |
| LLM baseline needs API key | Document expected behavior; user provides key |
| Docker can't go below 1.4GB | Document trade-off; libtorch_cpu.so is irreducible |
| SimpleMLP gradient patterns differ | Adapt extract_gradient_stats for MLP layers |

## SESSION_ID
- CODEX_SESSION: N/A
- GEMINI_SESSION: N/A
