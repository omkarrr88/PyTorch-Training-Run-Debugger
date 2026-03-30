---
name: Project Status as of 2026-03-30
description: Current build/test/deployment status, verified metrics, known limitations, and remaining work.
type: project
---

## Status: Code Complete, Deployment Pending

**Last verified**: 2026-03-30

### Verified Metrics
- **251 tests pass** (60s runtime due to real training)
- **95% coverage** on ml_training_debugger/ + server/
- **openenv validate** → `[OK] ML Debugger: Ready for multi-mode deployment`
- **Baseline bit-exact reproducible** across runs
- **Docker image: 885MB** (down from 1.96GB — 55% reduction)
- **Docker uses torch 2.5.1+cpu** (multi-stage build, strip --strip-unneeded)
- **8/8 validation checks pass** (real training curves)
- **All endpoints work** (health, tasks, grader, baseline, dashboard, validation-report, curriculum, leaderboard, replay, schema, ws)
- **All 7 tasks selectable via WS**: `{"type": "reset", "data": {"task_id": "task_007"}}`

### Baseline Scores (Heuristic)
```
task_001: 1.0, task_002: 1.0, task_003: 1.0, task_004: 0.45,
task_005: 1.0, task_006: 1.0, task_007: 1.0
```

### LLM Baseline Scores (Measured)
- **Llama 3.3 70B** (Groq): 1.0, 1.0, 0.4, 0.45, 1.0, —, — (5/7 before rate limit)
- **Llama 3.1 8B** (Cerebras): 0.6, 0.05, 0.4, 0.6, 1.0, 0.6, 0.6 (avg 0.55)
- **Llama 3.1 8B** (Groq): 0.6, 0.05, 0.4, 0.6, 1.0, 1.0, 0.6 (avg 0.61)

### Features Implemented
- 7 tasks with 3 difficulty tiers + difficulty scaling (1-5)
- Dual architecture: SimpleCNN + SimpleMLP
- Real 20-epoch PyTorch mini-training (cached per task/seed)
- Context-gated reward penalty
- Code-level debugging (Task 6, 4 bug variants, AST validation)
- Task 7: LR Scheduler misconfigured
- Confusion matrix in data batch stats
- Curriculum, leaderboard, replay endpoints
- PAPER.md research summary
- EXPLANATION.md simple explanation
- Multi-provider LLM baseline (Groq, Cerebras, Gemini, OpenAI)
- Exploit resistance test (20-seed variance)
- deploy-hf.sh deployment script

### Pending
- [ ] Push to **public GitHub repo**
- [ ] Deploy to **HF Spaces** (Docker type, tag `openenv`)
- [ ] Run 70B baseline for tasks 6-7 (Groq quota resets daily)
- [ ] Record dashboard GIF for README

### Docker Size History
1.96GB → 1.48GB → 1.09GB → **885MB** (irreducible: libtorch_cpu.so=329MB stripped)

### Known Limitations
- Docker 885MB (target was 500MB — libtorch_cpu.so is irreducible)
- HTTP /reset and /step are stateless (framework design — WS is primary interface)
- Heuristic outperforms LLMs on most tasks (environment rewards domain knowledge)
- `replace_optimizer` and `rollback_checkpoint` are no-op actions
