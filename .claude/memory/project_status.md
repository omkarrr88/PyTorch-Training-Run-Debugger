---
name: Project Status as of 2026-03-28
description: Current build/test/deployment status, what's working, what's pending, and known issues.
type: project
---

## Status: Code Complete, Deployment Pending

**Last verified**: 2026-03-28

### Passing
- 183/183 tests pass (5.84s)
- 97% coverage on `ml_training_debugger/` package
- `openenv validate` → `[OK] ML Debugger: Ready for multi-mode deployment`
- Baseline bit-exact reproducible across runs
- All 10 endpoints verified (health, tasks, grader, baseline, dashboard, validation-report, schema, state, docs, ws)
- Docker builds and serves correctly on port 7860
- Zero numpy in core, `import torch` in every core module
- Typed Pydantic models everywhere
- Context-gated penalty fires correctly (both paths tested)

### Docker Image
- Size: **1.48GB** (down from 1.96GB via single-layer cleanup)
- `libtorch_cpu.so` is 426MB — the irreducible PyTorch CPU minimum
- Spec target was <500MB (aspirational for PyTorch-native env)
- **Cannot remove**: torch/testing, torch/distributed, torch/cuda (all required at import time)
- **Safe to remove**: torch/test, torch/include, torch/share, torch/utils/benchmark, torch/utils/bottleneck, torch/utils/tensorboard, torch/lib/*.a, test .so files, caffe2, .pyi files

### Pending
- [ ] Push to **public GitHub repo**
- [ ] Deploy to **HF Spaces** (Docker type, tag with `openenv`)
- [ ] Submit HF Space URL + GitHub repo URL

### Known Limitations
- WS reset defaults to task_001 (framework limitation — no extra fields accepted)
- HTTP `/step` has session isolation issues (framework creates new env instances per request)
- `replace_optimizer` and `rollback_checkpoint` are no-op actions (acceptable)
- Heuristic only handles 2/4 code bug variants (eval_mode, detach_loss)
- Validation report at `/validation-report` is hardcoded, not computed from real runs
