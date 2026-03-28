---
name: Key spec documents and their roles
description: Which files are source of truth for what, and how they relate to each other.
type: reference
---

## Source of Truth Hierarchy

1. **`ml-training-debugger-spec.md`** — THE single source of truth. If anything conflicts with this, the spec wins.
2. **`CLAUDE.md`** — Coding rules, non-negotiable constraints, reward constants, commands. Derived from spec.
3. **`ROADMAP.md`** — Phase-by-phase implementation plan with acceptance criteria.
4. **`PRD.md`** — Product requirements (higher-level than spec).

## Key Spec Sections (by number)
- S5: Context-gated reward shaping (the differentiator)
- S6: PyTorch-native fault injection engine
- S10: Data models (typed Pydantic models)
- S11: The six core tasks (param ranges, grader breakdowns)
- S12: Reward function (7 components, exact constants)
- S13: Environment lifecycle (reset/step/done)
- S14: OpenEnv spec compliance (endpoint contracts)
- S16: Error handling (step() never raises)
- S17: Baseline inference design (heuristic decision tree)
- S18: PyTorch validation suite
- S22: Code fix validation pipeline (normalize → tokenize → semantic → AST)

## Non-Negotiable Rules (from CLAUDE.md)
- Context-gated -0.20 penalty: ONLY when `gradients_inspected=True AND gradients_were_normal=True`
- Task 6 diagnosis is ALWAYS `code_bug` (not `batchnorm_eval_mode` etc.)
- PyTorch-native only — no numpy in core modules
- Grader ≠ reward function (separate modules, separate purposes)
- Opaque task IDs (task_001-task_006, no descriptive names agent can see)
