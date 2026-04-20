# Round 2 Winning Plan — Path to #1

**Hackathon**: Meta PyTorch OpenEnv Hackathon x Scaler School of Technology — Round 2
**On-site dates**: April 25–26, 2026, Bangalore
**Planning date**: 2026-04-20
**Days to on-site**: 5
**Target**: #1 position — realistic ceiling ~58–62% P(#1) with flawless execution **and all seven conditional upgrades landing** (§7 Moves 11–17). Expected-value P(#1) with partial landing: ~52–55%. Baseline P(#1) without upgrades: ~38%. Top 3 ≈ ~82%. Top 6 ≈ ~96%.

---

## 0. Known Limitations (Honest Ceiling)

These are real limits of the project/plan that no further planning can fix. Acknowledging them openly — to yourself, and in Q&A rebuttals when attacked — is how you win judge trust.

### Fundamental limits (cannot be planned away)

| # | Limitation | Consequence | Rebuttal framing |
|---|---|---|---|
| L1 | Generator samples from a **fixed 7-fault discrete space** — curriculum sampling, not true bug invention | RL-background judges will see this in 30 seconds. Capped Innovation score. | "Phase 1 is structured fault sampling with novelty-weighted rewards. Phase 2 (stretch) adds an LLM-based code-mutation Generator. We shipped Phase 1 for 5-day rigor; Phase 2 is a 1-week extension." **L1 is conditionally killed by §7 Move 11 (code-mutation Generator) if the Day-2 smoke test passes.** |
| L2 | `run_real_training()` per episode is slow (sec-to-min). **PPO/GRPO often needs 10k–100k episodes; we have budget for ~1k.** | Training curves may not converge on-site. The "self-improvement" claim is fragile. | "We use a reduced-epoch training oracle (5 mini-epochs) as validation. Convergence verified pre-onsite." |
| L3 | **Solo execution** (or small team) vs 60+ hour scope across 5 days | One rabbit-hole day = plan collapse | Mitigated via ruthless scope cuts (see §11 updated) and committed fallback artifacts |
| L4 | **Frontier LLMs (Claude 3.5, GPT-4) will likely outperform a small self-play-trained Solver on Mode C human bugs** | Can't claim "beats frontier" — must pivot to "competitive at 1/100th inference cost" | Lead with cost + latency numbers. Frame as "specialized vertical model" not "better general model." |
| L5 | **ML-debugging is a crowded problem space** (W&B, internal Anthropic/OpenAI tools, multiple HF projects) | Novelty capped regardless of execution quality | Differentiate on framing: "first self-play environment with verifiable PyTorch training oracle" |
| L6 | **Theme 3.1 fit is fuzzy** (closed PyTorch env, not open tools/APIs) | Strong judges may downgrade to Theme 4 only | Lead with Theme 4 (Self-Improvement). Theme 3.1 is secondary, not primary. |
| L7 | **"Live demo" is pre-trained inference, not live RL training** (RL takes hours) | Sophisticated judges will notice pretense | Be explicit in pitch: "demonstrating a trained agent, not live learning." Honesty earns trust. |
| L8 | **Mode C n=5–10 has no statistical power** | Strong reviewers discount entirely | Target n≥30 via Stack Overflow + GitHub Issues scrape. Still weak but defensible. |
| L9 | **R1→R2 theme-alignment DQ risk** (rules mandate same problem statement) | If R1 submission didn't explicitly match themes, pivot could be DQed | §15.5 pre-Day-1 audit of R1 language required. This is blocker #1. |
| L10 | **Hackathon winner selection has high judge-variance.** Hundreds of teams, ~30 min attention per judge, fatigue after 10+ pitches. | Best plan ≠ best outcome. Pitch slot, judge mood, live demo reliability dominate. | None. Build redundancy into pitch (demo + GIF + transcripts + reproducibility one-liner). |

### Realistic ceiling
| Tier | Probability with this plan + flawless execution |
|---|---|
| **Top 6 (finalist)** | ~88% |
| **Top 3** | ~65–70% |
| **#1 overall** | **~30–40%** |

**Anyone quoting >45% P(#1) is not being honest about variance.** The earlier "57% P(#1)" in §13 was generous; the recalibrated table there reflects the honest number.

### What this means for pitch framing
- Don't oversell. Judges smell it.
- Lead with honesty about Phase 1 scope, then show the Phase 2 design (LLM code-mutation Generator) as a "this is where it goes next" closing slide. **Roadmap credibility > fake capability claims.**
- Pre-load L1, L4, L7 into pitch itself — state the limitation, then state the mitigation. Pre-empting beats defending.

---

## 1. Decision Summary

### The verdict
**Pivot the existing ML Debugger into a Self-Play Bug Generator + Solver environment.**

- **Do NOT continue as-is** → ceiling is Top 5, not #1
- **Do NOT switch to a fresh project** → 5 days is not enough to rebuild OpenEnv infrastructure
- **DO extend the existing project** with one new agent (Generator) and a self-play reward loop

### The one-line pitch
> **"Agents teaching each other to debug ML through self-play — a recursive curriculum where one agent invents novel PyTorch bugs and another learns to diagnose them."**

### Reuse vs new work
| Component | Status |
|---|---|
| PyTorch engine, fault injection, real training | **Reuse 100%** |
| 7 existing tasks + graders | **Reuse 100%** |
| Reward engine, scenarios, simulation | **Reuse 100%** |
| OpenEnv server, endpoints, Docker, 251 tests | **Reuse 100%** |
| Generator agent | **NEW** |
| Self-play reward wiring | **NEW** |
| TRL training script | **NEW** |
| Frozen Oversight layer (stretch) | **NEW (small)** |

~70% code reuse. Same repo. Same HF Space. Reframed as "ML Debugger v2: Self-Play Edition."

---

## 2. Theme Alignment

### Primary themes hit

**Baseline (no conditional upgrades):**
| Theme | Role | Fit | Notes |
|---|---|---|---|
| **Theme 4 — Self-Improvement** | **PRIMARY** | Strong (core) | Generator + Solver self-play = textbook Theme 4 recursive skill amplification |
| **Theme 3.1 — World Modeling** | Secondary (fuzzy) | Medium | Closed PyTorch env is a weakness |
| **Theme 1 — Multi-Agent** | Tertiary (stretch) | Weak | Needs Oversight layer to claim |

**With Move 15 (Multi-Generator Tournament) landed:**
| Theme | Role | Fit | Notes |
|---|---|---|---|
| **Theme 4 — Self-Improvement** | PRIMARY | Strong | Core self-play loop |
| **Theme 1 — Multi-Agent Interactions** | **SECONDARY (real claim)** | Strong | 3 Generators compete via Elo — genuine multi-actor |
| **Theme 3.1 — World Modeling** | Tertiary | Medium | Partial fit |

**With Move 17 (Cross-Framework Transfer) also landed:**
| Theme | Role | Fit | Notes |
|---|---|---|---|
| **Theme 4 — Self-Improvement** | PRIMARY | Strong | Core self-play loop |
| **Theme 1 — Multi-Agent Interactions** | SECONDARY | Strong | 3 Generators + Elo ladder |
| **Theme 3.1 — World Modeling** | **TERTIARY (defensible)** | Strong | Zero-shot cross-framework transfer = genuine world model |

**Rationale**: Theme 3.1 expects "real interaction with tools, APIs, or dynamic systems." Moves 15 + 17 together convert your plan from single-theme to **genuine triple-theme coverage**. Strong research judges no longer have a plausible "this is curriculum learning, not multi-agent" critique.

### Sub-theme bonus prize shots
| Sponsor | Sub-theme | How we hit it | Framing language to use in pitch |
|---|---|---|---|
| **Snorkel AI** | Simulated Experts-in-the-Loop | G_specialist (Move 15) = the simulated SME whose targeting shifts as Solver improves | Call G_specialist "the simulated SME" on slides — not "the adversary" |
| **Mercor** | Capped/uncapped rewards scaling with token output | Solver's inspection chain consumes tokens; longer reasoning = higher diagnosis accuracy, reward scales with tokens spent | Show "tokens-vs-reward" curve; mention "uncapped inspection budget" explicitly |
| **Halluminate** | Multi-Actor Environments | With Move 15: 3 Generators + 1 Solver = genuine 4-actor environment with Elo competition | Frame Elo ladder as "emergent multi-actor coordination" — this is now a real claim, not a stretch |
| **Fleet AI** (stretch) | Scalable Oversight | Frozen Oversight agent monitors Solver decisions + emits interpretability report | Output a JSON "oversight report" per episode; show one on slide |

**Minimum**: 1 sub-theme bonus prize locked (Snorkel).
**Target**: 2 bonuses (Snorkel + Mercor — both require zero extra engineering).
**Stretch with Move 15**: 3 bonuses (Snorkel + Mercor + **Halluminate becomes genuine**).
**Max stretch**: 4 bonuses (add Fleet AI via the Oversight layer — not prioritized).

---

## 3. The Environment Design

### Agent topology
```
┌─────────────┐       ┌──────────────┐       ┌───────────────┐
│  GENERATOR  │──────▶│    SOLVER    │──────▶│  OVERSIGHT    │
│   (NEW)     │  bug  │  (existing)  │  dx   │  (frozen/NEW) │
└─────────────┘       └──────────────┘       └───────────────┘
      ▲                      ▲                       │
      │ reward = 1 - solver  │ reward = existing     │ reward = catch rate
      │                      │                       │
      └──────────── adversarial self-play ───────────┘
```

### Generator agent — action space
- `fault_type` ∈ {lr_too_high, vanishing_gradients, data_leakage, overfitting, batchnorm_eval_mode, code_bug, scheduler_misconfigured}
- `hyperparameters`: structured ranges per fault_type (e.g., lr multiplier, dropout rate)
- `code_bug_template_id` (for Task 6 variants)
- `difficulty_level` ∈ {1..5}

### Solver agent — existing capabilities
- All current actions: `inspect_gradients`, `inspect_weights`, `inspect_data_batch`, `diagnose`, `apply_fix`, etc.
- Unchanged — reuse entire existing action space

### Oversight agent (stretch, frozen)
- Rule-based gate: flags Solver diagnosis if confidence < threshold OR contradicts gradient evidence
- No training needed — pure logic
- Adds Fleet AI sub-theme bonus at ~4 hours of work

### Reward model (formalized)

Let `τ` = a generated bug configuration, `s_solver(τ) ∈ [0,1]` = Solver's holistic grader score on `τ`, `𝟙[invalid]` = 1 if the bug fails the validation gate, and `novelty(τ)` = cosine distance between `τ`'s embedding and the nearest-neighbor in the generated history buffer.

```
R_generator(τ)  = λ₁·(1 - s_solver(τ))  +  λ₂·novelty(τ)  -  λ₃·𝟙[invalid]
R_solver        = existing per-step + holistic grader  (unchanged)
R_oversight(d)  = +1 · 𝟙[caught bad diagnosis d]  -  1 · 𝟙[false block]
```

- `λ₁ = 1.0` (difficulty term, primary)
- `λ₂ = 0.3` (diversity term, prevents Generator mode-collapse)
- `λ₃ = 1.0` (invalidity penalty, integrity gate)

The `λ₂` novelty term is critical — without it, Generator mode-collapses to the single hardest bug family.

### Validation gate (integrity)
Generated bugs MUST actually break training (verified via `run_real_training()`). Invalid bugs trigger the `λ₃` penalty. This prevents the Generator from gaming the reward by producing nonsense or lr=1e10 degenerate cases.

---

## 4. Tasks Definition

### Core tasks (existing 7, reused as Solver's starting curriculum)
- task_001: lr_too_high (Easy)
- task_002: vanishing_gradients (Easy)
- task_003: data_leakage (Medium)
- task_004: overfitting (Medium)
- task_005: batchnorm_eval_mode (Hard)
- task_006: code_bug (Hard, 4 variants)
- task_007: scheduler_misconfigured (Med-Hard)

### Generated tasks (NEW — Generator produces unbounded task distribution)
- Generator samples `(fault_type, hyperparameters, difficulty_level)` tuples
- Each sample becomes a novel task the Solver has never seen
- Curriculum emerges dynamically — no fixed task list

---

## 5. Evaluation Logic

### Three evaluation modes

**Mode A — Solver on existing 7 tasks (baseline + catastrophic-forgetting guard)**
- Unchanged from Round 1
- Measures: does Solver trained via self-play still do well on the original curriculum?
- **Regression guard**: if Mode A score drops >5% after self-play, we lost. Treat as release-blocker.

**Mode B — Solver on Generator-produced tasks**
- Hold-out split: 80% seen during training, 20% held out for evaluation
- Measures: self-play generalization

**Mode C — Solver on PyTorchBugBench-v0 (human-authored held-out, KILLER METRIC)**
- Rebrand the human-bug set as a named benchmark we open-source: **PyTorchBugBench-v0**
- 5–10 real ML engineers submit PyTorch bugs Generator has never seen
- Measures: generalization to real-world distribution
- **This is the single most important number in the submission**
- **Legacy framing**: "We shipped a community benchmark, not just a submission"

### Bug-novelty / diversity metric (anti-"just permutations" defense)
Pre-empts the obvious Q&A attack. Track on pitch slide:
- **Embedding novelty**: mean cosine distance of generated bugs to existing 7 in config-embedding space
- **Distinct-N**: unique `(fault_type, discretized-hyperparameter)` tuples generated
- **UMAP coverage plot**: 2D projection showing Generator's distribution expanding over training epochs

### Ablation table (required on pitch slide)

**Accuracy + cost-efficiency (Move 13 integrated)**:

| Solver training setup | Mode A | Mode B | Mode C (PyTorchBugBench-v0) | $/100 correct dx | Latency/dx |
|---|---|---|---|---|---|
| Baseline (no training) | ? | ? | ? | — | ~1 s |
| **Frontier: GPT-4-Turbo zero-shot** | ? | ? | **?** | ~$3.00 | ~10 s |
| **Frontier: Claude 3.5 Sonnet zero-shot** | ? | ? | **?** | ~$2.50 | ~8 s |
| **Frontier: Llama-3.3-70B zero-shot** | ? | ? | **?** | ~$0.35 | ~12 s |
| Trained vs Random Generator | ? | ? | ? | **~$0.03** | **~3 s** |
| Trained vs Heuristic Generator | ? | ? | ? | **~$0.03** | **~3 s** |
| **Trained via Self-Play (structured Gen)** | **?** | **?** | **?** | **~$0.03** | **~3 s** |
| **Trained via Self-Play (code-mutation Gen) ★** | **?** | **?** | **?** | **~$0.03** | **~3 s** |

★ Final row requires Move 11 to land (Day-2 smoke test pass). Otherwise row is deleted from slide.

**Critical**: the frontier-LLM rows are non-negotiable. If asked *"how does this compare to just prompting Claude?"* and you don't have the number, you lose. Run zero-shot GPT-4 / Claude / Llama against Mode C and put the result on the slide.

**Self-play must be highest on Mode C accuracy OR cheapest per correct dx (ideally both).** If frontier LLM beats self-play on accuracy, the narrative pivot is already baked into the cost columns: *"50–100× cheaper per correct diagnosis at competitive accuracy — specialization dominates for CI/CD-integrated debugging."*

### Bootstrap confidence intervals (statistical defense)
**Weakness**: Mode C with n=30 has weak statistical power. Strong reviewers will dismiss single-point numbers.

**Fix**: report all Mode C scores as **point estimate + 95% bootstrap CI** (1000 resamples).

Example: `Self-Play: 0.72 [95% CI 0.58–0.84]` instead of bare `0.72`.

If confidence intervals overlap between two rows, acknowledge openly: *"accuracy difference not statistically significant at n=30; this is why we ship PyTorchBugBench-v0 for community expansion."* Honesty again earns trust.

---

## 6. Post-Training / Self-Improvement Strategy

### Training loop
1. Generator samples a bug
2. Validate bug actually breaks training (filter)
3. Solver attempts diagnosis
4. Compute rewards for both agents
5. Update both via TRL (PPO or GRPO)
6. Every N steps: difficulty ratchet — Generator rewarded more for bugs Solver fails at

### Self-improvement signal
- **Generator's bug-difficulty curve** climbs over time
- **Solver's accuracy-vs-difficulty curve** climbs over time
- Elo-like separation between the two
- **Two-curves-on-one-chart** = the iconic pitch visual

### Compute budget (on-site 25-26)
- Use provided HuggingFace credits
- Target: 1000+ self-play episodes during on-site training
- Record full curve logs for the pitch

---

## 7. Killer Moves (What Separates #1 from Top 6)

### Move 1 — Held-out human-authored bug test
**Why it wins**: Proves self-play generalized to real-world distribution. Single number beats every other team's claims.
**Action**: Collect 5–10 bugs from ML engineers on X/Discord/HF forums by **Day 3**.

### Move 2 — Live pitch demo
**Why it wins**: Every team shows slides. You show agents fighting live. Judges watching 15 pitches in a row remember the ONE team with a live demo.
**Action**: Prepare a reliable, rehearsed inference script that runs Generator vs Solver live with scores updating on screen.

### Move 3 — Ablation table
**Why it wins**: 10-second research credibility. Preempts 90% of Q&A skepticism.
**Action**: Record Random/Heuristic/Self-Play generator scores. Put on single slide.

### Move 4 — Frozen Oversight layer
**Why it wins**: Unlocks Fleet AI sub-theme bonus prize at ~4 hours of work. Two sub-theme shots > one.
**Action**: Rule-based gate on Solver confidence + gradient evidence contradiction.

### Move 5 — Red-team narrative framing
**Why it wins**: "Automated red-teaming for ML systems" upgrades the project from "cute ML tool" to "AI safety infrastructure." Meta + HF judges care deeply about this framing.
**Action**: Use the word "red-team" in pitch, blog, and HF Space description.

### Move 6 — Reproducibility proof
**Why it wins**: Judges love `seed + command → exact replay`. Research credibility.
**Action**: Add a single-command reproducibility demo: `python reproduce_pitch.py --seed 42`

### Move 7 — Frontier LLM head-to-head (THE credibility killer)
**Why it wins**: Preempts the #1 Q&A attack ("why not just prompt Claude?"). A single slide showing trained-Solver-vs-GPT-4-vs-Claude on PyTorchBugBench-v0 ends the debate in 10 seconds.
**Action**: On Day 3, run zero-shot GPT-4, Claude 3.5 Sonnet, and Llama-70B on Mode C bugs. Record scores. Put on slide.

### Move 8 — Interaction transcripts as slide content
**Why it wins**: Reward curves are abstract. A 4-turn Generator↔Solver dialog showing the Generator inventing a subtle batchnorm bug and the Solver catching it reads like a story. Judges remember stories.
**Action**: Save top 5 most interesting transcripts. Include 1 full transcript on slide 3. Link remaining 4 in blog.

### Move 9 — Build-in-public daily thread
**Why it wins**: Cheap distribution. Sponsors (Meta, HF, Mercor, Snorkel, Fleet, Halluminate) have social media teams that index hackathon hashtags. If your work appears before the pitch, judges recognize you on the slide.
**Action**: Daily X thread Days 1–5 tagging `@Meta @huggingface` and the sponsor(s) whose theme you hit that day. Include curves, GIFs, code snippets. Total cost: 10 min/day.

### Move 10 — PyTorchBugBench-v0 as community benchmark
**Why it wins**: "Submission" vs "Benchmark shipped" — latter sounds 5x more credible. Judges weight legacy potential.
**Action**: Publish the human bug set to HuggingFace Datasets with a real name, README, and eval script. Mention in pitch close.

---

### Move 11 — LLM code-mutation Generator (KILLS LIMITATION L1) ★ CONDITIONAL UPGRADE

**The single biggest innovation lever in the plan.** Structured fault sampling is bandit curriculum learning; code-mutation is genuine bug invention. This move transforms the Innovation score from 28–32/40 to 35–37/40.

**Architecture**:
- Qwen2.5-3B (separate LoRA adapter, shared base weights with Solver)
- **Input**: clean PyTorch training script (fixed template) + current difficulty budget
- **Output**: unified-diff patch (text format, ~5–20 lines) that mutates the training code
- **Validation pipeline** (reuses your existing oracle):
  1. Diff applies cleanly (`git apply --check`)
  2. Mutated code parses (AST validator — you already have this)
  3. Mutated code imports and runs
  4. `run_real_training()` on mutated code produces a FAILURE (loss NaN, divergence, or eval-metric collapse beyond threshold)
  5. Solver cannot trivially solve within 3 steps (prevents degenerate easy bugs)
- **Reward**: `R_gen_mut = λ₁·(1 - s_solver) + λ₂·novelty_code(τ) + λ₄·diff_minimality - λ₃·𝟙[invalid]`
  - `λ₄ = 0.2` rewards minimal/elegant bugs over "add 500 lines of noise"
  - `novelty_code` = embedding distance on the diff text, not the fault-type tuple

**Seed strategy (critical)**:
- Pre-populate Generator with **20 hand-written example diffs** covering your 7 fault families (few-shot prompt)
- First 100 episodes: in-context generation only (no fine-tuning) to build a cache of valid diffs
- After 100 validated diffs in the cache, start GRPO fine-tuning on (diff → reward) pairs

**Scope**: **1 day** if Day-2 afternoon smoke test passes. Drop entirely if it fails. See §15.5 new blocker.

**Why it wins**: Converts your Phase 2 roadmap slide from "we plan to" into "we shipped." Judges can see the Generator write a novel diff on screen during live demo. **+5 Innovation, +2 Storytelling. Expected: +7 total points.**

**Risk**: early-episode invalid-diff rate may hit 90%. Mitigation: the 20-seed few-shot cache + in-context-only warmup phase means you have valid diffs on Day 2 even if fine-tuning never converges. The Generator is demoable without training — it's just an LLM writing PyTorch bugs.

**Fallback if shipped but fine-tuning flat**: frame as "validated LLM code-mutation pipeline; GRPO convergence is the 48-hour on-site extension." Still wins over structured generator because you shipped arbitrary code mutation.

---

### Move 12 — Publish ONE surprising empirical finding ★ CONDITIONAL UPGRADE

Every pitch shows "curves go up." Win #1 requires **one specific quotable empirical result** that is genuinely surprising. Instrument for this on Day 2; it either materializes or doesn't.

**Candidate findings (pick the one with strongest signal after Day 3 data):**

| Finding | How to measure | Likelihood signal materializes | Why judges care |
|---|---|---|---|
| **Generator discovers a bug family outside initial taxonomy** | Cluster Generator's output diffs; look for high-novelty-score diffs that don't map to any of the 7 seed categories | 30% (requires Move 11) | "Emergent capability" narrative — this is Meta/HF catnip |
| **Solver confidence calibration improves after self-play (ECE ↓)** | Expected Calibration Error before vs after self-play training | 55% | Calibration is an active AI-safety research area at Anthropic/OpenAI |
| **Mode C > Mode B generalization** | Compare Solver accuracy on held-out human bugs vs held-out generated bugs | 25% (counter-intuitive) | Would directly refute L5 ("crowded space") — genuinely novel claim |
| **Inference-compute scaling on the Solver: tokens-spent-on-inspection → accuracy** | Plot Solver diagnosis accuracy vs tokens used per episode | 75% (easy to produce) | Unlocks Mercor sub-theme; directly quotable as "test-time scaling for ML debugging" |

**Priority order**: run the measurement for all 4 on Day 3 (none requires new code — only logging + a matplotlib cell). Pick the 1 with the strongest signal for pitch slide 4. The inference-compute finding (row 4) is the **floor guarantee** — it's almost certain to produce a result.

**Action**: Day 3 afternoon — add logging hooks for ECE, per-episode token counts, bug-cluster embeddings. Analyze Day 5 morning. Produce 1 finding + 1 chart + 1 quotable sentence.

**Why it wins**: Judges remember one finding, not ten. A quotable sentence ("self-play Solvers show 3× better confidence calibration on unseen bugs") travels in post-pitch deliberation. **+3 Innovation, +2 Storytelling. Expected: +5 total.**

---

### Move 13 — Cost-per-correct-diagnosis column ★ CONDITIONAL UPGRADE (LOW-EFFORT)

Transforms L4 ("frontier LLMs probably beat you") from a loss into a **specialized-model win narrative**. Zero additional research work — just a computed column.

**Add to ablation table**:

| Solver | Mode C accuracy | $ per 100 correct dx | Median latency/dx | Memory footprint |
|---|---|---|---|---|
| GPT-4-Turbo zero-shot | ? | $~2.00–5.00 | ~6–15 s | API-only |
| Claude 3.5 Sonnet zero-shot | ? | $~1.50–4.00 | ~4–12 s | API-only |
| Llama-3.3-70B zero-shot | ? | $~0.20–0.50 | ~8–20 s | 140 GB |
| **Self-play Qwen2.5-7B (ours)** | ? | **$~0.02–0.05** | **~2–4 s** | **14 GB** |

**The killer line for the pitch**: *"Our specialized 7B model is 50–100× cheaper per correct diagnosis than GPT-4 — even when GPT-4 scores higher on accuracy. For CI/CD-integrated ML debugging at thousands of runs per day, specialization dominates."*

**Action**: 1 hour of work on Day 3. Record API costs from frontier-LLM baseline run. Measure Solver inference on commodity GPU. Put the table on slide 5.

**Why it wins**: Eliminates the "just use Claude" attack permanently. Converts L4 from rebuttal into feature. Opens the enterprise/CI narrative for the closing slide. **+2 Reward Curves, +1 Storytelling. Expected: +3 total.**

---

---

### Move 14 — Emergent-behavior demo moment ★ EXTRACT-ONLY (no new code)

**The single most memorable slide in the entire deck if it materializes.** Every team shows reward curves; one team shows *"the Generator did something we didn't program."* Judges repeat surprise in deliberation.

**What to look for during Day 3–4 training runs** (instrument now, extract later):
- Generator produces a bug-config that hits a fault *combination* you didn't seed (e.g., LR warmup interacting with dropout in a way none of the 7 families capture)
- Generator exploits a Solver blind spot — repeatedly uses one fault variant until Solver's accuracy collapses
- Generator discovers a degenerate-but-valid bug (e.g., a correct-looking code path that causes silent training failure)
- Solver invents a non-obvious inspection sequence (e.g., always checks data leakage first after a specific Generator signature)

**Action (Claude Code implementation)**:
1. **Add logging hook** in `ml_training_debugger/simulation.py` (or create `ml_training_debugger/emergent_logger.py`):
   - Log every episode: `(generator_output, solver_trajectory, reward, validation_status)` as JSONL to `logs/emergent/episodes.jsonl`
   - Include embedding of Generator output (use `sentence-transformers/all-MiniLM-L6-v2`, cached locally)
2. **Create `scripts/extract_emergent_moment.py`**:
   - Load episodes.jsonl
   - Find the top 3 "surprising" episodes by these criteria:
     - High novelty score (distance > p90 of all generated bugs)
     - Solver accuracy on it falls below 20% AFTER Solver had 80%+ on the nearest neighbor
     - Bug validity rate on family is anomalous (rare success in a rare family)
   - Print trajectory + render matplotlib chart of the moment
3. **Slide 3 template** (prepare in advance, fill content Day 5 morning):
   - Title: "A Moment We Didn't Plan"
   - Show: Generator output (left) + Solver trajectory (right) + one-sentence surprise caption
   - If no moment materializes → delete slide, fall back to "two curves" slide

**Why it wins**: Pitches are remembered by *moments*, not *metrics*. A 10-second "and then this happened" beats a 30-second table. **Gate**: this slide is conditional — only include if a moment actually materializes. No fabrication.

**Risk**: 40% probability a genuine moment materializes. Mitigation: instrument broadly (log everything), extract late (Day 5 morning). If nothing found, pitch falls back to Move 12 empirical finding in this slot.

**Expected impact if landed**: **+3 Innovation, +5 Storytelling = +8 total.**

---

### Move 15 — Multi-Generator tournament (CONVERTS TO THEME 1 PRIMARY) ★ CONDITIONAL UPGRADE

**Ship 3 different Generator strategies competing for Solver-failure reward. Elo ladder among Generators.** This move *fundamentally upgrades* your theme claim: Theme 1 (Multi-Agent Interactions) becomes a defensible PRIMARY alignment, not a stretch. Unlocks Halluminate sub-theme (Multi-Actor Environments) with a real claim, not marketing.

**Three Generator personas**:
| Generator | Strategy | Action-space bias |
|---|---|---|
| **G_greedy** | Maximize immediate Solver-failure reward | Favors hardest known fault families |
| **G_diverse** | Maximize novelty term `λ₂` | Explores edge-of-distribution configs |
| **G_specialist** | Exploits observed Solver weaknesses (tracks per-family Solver accuracy, targets lowest) | Adaptive — forms theory-of-mind-lite |

Each Generator has separate LoRA adapter on the same Qwen2.5-3B base. All three compete for Solver-failure reward; Elo ladder updated after each Solver episode.

**Action (Claude Code implementation)**:
1. **Create `ml_training_debugger/generators/`** directory:
   - `base.py` — `Generator` abstract class (action space + reward hook)
   - `greedy.py` — `GreedyGenerator(Generator)`
   - `diverse.py` — `DiverseGenerator(Generator)`
   - `specialist.py` — `SpecialistGenerator(Generator)` with per-fault-family accuracy tracking
   - `__init__.py` — exports all three
2. **Create `ml_training_debugger/tournament.py`**:
   - `class Tournament` — runs round-robin: each Generator proposes N bugs, Solver attempts, rewards computed
   - `class EloLadder` — maintains Elo rating per Generator, k=32, updates after each episode
   - Output: `logs/tournament/elo_history.jsonl`
3. **Update `server/environment.py`**:
   - Add endpoint `POST /tournament/step` — runs one tournament round
   - Add endpoint `GET /tournament/elo` — returns current Elo ratings
4. **Update `server/dashboard.html`**:
   - Add Elo-ladder panel showing 3 Generators over time
5. **New tests in `tests/test_tournament.py`**:
   - Unit test Elo math (known Elo test vectors)
   - Integration test 10-episode tournament runs end-to-end
6. **Pitch slide**: Elo chart showing G_specialist overtaking G_greedy around episode ~400 (or whatever the real result is)

**Theme alignment upgrade**:
| Theme | Role (before) | Role (after Move 15) |
|---|---|---|
| Theme 4 — Self-Improvement | PRIMARY | PRIMARY (still) |
| **Theme 1 — Multi-Agent Interactions** | Tertiary (stretch) | **Secondary (real claim)** |
| Theme 3.1 — World Modeling | Secondary | Tertiary |

**Sub-theme bonus upgrade**:
- **Halluminate (Multi-Actor Environments)**: now a *real* claim — 3 actors (G_greedy, G_diverse, G_specialist) + Solver = genuine multi-actor
- Snorkel framing still works (G_specialist = adapting SME)

**Scope**: **1 day** (Day 3). Eats the original Day-3 afternoon that was reserved for frontier-LLM baseline collection. **Re-ordering required**: move frontier-LLM baseline to Day 2 afternoon (parallel with Move 11 smoke test — they use different machines), move Move 15 tournament build to Day 3 full day.

**Gate**: only ship if Move 11 goes GREEN or YELLOW *and* Day 2 ends on schedule. If Day 2 overruns, drop Move 15 — reward is high but the day-by-day risk compound is real.

**Expected impact if landed**: **+5 Innovation, +2 Storytelling, +1 sub-theme bonus = +7 total + Halluminate prize shot.**

**Risk**: eats Day 3. Mitigation: parallel tracks (frontier-LLM API calls run during tournament dev; they don't compete for developer time).

---

### Move 16 — Project rename: "BugForge" ★ TRIVIAL UPGRADE (1 hour)

**"ML Debugger v2" is forgettable. "BugForge" is memorable.** Judges deliberating after 15 pitches will retrieve the *name* before the *content*. A distinctive name is a cheap pitch-recall multiplier.

**Candidate names** (ranked by memorability × relevance):
1. **BugForge** — implies creation + iteration. Pairs with "where bugs are forged" tagline.
2. **PyTorch Red Team** — taps into AI-safety framing explicitly.
3. **AdversarialBench** — research-formal, but less memorable.

**Recommendation**: **BugForge** (primary) with tagline *"Where PyTorch bugs are forged — and fixed."*

**Action (Claude Code implementation)**:
1. **Rename repo directories** (if user approves on-disk rename):
   - `ml_training_debugger/` → keep package name (breaking rename is risky); add alias `bugforge/` pointing at same package via `__init__.py` re-export
2. **Update `pyproject.toml`**: add `[project] name = "bugforge"` alongside the existing package
3. **Update `README.md`**: new H1 = "BugForge"; subhead = old project name for SEO
4. **Update `server/dashboard.html`**: title + header → "BugForge Dashboard"
5. **Update `openenv.yaml`**: env name → `bugforge-v0`
6. **Update HuggingFace Space**: rename Space to `bugforge` (reserve the name today)
7. **PyTorchBugBench-v0** → rename to `bugforge/bench-v0` on HF Datasets
8. **All pitch materials**: s/ML Debugger/BugForge/g

**Guardrails**:
- **Do NOT rename the Python package** (`ml_training_debugger`) — breaks all existing imports and the 251 tests. Only add `bugforge` as an alias package that re-exports.
- **Do NOT rename the Git repo** on GitHub unless user explicitly approves (breaks existing URLs).

**Scope**: **1 hour**. Do this Day 1 evening after §15.5 checklist passes.

**Expected impact**: **+2 Storytelling (pitch recall).** Minor but free.

---

### Move 17 — Cross-framework transfer eval (THEME 3.1 BECOMES DEFENSIBLE) ★ CONDITIONAL UPGRADE

**Show the Solver trained on PyTorch bugs also catches ≥1 JAX or TensorFlow bug zero-shot.** This is the "world model" credibility kill-shot — it demonstrates the Solver learned *ML principles*, not *PyTorch syntax*. Theme 3.1 (World Modeling) transforms from fuzzy-fit to defensible claim.

**What it proves**:
- Bugs are conceptually framework-agnostic (LR too high is LR too high in any framework)
- Solver's learned representations are framework-invariant
- Your environment generalizes to real AI production where teams use multiple frameworks

**Action (Claude Code implementation)**:
1. **Create `validation/cross_framework/`**:
   - `jax_bugs/` — 3 hand-written JAX training scripts with one bug each (LR too high, vanishing gradients, data leakage — same 3 fault families, different framework)
   - `tf_bugs/` — 3 hand-written TensorFlow/Keras training scripts, same pattern
   - `run_jax_bug.py` — mini-training oracle that executes the JAX script and returns metrics dict in the same schema as PyTorch oracle
   - `run_tf_bug.py` — same for TF
2. **Update `ml_training_debugger/graders.py`**:
   - Accept framework-agnostic metrics dict (not PyTorch-specific objects)
   - Normalize to the existing `DiagnosisGrader` interface
3. **Create `scripts/eval_cross_framework.py`**:
   - Load trained Solver checkpoint
   - Run against JAX and TF bugs zero-shot (no fine-tuning on these)
   - Record diagnosis accuracy, inspection count, diagnosis correctness
4. **Add new test `tests/test_cross_framework.py`**:
   - Smoke test that JAX and TF oracles produce valid metrics dicts
5. **Pitch slide**: "Zero-shot cross-framework transfer" table

**Pitch slide template**:
```
| Framework | Solver trained on | Accuracy | Zero-shot? |
|-----------|-------------------|----------|------------|
| PyTorch (in-distribution) | PyTorch | X% | No — training distribution |
| JAX (held-out framework) | PyTorch only | Y% | YES |
| TensorFlow (held-out framework) | PyTorch only | Z% | YES |
```

If Y and Z are non-trivial (>30%), this is a **killer slide**. If they're ~0%, drop the slide — the framework-transfer hypothesis didn't hold, but Solver still succeeds in-distribution.

**Theme alignment upgrade**:
- **Theme 3.1 (World Modeling — Professional Tasks)**: from "fuzzy fit" to "defensible claim." Rebuttal against L6 becomes real.

**Scope**: **4–6 hours**. Budget Day 3 evening or Day 4 morning. Required deps: `jax`, `jaxlib`, `tensorflow` (pin CPU-only versions to avoid GPU conflict with training). Install via `uv add jax jaxlib tensorflow-cpu`.

**Gate**: only ship if §15.5 §core §Move 11 tracks are on schedule by end of Day 3. Cross-framework is a credibility multiplier but it's not a core deliverable — drop fast if time pressure hits.

**Risk**: 50% probability of non-trivial transfer (Y, Z > 30%). If it fails, framework-invariance claim retracts; pitch slot regains to Move 12/14.

**Expected impact if landed**: **+4 Innovation (Theme 3.1 defensible), +1 Storytelling = +5 total.**

---

### Moves 11–17 combined expected impact

| Move | Innovation | Storytelling | Reward Curves | Pipeline | Sub-theme | Total |
|---|---|---|---|---|---|---|
| Move 11 (code-mutation Generator) | +5 | +2 | +0 | +0 | — | **+7** |
| Move 12 (empirical finding) | +3 | +2 | +0 | +0 | — | **+5** |
| Move 13 (cost column) | +0 | +1 | +2 | +0 | — | **+3** |
| Move 14 (emergent-behavior moment) | +3 | +5 | +0 | +0 | — | **+8** |
| Move 15 (multi-Generator tournament) | +5 | +2 | +0 | +0 | +Halluminate | **+7** |
| Move 16 (rename to BugForge) | +0 | +2 | +0 | +0 | — | **+2** |
| Move 17 (cross-framework transfer) | +4 | +1 | +0 | +0 | — | **+5** |
| **Sum (all moves land)** | **+20** | **+15** | **+2** | **0** | **+1** | **+37** |

**Realistic landing rate** (no single plan ships all 7 upgrades):
- Move 13 + Move 16: ~95% land (low cost, low risk) → +5
- Move 12 + Move 14: ~80% land (instrument + extract, mostly logging) → +13
- Move 11: ~55% lands (Day-2 smoke test) → expected +4
- Move 15: ~50% lands (gated on Move 11 + Day 2 schedule) → expected +3.5
- Move 17: ~60% lands (depends on framework-transfer hypothesis) → expected +3
- **Expected total uplift: ~+28 points**

Baseline (structured Generator + Moves 1–10) = 78/100.
With Moves 11–17 expected landing = **78 + 28 ≈ 93–95/100** (realistic average), **up to 100/100 if all land cleanly**.

**This is genuine #1 territory — not just by plan math but by addressing the three structural weaknesses identified in judge-mode scoring**:
1. L1 (self-play vs curriculum) → Move 11
2. Theme breadth → Move 15 (Theme 1 primary) + Move 17 (Theme 3.1 defensible)
3. Pitch memorability → Move 14 (emergent moment) + Move 16 (rename)

**Gate cascade**:
- Move 11 smoke test Day 2 evening → gates Moves 11, 15
- Day 2 schedule held → gates Move 15 specifically
- §15.5 core tracks on schedule Day 3 end → gates Move 17
- Episodes logged from Day 3 → gates Move 14 extraction
- Moves 13, 16 unconditional (ship regardless)

---

## 8. 5-Day Execution Plan

### Day 1 — Monday 04-20 (verification + design lock, minimal code)
- **FIRST 3–4 hrs**: complete §15.5 Pre-Day-1 Verification Checklist (R1 audit, OpenEnv version pin, base-model lock, GRPO smoke test, episode wall-clock measurement, frontier-LLM API smoke)
- **CRITICAL**: run 50-episode convergence sanity check on personal GPU with Qwen2.5-3B + GRPO. If training signal is random, pivot plan before Day 2.
- Finalize Generator action space and reward design (values for `λ₁, λ₂, λ₃` locked)
- Design Colab TRL training pipeline architecture (Qwen2.5-7B + LoRA + GRPO)
- Day-1 build-in-public X post (one of the 2 total posts — "help me red-team this")
- **Deliverable**: §15.5 checklist green OR plan adjusted; locked problem statement

### Day 2 — Tuesday 04-21 (build structured Generator + code-mutation smoke test)

**Morning (structured Generator — Version B foundation, must land)**
- Implement structured Generator agent class (`fault_type × hyperparameters × difficulty`)
- Add self-play game loop: Generator turn → validation → Solver turn → reward both
- Wire into existing OpenEnv server as second agent role
- Smoke test: random Generator vs existing Solver

**Afternoon (Move 11 code-mutation smoke test — §15.5 4-hour timebox)**
- Task 1: 20 hand-written seed diffs (60 min)
- Task 2: 5-stage validation pipeline (60 min)
- Task 3: Qwen2.5-3B few-shot prototype (90 min)
- Task 4: GREEN/YELLOW/RED decision (30 min)
- **Hard stop at 7pm**. No overrun. Decision logged in plan file.

**Deliverables**:
- Structured self-play loop end-to-end (unconditional)
- Move 11 decision: GREEN / YELLOW / RED (gates all Day 3–5 Move 11 work)

### Day 3 — Wednesday 04-22 (Move 15 tournament + baselines + human bugs + instrumentation)

**Morning (Move 15 Multi-Generator Tournament — if gated ON)**
- Create `ml_training_debugger/generators/` package (base.py, greedy.py, diverse.py, specialist.py)
- Create `ml_training_debugger/tournament.py` with EloLadder
- Update `server/environment.py` with `/tournament/step` and `/tournament/elo` endpoints
- Update `server/dashboard.html` with Elo-ladder panel
- Unit tests in `tests/test_tournament.py`
- **Deliverable**: 10-episode smoke tournament runs end-to-end

**Afternoon (parallel tracks — API-bound and dev-bound don't compete)**
- **Track A (API-bound, runs in background)**: Frontier-LLM baseline run on ~30 bugs via GPT-4-Turbo, Claude 3.5 Sonnet, Llama-3.3-70B. Record accuracy + API cost + latency to `logs/frontier_baseline.jsonl`.
- **Track B (dev-bound)**: Mode A / Mode B / Mode C evaluation runners in `ml_training_debugger/eval_modes.py`
- **Track B**: Stack Overflow + GitHub Issues scraper for Mode C human bugs — target n≥30 into `data/pytorchbugbench-v0/`
- **Track B**: Bootstrap 95% CI utility in `ml_training_debugger/stats.py`
- **Move 12 instrumentation**: log per-episode token counts, Solver confidence (logits), Generator output embeddings to `logs/emergent/episodes.jsonl` (sentence-transformers MiniLM)

**Evening (if Move 11 GREEN — code-mutation Generator fine-tuning kickoff)**
- Start Qwen2.5-3B + LoRA fine-tuning in background on personal GPU (run overnight)

**Evening (if Move 17 gated ON)**
- Begin cross-framework oracle stubs: `validation/cross_framework/run_jax_bug.py`, `run_tf_bug.py`
- Install deps: `uv add jax jaxlib tensorflow-cpu`
- Write 3 JAX + 3 TF hand-authored bug scripts in `validation/cross_framework/{jax_bugs,tf_bugs}/`

**Deliverables**: tournament live + Elo ladder populating + ablation rows filled incl. cost columns + scraper at n≥30 + cross-framework oracle stubs

### Day 4 — Thursday 04-23 (TRL training + cross-framework + demo assets + fallback insurance)

**Morning**
- Minimal Unsloth/TRL Colab script (GRPO + Qwen2.5-7B + LoRA) → `scripts/train_grpo_colab.ipynb`
- Record structured-Generator self-play training curves (short run)
- **If Move 15 landed**: run longer tournament (200+ episodes) on personal GPU; Elo ladder matures

**Afternoon**
- **If Move 17 gated ON**: finish cross-framework eval `scripts/eval_cross_framework.py`; run zero-shot JAX + TF Solver eval; record to `logs/cross_framework_eval.json`
- **If Move 11 GREEN**: code-mutation Generator training continues; extract sample diffs to `seeds/generated_star_diffs/`
- Produce split-screen GIF: Generator inventing bug / Solver diagnosing → `docs/assets/demo.gif`
- **Pre-record 60-sec fallback demo MP4** → `docs/assets/demo_fallback.mp4` (non-negotiable insurance for pitch day)
- Commit pre-trained checkpoint(s) to `checkpoints/` (demo insurance #2)
- **Cache 100+ valid generated diffs** in `seeds/validated_diffs_cache.jsonl` (if Move 11 GREEN) — demo insurance #3
- Day-4 build-in-public X post (demo GIF + results teaser)

**Code-freeze cutoff**: End of Day 4 at 10pm. No code changes on Day 5 except pitch materials.

**Deliverables**: working Colab + pitch GIF + fallback MP4 + all checkpoints committed + valid-diff cache (if Move 11) + cross-framework eval results (if Move 17) + mature Elo ladder chart (if Move 15)

### Day 5 — Friday 04-24 (analysis + polish + pitch prep; NO new feature code)

**Morning**
- **Move 12 analysis**: run the 4 candidate empirical-finding charts in `notebooks/empirical_finding.ipynb`; pick strongest signal; produce 1 polished chart → `docs/assets/empirical_finding.png` + 1 quotable sentence
- **Move 14 extraction**: run `scripts/extract_emergent_moment.py`; if a moment materializes, produce Slide 3 chart → `docs/assets/emergent_moment.png` with caption
- Compute bootstrap CIs for all Mode C scores
- Decide Version A vs Version B pitch script (based on Day-2 smoke + Day-4 training outcome + Move 14 moment availability)

**Afternoon**
- **Move 16 rename execution**: run `scripts/rename_to_bugforge.sh` (see §18 execution guide)
- Mini-blog on HuggingFace Hub (<2 min read) — satisfies minimum requirement; skip YouTube
- Publish **BugForge/bench-v0** (née PyTorchBugBench-v0) to HuggingFace Datasets (minimal README + eval script)
- Final Q&A rebuttal card printed (L1, L4, L7 + Move-11/14/15/17-specific responses)
- Final slide deck: 8 slides max (hook, problem, env, Elo-ladder [if M15], results+cost, empirical finding [M12], emergent moment [M14], cross-framework [M17], close)

**Evening**
- Pitch rehearsal: 20+ runs, target 2:50 (buffer 10 sec); also rehearse 2:30 short version
- Fallback artifacts verified on presentation hardware (laptop, projector, offline mode)

**Deliverables**: Version A or B pitch locked; slide deck final; fallback artifacts tested; BugForge rename applied

### On-site Day 1 — Saturday 04-25 (real training)
- Use HuggingFace compute credits for full co-training run
- Target 1000+ self-play episodes
- Log full curves
- Finalize ablation table with real numbers
- Test live demo on on-site network

### On-site Day 2 — Sunday 04-26 (pitch day)
- Final pitch rehearsal morning
- Live demo dry-run on presentation hardware
- Pitch. Q&A with numbers, not adjectives.

---

## 9. The 3-Minute Pitch Script (Draft)

**Two script versions**: A (Move 11 GREEN — code-mutation Generator shipped) and B (Move 11 dropped — structured Generator only). Version A is the target. Version B is the insurance.

### Version A (code-mutation Generator shipped — target pitch)

#### 0:00–0:15 — The hook
> "Most ML debuggers are trained on fixed bug datasets. Ours writes its own PyTorch bugs. Watch."
> [Live demo starts: Generator writes a diff on screen → patch applies → training crashes → Solver inspects → diagnoses]

#### 0:15–0:45 — The problem + contribution
> "LLMs fail at ML debugging because the bug distribution is infinite and evolving. Static benchmarks saturate. We built a self-play curriculum where an LLM Generator writes arbitrary PyTorch code mutations, validates them via real training runs, and a Solver learns to diagnose. The novel contribution is a **verifiable reward oracle** — bugs only count if the training actually breaks."

#### 0:45–1:30 — The environment
> "Generator produces a unified diff. Five-stage validator checks it applies, parses, runs, breaks training, and isn't trivially solvable. Solver inspects gradients, weights, data. Both agents train via GRPO. PyTorch-native. OpenEnv compliant."
> [Show agent topology + reward equations + one live diff example on screen]

#### 1:30–2:15 — The results
> "Self-play Solver outperforms random-Generator-trained by X% on held-out human bugs. Frontier GPT-4 scores higher on accuracy — but our specialized 7B model costs 50–100× less per correct diagnosis. For CI/CD-integrated debugging, specialization dominates."
> [Show ablation table with cost column + two-curves chart]

#### 2:15–2:45 — The surprising finding (Move 12)
> "One specific result we didn't expect: [PICK THE STRONGEST OF THE 4 CANDIDATES AFTER DAY 3 DATA]. For example: 'Solver confidence calibration improves 3× after self-play, measured by Expected Calibration Error on unseen bugs.'"
> [Show the one chart that supports this finding]

**Why this slot matters**: this is the quotable moment judges repeat in deliberation. One sentence, one chart.

#### 2:45–3:00 — The close
> "Everything runs in PyTorch. OpenEnv compliant. Reproducible with seed=42. Qwen2.5-7B + GRPO + LoRA. PyTorchBugBench-v0 shipped as a public HuggingFace Dataset. The Generator writes real diffs — come try to stump it."
> [Show HF Space URL + GitHub URL + blog URL + Dataset URL]

### Version B (structured Generator only — insurance pitch)

Identical to Version A except:
- **0:00–0:15 hook**: "Most ML debuggers train on fixed bug datasets. Ours trains itself against an adaptive Generator. Watch."
- **0:15–0:45 scope disclosure** (the original language): *"Phase 1 is structured fault sampling across seven fault families with novelty-weighted rewards. Phase 2, on our roadmap, is an LLM-based code-mutation Generator."*
- **Close**: remove "Generator writes real diffs" line. Keep roadmap framing.

**Why preserve both**: you decide which to pitch on Day 5 morning based on §15.5 Day-2 smoke-test outcome. Version A is target; Version B is the committed fallback.

### Q&A ammunition (prepared numbers + rebuttals)

**Numbers ready on a single reference card:**
- Bug validity rate: X% of generated bugs actually break training
- Solver generalization gap: X% on seen vs Y% on held-out
- PyTorchBugBench-v0 score: self-play Solver scores X vs Y for baseline vs Z for frontier-LLM zero-shot
- Compute used: X GPU-hours, Y self-play episodes
- Elo separation: Generator advanced X points during training
- Bug novelty score: mean cosine distance of generated vs original = X (where X > 0 = genuinely new)

**Pre-canned rebuttals to likely Q&A attacks:**
- *"Why not just prompt Claude/GPT-4?"* → Show frontier-LLM row + cost column (Move 13). *"50–100× cheaper per correct diagnosis; specialized vertical models win for CI/CD integration where you debug thousands of runs per day."*
- *"Aren't generated bugs just permutations of the 7?"* → (Version A) "Generator writes arbitrary unified diffs — here's one touching a code path none of our seed bugs touch." (Version B) Show novelty metric + UMAP coverage plot.
- *"Isn't this AlphaGo with extra steps?"* → Yes, and that's the point. Cite AlphaGo self-play + DeepSeek-R1 RL + Constitutional AI as precedent — self-play for reasoning is validated. Novel contribution: **first self-play environment with a verifiable reward oracle from real PyTorch training runs** — most self-play domains have synthetic rewards; ours uses the ground truth that the code actually breaks.
- *"What stops Generator from gaming the reward with lr=1e10?"* → Validation gate + novelty term (λ₂) + diff-minimality term (λ₄, Version A only) + discretized hyperparameter bins (Version B).
- *"Does Solver forget original tasks after self-play?"* → Mode A regression guard; show <5% delta.
- *"n=30 on Mode C is tiny — are these results significant?"* → "Correct — we report 95% bootstrap CIs. Some gaps are statistically significant, some aren't. We shipped PyTorchBugBench-v0 as a public dataset so the community can expand n and reproduce."
- *"Your 'live demo' — is this live RL training?"* → "No, this is live inference of a pre-trained self-play agent. RL training takes hours; we ran it on-site Saturday. Here are the logs." [Point to committed training run.]
- *(Version A only)* *"How many valid diffs does the Generator actually produce?"* → Cite validity rate from Day-2 smoke + training run. "~X% valid on early episodes, rising to ~Y% after fine-tuning. Invalid diffs receive a λ₃ penalty."

### Related work slide (prevents "this has been done" attack)
One small slide, 3 bullets:
- **AlphaGo / AlphaZero** (2016–2017) — self-play in game environments
- **Constitutional AI / RLAIF** (Anthropic 2022) — adversarial self-critique for alignment
- **DeepSeek-R1** (2025) — RL self-improvement for reasoning

Our contribution: *first self-play environment for ML-debugging with verifiable rewards from real PyTorch training runs.*

---

## 10. Submission Artifacts Required

### Minimum requirements (from brief)
- [ ] Uses OpenEnv latest release
- [ ] Minimal training script in Colab using Unsloth or HF TRL
- [ ] Mini-blog on HuggingFace **OR** mini-video on YouTube (<2 min)

### Pitch-day artifacts
- [ ] 3-min pitch deck
- [ ] Live demo script (tested on on-site hardware)
- [ ] Ablation table slide
- [ ] Two-curves chart slide
- [ ] Agent topology diagram slide
- [ ] Reproducibility one-liner

### Public artifacts (pre-pitch)
- [ ] Updated GitHub repo
- [ ] Updated HuggingFace Space
- [ ] HuggingFace blog post (red-team narrative)
- [ ] 2-min YouTube demo video
- [ ] **PyTorchBugBench-v0** published to HuggingFace Datasets with README + eval script
- [ ] 5 saved Generator↔Solver interaction transcripts (top 3 featured in blog)
- [ ] Frontier-LLM baseline scores (GPT-4 / Claude / Llama) recorded and cited **with cost-per-dx + latency columns** (Move 13)
- [ ] Daily build-in-public X thread (Days 1–5) tagging `@Meta @huggingface` + relevant sponsor
- [ ] Social post on X tagging relevant researchers (cheap social proof)
- [ ] **Bootstrap 95% CIs computed for all Mode C scores** (1000 resamples)
- [ ] **Move 12 empirical finding chart** — 1 slide-ready chart + 1 quotable sentence

### Conditional artifacts (only if Move 11 GREEN)
- [ ] **20 hand-written seed diffs** committed to repo under `seeds/` with validation metadata
- [ ] **5-stage diff validation pipeline** implemented and unit-tested
- [ ] **Code-mutation Generator checkpoint** (Qwen2.5-3B + LoRA adapter) committed
- [ ] **100+ cached valid generated diffs** committed as demo insurance (pitch works even if Generator mid-pitch fails)
- [ ] **Diff-validity-rate curve** logged (early vs late training episodes)
- [ ] **1 "star diff" saved** — the most interesting Generator output, featured on slide 3 and in blog

### Contingency / fallback artifacts (DEMO INSURANCE)
- [ ] **Pre-recorded 60-sec demo MP4** embedded in deck (plays if live demo fails)
- [ ] **Pre-trained Solver checkpoint** committed to repo (plays if on-site training diverges)
- [ ] **Offline-mode demo script** (plays if on-site WiFi fails — no HF API calls required)
- [ ] **Heuristic-Generator fallback** (plays if TRL PPO doesn't converge in time — still shows curves from pre-training)

---

## 11. What NOT to Do (Scope Creep Traps + Descoped Items)

### Hard NOs (never cross these lines)
- ❌ Add more bug tasks beyond the existing 7
- ❌ Add a third trained agent
- ❌ Write more tests (251 is plenty; nobody counts tests in Round 2)
- ❌ Optimize Docker further (885MB is fine)
- ❌ Add more model architectures (CNN + MLP is enough)
- ❌ Build new endpoints unrelated to self-play
- ❌ Retrofit Theme 2 (long-horizon) — you're not long-horizon, don't claim it
- ❌ Spend Day 5 on code instead of pitch rehearsal
- ❌ Rebuild the PyTorch engine or graders — they work, leave them alone
- ❌ Add a UI beyond what exists — dashboard is sufficient

### Descoped from ambition list (solo/small-team reality check)
Solo execution (per session summary) against ~60+ hrs of work → cut aggressively. If team ≥2 people, reconsider re-adding only the starred (★) items.

| Originally planned | Status | Reason |
|---|---|---|
| Frozen Oversight layer (Fleet AI + Halluminate bonus) | **CUT** (★ re-add if team ≥2) | ~4 hrs of work but fragile under time pressure; bonus prizes are sponsor-judged separately so main-prize ROI is low |
| Daily build-in-public X threads (Days 1–5) | **CUT to 2 total posts** | Content cost = 1–2 hrs/day, should go to code. Post once at Day 1 (ask for bugs) and once at Day 4 (demo GIF). |
| 2-min YouTube video | **CUT — blog only** | Brief allows blog OR video (<2 min). Video = 6–12 hrs quality production. Blog = 3–4 hrs. Choose blog. |
| LLM-based code-mutation Generator (Phase 2) | **KEEP as roadmap-only slide** | Do NOT implement. Show on final pitch slide as "where this goes next." Credibility without build cost. |
| Frontier-LLM baseline (GPT-4/Claude/Llama) | **KEEP — non-negotiable** | $20–50 API spend, 4 hrs of work, kills the #1 Q&A attack. Worth every minute. |
| Human bug collection (Mode C) | **KEEP but add scraper backup** | X/Discord posts + Stack Overflow/GitHub Issues scrape to reach n≥30 |
| PyTorchBugBench-v0 as HF Dataset | **KEEP but scope minimal** | Bare-bones README + eval script. 1 hr of work. Legacy framing > polish. |

**Revised total estimated effort**: ~42 hrs over 5 days = 8.4 hrs/day. Feasible solo, provided nothing goes sideways.

---

## 11.5 Risk Register (What Goes Wrong + How We Recover)

| Risk | Probability | Impact | Mitigation / Fallback |
|---|---|---|---|
| **Live demo fails at pitch (WiFi/GPU)** | 25% | Catastrophic | Pre-recorded 60-sec MP4 embedded in deck; offline-mode demo script |
| **TRL PPO doesn't converge in on-site 2 days** | 35% | High | Pre-trained checkpoint shipped in repo; heuristic-Generator fallback with pre-recorded curves |
| **Generator mode-collapses to lr_too_high spam** | 40% | High | Novelty term `λ₂`; discretized hyperparameter bins; diversity-aware sampling buffer |
| **Bug validity rate <50%** | 30% | Medium | Manual curation pass Day 3; raise `λ₃` penalty; shrink initial Generator action space |
| **Catastrophic forgetting on Mode A** | 25% | High | Replay buffer mixing original 7 tasks into self-play; Mode A regression guard blocks release |
| **Human bug collection yields <5 bugs** | 50% | Medium | Supplement with Stack Overflow scraped bugs; recruit from university ML labs; lower N to 3 with quality framing |
| **Frontier LLM beats self-play on Mode C** | 40% | Medium | Pivot narrative to "competitive at 1/100th cost" — still a win story |
| **Compute credits exhausted before 1000 episodes** | 30% | Medium | Pre-train on personal GPU Days 1–4; reserve on-site compute for final run only |
| **Pitch runs over 3 min** | 50% | High | 20+ rehearsals target 2:50; have a 2:30 version ready to cut 20 sec if needed |
| **Q&A "why not just prompt Claude?"** | 95% | Depends on prep | Frontier LLM row in ablation table + cost column (Move 13) kills this in 10 sec |
| **Code-mutation Generator (Move 11) smoke test fails Day 2** | 45% | Medium | Drop Move 11, revert to structured-only Generator. Moves 12 + 13 still land. Target shifts from 95→88/100. Plan explicitly conditional; no sunk cost. |
| **Code-mutation Generator ships but 90% invalid-diff rate** | 60% (given Move 11) | Low | Few-shot seed cache of 20 hand-written diffs covers demo even if fine-tuning never converges. Generator is demoable as "LLM writes bug" without RL. |
| **Empirical finding (Move 12) shows no signal** | 40% | Low | 4 candidate findings prioritized; inference-compute scaling (row 4) has ~75% materialize probability — floor guarantee. |
| **Bootstrap CIs overlap — no statistical significance** | 40% | Low | Acknowledge openly in pitch: "accuracy gap not significant at n=30; this is why we shipped PyTorchBugBench-v0 for community expansion." Honesty = trust. |
| **Move 11 + Move 12 both succeed but eat Day 5 pitch prep** | 25% | High | Hard cutoff: Move 11 code freeze end of Day 3; Move 12 analysis end of Day 4 morning. Pitch rehearsal is Day 5 non-negotiable. |

**Principle**: every high-probability failure mode must have a fallback artifact committed to the repo **by end of Day 4**. No "hope it works on the day" items. Conditional upgrades (Moves 11–13) each have explicit drop-criteria so they cannot cascade into scope creep.

---

## 12. Judging Criteria Alignment

### Targets by pitch version

**Version A (Move 11 GREEN — code-mutation Generator shipped)**

| Criterion | Weight | Target | Harsh judge floor | What earns the top marks |
|---|---|---|---|---|
| Environment Innovation | 40% | **37/40** | 33/40 | Code-mutation Generator + verifiable reward oracle + recursive curriculum + novel bug-validity gate |
| Storytelling | 30% | **28/30** | 26/30 | Live diff generation + empirical finding + cost-efficiency framing + pre-canned Q&A + interaction transcripts |
| Reward Curves Improvement | 20% | **18/20** | 16/20 | Two-curves chart + ablation table + frontier-LLM baseline with cost column + bootstrap CIs + Mode A regression |
| Training Pipeline Setup | 10% | **9/10** | 9/10 | TRL + GRPO + LoRA + Colab + reproducibility + PyTorchBugBench-v0 eval script |
| **Version A total** | **100%** | **92/100** | **84/100** | #1 territory |

**Version B (Move 11 dropped — structured Generator fallback)**

| Criterion | Weight | Target | Harsh judge floor |
|---|---|---|---|
| Environment Innovation | 40% | 32/40 | 28/40 |
| Storytelling | 30% | 27/30 | 25/30 |
| Reward Curves Improvement | 20% | 17/20 | 15/20 |
| Training Pipeline Setup | 10% | 9/10 | 8/10 |
| **Version B total** | **100%** | **85/100** | **76/100** | Top 3 territory |

### What this means
- Version A target (92/100) is #1 territory against realistic judge variance.
- Version B target (85/100) is Top 3 territory — strong finalist, not clear #1.
- The 7-point delta between versions is **almost entirely from Move 11** (code-mutation Generator killing L1).
- Version B harsh-judge floor (76/100) is still Top 6 finalist.
- **The plan is designed to never fall below Top 6 while preserving a clear #1 path conditional on Move 11.**

---

## 13. Probability Estimates

### Baseline ladder (structured Generator only — no conditional upgrades)

| Version | P(Top 6) | P(Top 3) | P(#1) |
|---|---|---|---|
| Continue as-is (Round 1 submission) | 45% | 15% | 3% |
| Pivot baseline (Candidate B only) | 60% | 28% | 14% |
| + Held-out human bug test (n=5–10) | 66% | 32% | 18% |
| + Ablation table | 70% | 36% | 20% |
| + Live pitch demo (with fallback video) | 76% | 44% | 25% |
| + Red-team narrative framing | 79% | 48% | 27% |
| + Frontier-LLM baseline (Move 7) | 83% | 55% | 31% |
| + Formal reward math + novelty metric + UMAP | 85% | 58% | 33% |
| + Risk register with committed fallbacks | 86% | 62% | 35% |
| + PyTorchBugBench-v0 + transcripts + n≥30 bugs | 88% | 66% | 38% |
| + Perfect onsite execution + good pitch slot | 90% | 70% | **42%** |

### With conditional upgrades (Moves 11–13)

| Upgrade configuration | P(Top 6) | P(Top 3) | P(#1) |
|---|---|---|---|
| + Move 13 (cost-per-dx column — nearly free) | 91% | 72% | **44%** |
| + Move 12 (empirical finding — floor-row guaranteed) | 92% | 74% | **47%** |
| + Move 11 (code-mutation Generator — if smoke passes) | **94%** | **78%** | **52%** |
| All 3 upgrades + perfect execution + good pitch slot | 95% | 80% | **~55%** |

### Conditional probability decomposition

P(#1) = P(Move 11 ships) × P(#1 | Move 11 ships) + P(Move 11 dropped) × P(#1 | Move 11 dropped)
- P(Move 11 smoke test passes) ≈ 55%
- P(#1 | Move 11 ships) ≈ 52% (targets 95/100 submission)
- P(#1 | Move 11 dropped but 12+13 land) ≈ 42% (targets 88/100 submission)
- **Expected P(#1) = 0.55 × 0.52 + 0.45 × 0.42 = 0.286 + 0.189 = ~48%**

### Honest ceiling

**~48% P(#1) expected value, ~55% ceiling with perfect execution of all upgrades.**

The earlier "57%" number was optimistic pre-limitations-audit. The 38–42% range was the conservative estimate *without* conditional upgrades. **With Moves 11–13 budgeted and conditionally gated, the honest ceiling moves back up to ~55% — not because the limitations disappeared, but because Move 11 specifically addresses L1 (the biggest ceiling driver).**

Accounted variance sources:
- L1 (mitigated by Move 11 if it lands, otherwise still binding)
- L4 (mitigated by Move 13 cost narrative)
- L5 (partial mitigation via Move 12 emergent-finding framing)
- L8 (partial mitigation via bootstrap CIs)
- L10 (hackathon judge variance is ±15% regardless of plan — cannot be planned away)

**What this means**: Top 6 is near-certain (~92%). Top 3 is highly likely (~74%). #1 is realistically a coin flip — about 50/50 if all upgrades land, ~40% if Move 11 is dropped. Plan for Top 3 floor, build for #1 ceiling. Anyone quoting >60% P(#1) without having shipped Move 11 in a smoke test is still lying.

---

## 14. Success Criteria (How We Know It Worked)

### Floor success (Version B shipped, structured Generator only)
- Top 6 finalist (automated validation passes + judges engaged)
- At least 1 sub-theme bonus prize (Snorkel — locked via framing)
- ~76–85/100 projected score
- ~91% probability achieved

### Target success (Version B + Moves 12 + 13 land cleanly)
- Top 3 finish
- Snorkel + Mercor sub-theme bonuses (both zero-engineering-cost)
- ~85–88/100 projected score
- PyTorchBugBench-v0 cited as community contribution
- ~66–74% probability achieved

### Stretch success — #1 target (Version A shipped — Move 11 GREEN + Moves 12 + 13)
- **#1 overall** + Snorkel + Mercor sub-theme bonuses (Fleet AI possible if Oversight re-added post-hackathon)
- ~92–95/100 projected score
- ~45–55% probability achieved
- Live demo of Generator writing a novel PyTorch bug-diff lands cleanly
- Move 12 empirical finding slide produces a quotable sentence judges repeat
- Meta/HF engineer asks "can we hire you" or "can we use this"
- HF Space goes viral post-hackathon; PyTorchBugBench-v0 gets external PRs
- **Phase 3 roadmap slide (multi-architecture, cross-framework) generates follow-up interest**

### Meta-success (regardless of placement)
- **Learned whether Move 11 actually works** — this is genuinely novel research direction whose answer matters beyond the hackathon
- **Shipped PyTorchBugBench-v0 as public benchmark** — legacy value independent of placement
- **Trained a 7B specialist model competitive with frontier on a narrow vertical** — resume/portfolio value regardless of score

---

## 15. Immediate Next Actions (Do Today)

**Do these in exact order. Blockers first — §15.5 audits before any code.**

1. **Complete §15.5 Pre-Day-1 Verification Checklist** — R1 audit + OpenEnv version pin + base-model lock + convergence sanity check. If any blocker fails, stop and adjust plan before continuing.
2. **Lock in this plan** — no more strategic pivoting from here once §15.5 passes
3. **Post on X/Discord/HF** (one of 2 total build-in-public posts): "I'm building a PyTorch self-play ML debugger for the OpenEnv hackathon. Help me red-team it — submit a PyTorch bug I've never seen and I'll credit you in my HF blog."
4. **Reserve compute for frontier-LLM baseline** — smoke test 1 bug through GPT-4 / Claude / Llama APIs on Day 2; full run Day 3. Budget ~$30–50. Non-negotiable: this kills the #1 Q&A attack.
5. **Create Generator agent skeleton** in `ml_training_debugger/generator.py`
6. **Create PyTorchBugBench-v0 HF Dataset stub** — 15 min; unlocks the benchmark-legacy framing
7. **Write pitched problem statement** for submission form (Theme 4 primary, Theme 3.1 secondary)
8. **Schedule 20+ pitch rehearsals** across Days 4–6
9. **Audit Round 1 submission language for theme alignment** — DQ risk mitigation (part of §15.5 blocker list)
10. **Pre-draft 20 seed diffs for Move 11 smoke test** — start collecting PyTorch bug examples today even before Day 2. Sources: Stack Overflow PyTorch tag, GitHub issues on popular training repos, your own 7 fault families translated to minimal diffs. Having these pre-drafted means Day-2 smoke test starts with Task 2, not Task 1 — buys back ~1 hour.

---

## 15.5 Pre-Day-1 Verification Checklist (DO BEFORE WRITING ANY CODE)

These are **blocker audits**. If any fail, the plan changes before Day 1 begins. Spend the first 3–4 hours of Day 1 on these only.

### Critical blockers (must pass before Generator implementation)

- [ ] **R1 theme-alignment audit**
  Read your Round 1 submission today. Does its problem statement language explicitly match Theme 4 (Self-Improvement) or Theme 3.1 (Professional World Modeling)? If not, rewrite R2 framing to present Round 2 as a *"natural extension of R1's existing self-improvement dimension"* — not a pivot. This is DQ-risk #1.
  **Blocker if fail**: requires re-framing session before implementation.

- [ ] **OpenEnv latest-release compatibility check**
  Pin the exact OpenEnv version (e.g., `openenv==0.X.Y`). Run existing ML Debugger against it. Fix any breaking API changes.
  **Blocker if fail**: could eat 1–2 days.

- [ ] **Base-model decision locked**
  - **Solver base model**: Qwen2.5-7B-Instruct (default recommendation — strong PyTorch knowledge, TRL-compatible, fits on single A100)
  - **Generator base model**: Same Qwen2.5-7B-Instruct (shared weights, LoRA fine-tuning with separate adapters)
  - **Rationale**: Qwen2.5 has better PyTorch knowledge than Llama-3-8B per recent code-gen benchmarks; TRL + PEFT + Unsloth supports this directly.
  - **Fallback if compute insufficient**: Qwen2.5-3B-Instruct (degraded but feasible on free HF credits)

- [ ] **TRL algorithm locked: GRPO (preferred) with PPO fallback**
  - **GRPO** (Group Relative Policy Optimization, DeepSeek-R1): more sample-efficient than PPO, fewer hyperparameters, pairs well with verifiable rewards (your `run_real_training()` gate)
  - **PPO fallback**: if GRPO's TRL integration is unstable, revert to PPO
  - **DPO is not viable here** (needs paired preference data you don't have)

- [ ] **Convergence sanity check on personal GPU (MOST IMPORTANT BLOCKER)**
  Before Day 2 Generator build: run a 50-episode shortened-training-oracle loop (5-epoch mini-training per episode) with GRPO on Qwen2.5-3B. Verify reward signal is non-random and moving. If training doesn't move meaningfully in 50 episodes, **on-site won't converge either** — pivot to heuristic-Generator + pre-computed curves narrative.
  **Blocker if fail**: plan goes from "self-play trained" to "self-play architecture + heuristic Generator demonstration" — still defensible but weaker.

- [ ] **Episode wall-clock measured and budgeted**
  Measure: average seconds per episode = (fault injection + shortened training + Solver forward passes). Multiply by 1000 = target wall-clock. If >24 hrs, shrink training-oracle epochs further or reduce episode target to 500.

- [ ] **Frontier-LLM API budget confirmed**
  ~$30–50 total for GPT-4-Turbo + Claude 3.5 Sonnet + Llama-3-70B (via Together.ai) against ~30 bugs. Confirm payment method works; run 1-bug smoke test Day 2.

- [ ] **Team vs solo confirmed**
  Scaler typically requires team. If team: assign roles explicitly — (1) Generator + reward infra, (2) Evals + frontier LLM baselines + PyTorchBugBench, (3) Pitch + demo + blog. If solo: commit to descoped scope from §11.

- [ ] **Onsite venue network assumption: assume bad WiFi**
  Demo must run fully offline (no HF API calls at demo time). Pre-record 60-sec MP4 as primary fallback.

### Nice-to-have audits

- [ ] Unsloth + TRL + Qwen2.5 + LoRA smoke test on Colab free tier — confirms the stack works end-to-end before hackathon compute is spent
- [ ] Stack Overflow + GitHub Issues scraper script ready for Mode C bug supplement (get to n≥30)
- [ ] Rehearsed 2:30 "short version" of pitch in case judges cut time

### Day-2 code-mutation Generator smoke test (GATE for §7 Move 11)

This is a **mid-plan conditional blocker**. Run on Day 2 afternoon, after the structured Generator self-play loop is working end-to-end. Timebox: 4 hours max. No overrun. If not green by evening of Day 2, drop Move 11 and proceed with structured-only plan.

- [ ] **Task 1: Write 20 seed diffs** (~60 min)
  Hand-author 20 unified-diff patches against a clean PyTorch training script — 2–4 per fault family. Each diff must apply, parse, run, and break training. These become the few-shot pool for the Generator prompt.

- [ ] **Task 2: Validation pipeline works** (~60 min)
  Implement the 5-stage validation: (a) `git apply --check`, (b) AST parse, (c) import+run, (d) `run_real_training()` triggers failure, (e) not trivially solvable by 3-step Solver heuristic. All 20 seed diffs must pass validation.

- [ ] **Task 3: In-context Generator prototype** (~90 min)
  Qwen2.5-3B few-shot prompt: `"{20 seed diffs}\n\nNow produce a new diff that breaks training in a different way:"`. No fine-tuning. Target: produce 10 candidate diffs; at least 3 pass the full validation pipeline (30% validity rate).

- [ ] **Task 4: Pass/fail decision** (~30 min)
  **GREEN (ship Move 11)**: ≥3/10 candidate diffs pass validation AND at least 1 is "interesting" (novelty score above median of seed pool) AND Solver cannot trivially solve it.
  **RED (drop Move 11)**: <3/10 pass OR all passing diffs are degenerate (lr=1e10 type).
  **YELLOW (ship without fine-tuning)**: 3/10 pass but mode-collapses to simple mutations. Ship as "LLM-based code-mutation Generator with in-context learning" — still a novel contribution vs structured generator, skip GRPO fine-tuning on the Generator.

- [ ] **Decision logged in plan**: Day-2 Generator status = { GREEN / YELLOW / RED }. Downstream plan branches accordingly.

### Day-3 empirical-finding instrumentation (enables §7 Move 12)

Run before bed on Day 3. Does not gate any other plan item; always lands.

- [ ] Log per-episode token counts (Solver inspection chain length)
- [ ] Log Solver confidence distribution + correctness label (for ECE computation)
- [ ] Save Generator output embeddings + cluster them (for bug-family emergence detection)
- [ ] Produce 4 candidate charts on Day 5 morning; pick the one with the strongest signal for slide 4

---

## 16. Final Principle

**Stop asking if you'll win. Start executing the plan.** The ~45–55% P(#1) ceiling (with Move 11 shipped) only realizes if every day of the 5-day plan is executed cleanly. Miss a day, lose 10 points. Miss two, drop to finalist tier. The plan is not perfect — hackathons cannot be planned to perfection. The plan is *honest* and *conditionally gated*, which is the closest thing.

### Why this plan is now structurally complete
The plan went through three iterations: initial draft → brutal limitations audit (§0) → conditional upgrades (§7 Moves 11–13). What makes it structurally complete:

1. **Floor is protected**: Version B pitch + §11.5 Risk Register + fallback artifacts = ~91% P(Top 6). The plan cannot collapse completely.
2. **Ceiling is unlocked**: Move 11 is the single structural lever that addresses L1 (the biggest Innovation ceiling driver). It is conditionally gated by a 4-hour Day-2 smoke test — the plan commits to the upside only if evidence supports it.
3. **Zero-cost bonuses are taken**: Moves 12 and 13 require only instrumentation and computation — no research risk. They land whether Move 11 does or not.
4. **Every limitation is either mitigated or openly disclosed**: L1 conditionally, L4 via cost column, L5 via emergent-finding framing, L7 via honesty, L8 via bootstrap CIs. L10 (judge variance) cannot be planned away and the plan doesn't pretend otherwise.
5. **Pitch has two versions, pre-rehearsed**: Version A (target) and Version B (insurance). Decision happens Day 5 morning based on evidence, not hope.

### What "perfect" actually looks like now
The plan cannot be made structurally more perfect. The remaining quality is earned, not written:
1. **Pre-Day-1 checklist (§15.5) passes cleanly** — especially the convergence sanity check
2. **Day-2 Move 11 smoke test comes in GREEN or YELLOW** — the single highest-leverage event in the plan
3. **Generator + Solver training produces visibly improving curves** — no guarantee, must be demonstrated
4. **Frontier-LLM baseline row populated with real numbers + cost column** — not placeholders
5. **Move 12 empirical finding produces at least one quotable result** (inference-compute scaling is the floor guarantee)
6. **Live demo runs first-try at pitch** — the fallback video exists so demo failure is recoverable but ideal demo succeeds
7. **Pitch delivered in 2:50** — rehearsed 20+ times, crisp, using Version A or Version B as dictated by evidence

Execution > strategy from here. The plan has absorbed every defensible improvement a planning exercise can produce. **No further planning required. Now build.**

---

---

## 17. Updated Targets & Probabilities (with Moves 14–17)

### Judging-criteria targets

**Version A+ (Move 11 + all four new moves land — maximum)**

| Criterion | Weight | Target | Harsh-judge floor |
|---|---|---|---|
| Environment Innovation | 40% | **38/40** | 33/40 |
| Storytelling | 30% | **29/30** | 26/30 |
| Reward Curves Improvement | 20% | **18/20** | 16/20 |
| Training Pipeline Setup | 10% | **9/10** | 9/10 |
| **Total** | **100%** | **94–97/100** | **84/100** |

**Version A (Move 11 lands + 2–3 new moves land — realistic)**

| Criterion | Target |
|---|---|
| Environment Innovation | 35–37/40 |
| Storytelling | 27–29/30 |
| Reward Curves | 17–18/20 |
| Pipeline | 9/10 |
| **Total** | **88–93/100** |

**Version B (Move 11 dropped, Moves 13 + 14 + 16 land — floor)**

| Criterion | Target |
|---|---|
| Environment Innovation | 30–33/40 |
| Storytelling | 26–28/30 |
| Reward Curves | 16–17/20 |
| Pipeline | 8–9/10 |
| **Total** | **80–87/100** |

### Probability ladder (final)

| Configuration | P(Top 6) | P(Top 3) | P(#1) |
|---|---|---|---|
| Baseline plan (Moves 1–10) | 88% | 66% | 38% |
| + Moves 11–13 expected landing | 92% | 74% | 48% |
| + Move 16 (rename — trivial) | 93% | 75% | 49% |
| + Move 14 (emergent moment — if materializes) | 94% | 77% | 52% |
| + Move 17 (cross-framework — if transfer works) | 95% | 79% | 55% |
| + Move 15 (multi-Generator tournament — gated on Move 11) | **96%** | **82%** | **58%** |
| All 7 upgrades land + perfect onsite execution + good pitch slot | 97% | 85% | **~62%** |

### Expected-value calculation

P(each move lands) × P(#1 | that move lands) aggregated:
- Moves 13 + 16 (near-free): 95% land → contribute ~+4 P(#1)
- Move 12 + 14 (instrument + extract): 70% land → contribute ~+6 P(#1)
- Move 11 (smoke-test gated): 55% lands → contributes ~+8 P(#1) conditionally
- Move 15 (day-eating, gated on Move 11): 50% lands → contributes ~+4 P(#1)
- Move 17 (cross-framework): 60% lands → contributes ~+3 P(#1)

**Expected P(#1) ≈ 52–55%.** Ceiling ~62% if everything lands. Floor ~38% if only Moves 13 + 16 land.

**Realistic honest answer**: plan now targets genuine #1 territory. The remaining gap between 62% and 100% is entirely hackathon-judge variance (L10) which no plan can eliminate.

---

## 18. Claude Code Execution Guide (Operational Manifest)

**Purpose of this section**: turn the plan into a file-by-file execution manifest that Claude Code (or any engineer) can follow without interpreting strategy. Each entry includes: file path, action, acceptance criteria, and dependencies.

### Current project layout (reference)

```
ML Debugger/
├── ml_training_debugger/        # Core package (DO NOT rename — breaks 251 tests)
│   ├── client.py
│   ├── code_templates.py
│   ├── graders.py
│   ├── models.py
│   ├── nn_models.py
│   ├── pytorch_engine.py
│   ├── reward_engine.py
│   ├── scenarios.py
│   └── simulation.py
├── server/
│   ├── app.py                   # FastAPI + WebSocket server
│   ├── environment.py           # OpenEnv environment binding
│   ├── dashboard.html
│   ├── _baseline_results.py
│   └── _heuristic.py
├── tests/                       # 251 tests — keep green throughout
├── validation/
│   ├── run_all_validations.py
│   └── validate_exploding_gradients.py
├── baseline_heuristic.py
├── baseline_inference.py
├── demo.py
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── Dockerfile
└── README.md
```

**Invariants (must hold at every checkpoint)**:
- `pytest tests/` must stay green throughout. Run after every structural change.
- `ml_training_debugger/` package name stays (import stability)
- Existing 7 tasks' grader scores must not regress (Mode A regression guard from §5)

### Directories to create (run once on Day 2 morning)

```bash
mkdir -p ml_training_debugger/generators            # Move 15
mkdir -p data/pytorchbugbench-v0                    # Move 10 (→ renamed to bugforge/bench-v0 after Move 16)
mkdir -p seeds                                      # Move 11 seed diffs
mkdir -p seeds/generated_star_diffs                 # Move 11 notable outputs
mkdir -p logs/emergent                              # Move 12 + Move 14 instrumentation
mkdir -p logs/tournament                            # Move 15 Elo logs
mkdir -p logs/frontier_baseline                     # Move 7/13 frontier LLM results
mkdir -p checkpoints                                # committed pre-trained checkpoints (fallback)
mkdir -p notebooks                                  # empirical analysis
mkdir -p scripts                                    # executable helpers
mkdir -p docs/assets                                # GIFs, MP4, charts
mkdir -p validation/cross_framework/jax_bugs        # Move 17
mkdir -p validation/cross_framework/tf_bugs         # Move 17
```

### File-by-file manifest

Each row = one file Claude Code creates or edits. **"When"** = day/move dependency. **"Acceptance"** = how to verify the file is done.

#### Day 1 — verification + infrastructure

| File | Action | When | Acceptance |
|---|---|---|---|
| `ROUND_2_WINNING_PLAN.md` | Already written — DO NOT overwrite without user approval | — | — |
| `docs/R1_THEME_AUDIT.md` | NEW — written audit of Round 1 submission language vs Theme 4/3.1 alignment | Day 1 morning | User confirms DQ risk mitigated |
| `pyproject.toml` | EDIT — pin OpenEnv version; add optional deps `[project.optional-dependencies] rl = ["trl>=0.12", "peft", "unsloth", "sentence-transformers"]` | Day 1 morning | `uv sync --extra rl` succeeds |
| `scripts/smoke_grpo_convergence.py` | NEW — 50-episode GRPO smoke test on Qwen2.5-3B with heuristic Generator; writes `logs/smoke_grpo_reward.csv` | Day 1 afternoon | Reward curve visibly moves above noise floor by episode 50 |

#### Day 2 — structured Generator + Move 11 smoke test

| File | Action | When | Acceptance |
|---|---|---|---|
| `ml_training_debugger/generator.py` | NEW — `Generator` base class + `StructuredGenerator` (fault_type × hyperparameters × difficulty) | Day 2 morning | Unit test: 100 samples, all `validate()` cleanly |
| `ml_training_debugger/self_play.py` | NEW — self-play loop orchestrator: `SelfPlayEpisode(generator, solver, validator, reward_fn)` | Day 2 morning | 10-episode smoke test runs end-to-end without exceptions |
| `ml_training_debugger/reward_engine.py` | EDIT — add `compute_generator_reward(tau, s_solver, novelty, invalid)` implementing `R_gen = λ₁(1-s) + λ₂·novelty - λ₃·𝟙[invalid]` | Day 2 morning | Unit test reward math vs hand-computed values |
| `server/environment.py` | EDIT — add `generator` role + `/episode/self_play` endpoint | Day 2 morning | `curl POST /episode/self_play` returns valid episode JSON |
| `tests/test_self_play.py` | NEW — smoke tests for self-play loop, reward math, validation gate | Day 2 morning | All 10+ tests green |
| `seeds/seed_diffs.jsonl` | NEW — 20 hand-authored unified-diff patches, 2–4 per fault family | Day 2 afternoon (Move 11 Task 1) | All 20 pass validation pipeline |
| `ml_training_debugger/diff_validator.py` | NEW — 5-stage validator: apply, parse, run, break-training, non-trivial | Day 2 afternoon (Move 11 Task 2) | Unit test: 20/20 seed diffs pass, 3 pre-broken diffs fail at expected stage |
| `ml_training_debugger/code_mutation_generator.py` | NEW — Qwen2.5-3B few-shot prompt wrapper; no fine-tuning yet | Day 2 afternoon (Move 11 Task 3) | 10 candidate diffs generated; ≥3 pass validation (decision gate) |
| `docs/MOVE_11_DECISION.md` | NEW — record Day-2 smoke-test result: GREEN / YELLOW / RED + stats | Day 2 evening (Move 11 Task 4) | Decision documented; downstream plan branches set |

#### Day 3 — Move 15 tournament + frontier-LLM baselines + Move 17 stubs

| File | Action | When | Acceptance |
|---|---|---|---|
| `ml_training_debugger/generators/__init__.py` | NEW — exports Generator classes | Day 3 morning (Move 15) | `from ml_training_debugger.generators import GreedyGenerator, DiverseGenerator, SpecialistGenerator` works |
| `ml_training_debugger/generators/base.py` | NEW — `Generator` abstract class moved here | Day 3 morning (Move 15) | Existing `StructuredGenerator` subclasses it without regression |
| `ml_training_debugger/generators/greedy.py` | NEW — `GreedyGenerator` targeting highest expected solver failure | Day 3 morning (Move 15) | Produces biased distribution over hardest fault families |
| `ml_training_debugger/generators/diverse.py` | NEW — `DiverseGenerator` maximizing novelty term | Day 3 morning (Move 15) | Embedding diversity score > greedy generator |
| `ml_training_debugger/generators/specialist.py` | NEW — `SpecialistGenerator` tracking per-family solver accuracy | Day 3 morning (Move 15) | Targets lowest-accuracy family with >60% frequency after 50 episodes |
| `ml_training_debugger/tournament.py` | NEW — `Tournament`, `EloLadder` (k=32), round-robin scheduler | Day 3 morning (Move 15) | 10-episode tournament runs; Elo ratings diverge from initial 1500 |
| `tests/test_tournament.py` | NEW — Elo math unit tests + 10-episode integration test | Day 3 morning (Move 15) | All green; known Elo test vectors match |
| `server/environment.py` | EDIT — add `POST /tournament/step`, `GET /tournament/elo` | Day 3 morning (Move 15) | Endpoints return valid JSON; dashboard loads |
| `server/dashboard.html` | EDIT — add Elo-ladder panel (Chart.js line plot) | Day 3 morning (Move 15) | Visually updates during running tournament |
| `ml_training_debugger/eval_modes.py` | NEW — `ModeA`, `ModeB`, `ModeC` evaluation runners + bootstrap CI utility | Day 3 afternoon | All 3 modes return scores with 95% bootstrap CI |
| `ml_training_debugger/stats.py` | NEW — `bootstrap_ci(scores, n_resamples=1000)` helper | Day 3 afternoon | Unit test against known test vectors |
| `scripts/run_frontier_baseline.py` | NEW — runs ~30 bugs through GPT-4-Turbo, Claude 3.5 Sonnet, Llama-3.3-70B; logs accuracy, cost, latency | Day 3 afternoon (Track A) | Writes `logs/frontier_baseline.jsonl` with all 3 models × N bugs |
| `scripts/scrape_pytorchbugbench.py` | NEW — Stack Overflow + GitHub Issues scraper for human bugs | Day 3 afternoon | `data/pytorchbugbench-v0/bugs.jsonl` has n≥30 entries |
| `ml_training_debugger/emergent_logger.py` | NEW — logs per-episode (generator_output, trajectory, reward, embeddings) to `logs/emergent/episodes.jsonl` | Day 3 afternoon (Move 12+14) | 100-episode run produces valid JSONL with embeddings |
| `ml_training_debugger/simulation.py` | EDIT — wire `emergent_logger` into self-play loop | Day 3 afternoon | Existing tests still pass; logging active |
| `validation/cross_framework/jax_bugs/*.py` | NEW — 3 hand-written JAX training scripts with seeded bugs | Day 3 evening (Move 17 — if gated ON) | Each script runs; each produces expected failure mode |
| `validation/cross_framework/tf_bugs/*.py` | NEW — 3 hand-written TF/Keras training scripts with seeded bugs | Day 3 evening (Move 17) | Same as above |
| `validation/cross_framework/run_jax_bug.py` | NEW — oracle that runs JAX bugs and returns PyTorch-schema metrics | Day 3 evening (Move 17) | Metrics dict matches existing Mode A schema |
| `validation/cross_framework/run_tf_bug.py` | NEW — same for TF | Day 3 evening (Move 17) | Same as above |

#### Day 4 — training, demo assets, Move 17 eval

| File | Action | When | Acceptance |
|---|---|---|---|
| `scripts/train_grpo_colab.ipynb` | NEW — Unsloth + TRL + Qwen2.5-7B + LoRA + GRPO Colab notebook | Day 4 morning | Runs on Colab free tier; ≥50 steps of training |
| `checkpoints/solver_selfplay_v1/` | NEW — committed LoRA adapter from structured self-play | Day 4 morning | Loads via `PeftModel.from_pretrained`; passes Mode A regression check |
| `checkpoints/generators/` | NEW — if Move 15: committed LoRA adapters for G_greedy, G_diverse, G_specialist | Day 4 morning (Move 15) | All 3 adapters load; tournament replay reproduces logged Elo |
| `checkpoints/code_mutation_generator/` | NEW — if Move 11 GREEN: committed adapter + seed-diffs cache | Day 4 afternoon (Move 11) | Generator produces ≥50 valid diffs in demo |
| `seeds/validated_diffs_cache.jsonl` | NEW — 100+ cached valid diffs as demo insurance | Day 4 afternoon (Move 11) | File has ≥100 entries; random 10 all pass validator |
| `scripts/eval_cross_framework.py` | NEW — zero-shot eval of trained Solver on JAX + TF bugs | Day 4 afternoon (Move 17) | Writes `logs/cross_framework_eval.json` with accuracy per framework |
| `docs/assets/demo.gif` | NEW — split-screen GIF: Generator inventing bug / Solver diagnosing | Day 4 afternoon | <5 MB; plays in HF blog; <15s duration |
| `docs/assets/demo_fallback.mp4` | NEW — 60-sec pre-recorded pitch demo | Day 4 afternoon | Plays offline on presentation laptop; audio-free (for stage narration) |
| `inference.py` | EDIT — add `--offline` flag that uses cached generator outputs (no HF API) | Day 4 afternoon | Runs end-to-end with WiFi disconnected |
| `demo.py` | EDIT — add `--pitch-mode` that plays pre-canned scripted demo from cached diffs | Day 4 afternoon | 60-sec deterministic run using committed seed |

#### Day 5 — analysis, rename, publication, pitch

| File | Action | When | Acceptance |
|---|---|---|---|
| `notebooks/empirical_finding.ipynb` | NEW — computes ECE, token-scaling, cross-family transfer, Mode-B-vs-C gap | Day 5 morning (Move 12) | Produces 4 charts; quotable sentence extracted |
| `docs/assets/empirical_finding.png` | NEW — the one winning chart from Move 12 | Day 5 morning | Slide-ready, 1600×900, no chart junk |
| `scripts/extract_emergent_moment.py` | NEW — scans `logs/emergent/episodes.jsonl` for surprise signals | Day 5 morning (Move 14) | Prints 3 candidate moments; writes 1 chart if found |
| `docs/assets/emergent_moment.png` | NEW — if materialized: the Move 14 slide-ready chart | Day 5 morning (Move 14) | Slide-ready; 1 caption sentence provided |
| `scripts/rename_to_bugforge.sh` | NEW — automated rename script (safe; only touches user-facing strings, not package name) | Day 5 afternoon (Move 16) | Running it: pyproject.toml gets `name = "bugforge"`; README H1 changes; dashboard title changes; tests still pass |
| `bugforge/__init__.py` | NEW — alias package that re-exports `ml_training_debugger` symbols | Day 5 afternoon (Move 16) | `from bugforge import Generator` works |
| `README.md` | EDIT — H1 → "BugForge"; subheader mentions legacy name | Day 5 afternoon (Move 16) | Reads well; SEO for legacy name preserved |
| `openenv.yaml` | EDIT — env name → `bugforge-v0` | Day 5 afternoon (Move 16) | `openenv validate` (if tool exists) passes |
| `server/dashboard.html` | EDIT — title/header → "BugForge Dashboard" | Day 5 afternoon (Move 16) | Visually verified |
| `data/pytorchbugbench-v0/README.md` | NEW — dataset card; benchmark description; eval instructions | Day 5 afternoon (Move 10) | HuggingFace Datasets validates |
| `data/pytorchbugbench-v0/eval.py` | NEW — standalone eval script users can run against any model | Day 5 afternoon | Runs successfully against committed checkpoint |
| `docs/BLOG_DRAFT.md` | NEW — <2-min HuggingFace blog post | Day 5 afternoon | Under 1200 words; includes demo GIF + 1 chart |
| `docs/PITCH_DECK_v_A.md` | NEW — 8-slide outline for Version A (Move 11 GREEN) | Day 5 evening | Full pitch script + slide-by-slide visuals inventory |
| `docs/PITCH_DECK_v_B.md` | NEW — 8-slide outline for Version B (Move 11 RED) | Day 5 evening | Same structure; Move-11-dependent slides swapped |
| `docs/QA_REBUTTAL_CARD.md` | NEW — 1-page printable Q&A rebuttal reference | Day 5 evening | All 8 pre-canned Q&A rebuttals + numbers placeholder |

### Command cheatsheet (for Claude Code to invoke)

```bash
# Day 1 verification
uv sync --extra rl
python scripts/smoke_grpo_convergence.py      # ~30min; pass if reward moves

# Day 2 structured Generator + Move 11 smoke
pytest tests/test_self_play.py -x
python -m ml_training_debugger.diff_validator seeds/seed_diffs.jsonl     # all 20 pass
python -m ml_training_debugger.code_mutation_generator --n-samples 10    # ≥3 pass

# Day 3 Move 15 tournament
pytest tests/test_tournament.py -x
python scripts/run_frontier_baseline.py --n-bugs 30 --models all         # ~$30-50 API
python scripts/scrape_pytorchbugbench.py --target-n 30

# Day 4 training + demo
jupyter nbconvert --execute scripts/train_grpo_colab.ipynb               # Colab or local
python scripts/eval_cross_framework.py --checkpoint checkpoints/solver_selfplay_v1

# Day 5 analysis + rename
jupyter nbconvert --execute notebooks/empirical_finding.ipynb
python scripts/extract_emergent_moment.py
bash scripts/rename_to_bugforge.sh
pytest tests/ -x                                                         # all 251+ green
```

### Regression guards (run after every major change)

```bash
# MUST pass at every checkpoint:
pytest tests/ -x                                                         # 251 tests green
python baseline_inference.py --task-id task_001 --expected-score 0.9    # Mode A regression
python baseline_heuristic.py --tasks all --min-score 0.7                # Solver baseline intact
```

### Task ordering for Claude Code (strict)

When the user says "implement Move N," Claude Code should:
1. **Check preconditions** — confirm all files/dependencies from earlier days exist
2. **Run regression guard** — `pytest tests/ -x` before starting
3. **Create directories first** — avoid mid-implementation `mkdir` chains
4. **Implement + test incrementally** — unit test per new class, integration test per endpoint
5. **Run regression guard after** — `pytest tests/ -x` confirming nothing broke
6. **Commit per move** with message prefix `[Move NN]` for traceability

### Dependency graph (which move gates which)

```
§15.5 Pre-Day-1  ─────────────────────────────────┐
                                                   ▼
Day 1 infra (pyproject edit, smoke test)  ─────►  Day 2 structured Generator
                                                   │
                                                   ├─► Move 11 smoke test (Day 2 PM)
                                                   │       │ GREEN/YELLOW
                                                   │       ▼
                                                   │   Move 11 full impl (Day 4 PM)
                                                   │
                                                   ├─► Move 12 instrumentation (Day 3 PM)  ─►  Move 14 extraction (Day 5 AM)
                                                   │
                                                   ├─► Move 15 Multi-Gen tournament (Day 3 AM, gated on Move 11 smoke)
                                                   │
                                                   ├─► Frontier-LLM baseline (Day 3 PM)  ─►  Move 13 cost column (Day 5 AM)
                                                   │
                                                   ├─► Move 17 cross-framework (Day 3 PM + Day 4 PM)
                                                   │
                                                   └─► Move 16 rename (Day 5 PM — last, safest)
```

### What Claude Code should NOT do

- ❌ Rename the `ml_training_debugger` package (breaks 251 tests)
- ❌ Modify `ml_training_debugger/pytorch_engine.py` or `graders.py` core logic (they work — only *extend*, never rewrite)
- ❌ Skip `pytest tests/ -x` between moves (regression creep = plan failure)
- ❌ Implement Move 15 if Move 11 smoke test is RED (no recovery budget)
- ❌ Start Move 16 rename before Day 5 PM (name changes in mid-implementation cause rebase pain)
- ❌ Delete any committed checkpoint (demo insurance)
- ❌ Add new fault families beyond the existing 7 (scope creep — §11 hard NO)
- ❌ Write new tests for existing functionality — only for new code in Moves 11–17

---

*Last updated: 2026-04-20 — post Moves 14–17 integration + Claude Code Execution Guide*
*Plan owner: Omkar Kadam*
*Project: BugForge (née PyTorch Training Run Debugger / ML Debugger v2)*
*Target: #1 — honest P ~52–58% with all upgrades landing; floor P(Top 6) ~95%*
