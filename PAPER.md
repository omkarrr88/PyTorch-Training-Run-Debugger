# Context-Gated Reward Shaping for Evidence-Based ML Debugging

## Abstract

We present a reinforcement learning environment for training AI agents to debug broken PyTorch training runs. The environment introduces **context-gated reward shaping** — a penalty mechanism that distinguishes between reasonable prior actions (no penalty) and actions that ignore evidence the agent has already gathered (penalty). This single mechanic encodes evidence-based reasoning directly into the reward signal, teaching agents to reason about their accumulated knowledge rather than follow fixed playbooks. The environment covers 7 failure scenarios across 3 difficulty tiers, uses real PyTorch model internals (torch.nn.Module, torch.autograd, state_dict()), and includes a code-level debugging task where agents must read and fix actual Python source code.

## Motivation

ML teams spend 15-25% of engineer time debugging silent training failures — runs that produce no errors, just mysteriously bad metrics. Each misdiagnosed restart wastes $2-8/hour/GPU. Existing RL environments focus on games, navigation, or text tasks. No environment trains agents for the diagnostic reasoning process that ML engineers perform daily: gathering evidence from gradients, weights, data, and code; forming hypotheses under uncertainty; and making evidence-based decisions about which fix to apply.

## Method: Context-Gated Reward Shaping

Standard RL environments use stateless rewards: "did action X happen?" Our environment tracks the agent's information state and conditions penalties on what the agent has already observed.

**Core mechanic:** An agent that adds gradient clipping *before* inspecting gradients follows a reasonable prior — no penalty. An agent that inspects gradients, sees they are normal, and *then* adds gradient clipping is ignoring counter-evidence — **-0.20 penalty**.

Formally: the penalty fires when `gradients_inspected == True AND gradients_were_normal == True AND action == add_callback`. This gate requires two conditions to be jointly true, both of which depend on prior agent actions.

This teaches agents a transferable skill: *don't ignore what you've already learned*. In real MLOps, ignoring gathered evidence leads to wasted GPU hours and delayed incident resolution.

## Environment Design

- **7 tasks** across 3 difficulty tiers (easy, medium, hard) with difficulty scaling (1-5)
- **Real PyTorch models**: SimpleCNN (~50K params) and SimpleMLP (~20K params) with real torch.autograd gradients
- **Progressive information reveal**: agents must actively choose what to investigate
- **Code-level debugging** (Task 6): agent reads PyTorch source and submits line-by-line fixes
- **Red herring injection** (Task 5): misleading gradient spikes, GPU memory warnings, near-vanishing layers
- **Confusion matrices** in data batch inspection for richer diagnostic signals
- **7 diagnosis types**: lr_too_high, vanishing_gradients, data_leakage, overfitting, batchnorm_eval_mode, code_bug, scheduler_misconfigured

## Results

Baseline scores demonstrate meaningful difficulty progression:

| Task | Heuristic | Description |
|------|-----------|-------------|
| task_001 | 1.00 | Exploding gradients — direct signal |
| task_002 | 1.00 | Vanishing gradients — direct signal |
| task_003 | 1.00 | Data leakage — class overlap detection |
| task_004 | 1.00 | Overfitting — train-val divergence |
| task_005 | 0.35 | BatchNorm eval mode — red herrings trap heuristic |
| task_006 | 1.00 | Code bug — pattern matching catches 2/4 variants |
| task_007 | 0.60 | Scheduler misconfigured — stagnation detection |

The rule-based heuristic scores 0.35 on Task 5 because its fixed investigation order causes it to chase the gradient spike red herring before checking model modes. A reasoning agent that inspects model modes would avoid this trap.

## Conclusion

Context-gated reward shaping is a general technique applicable to any RL environment where agents must reason about accumulated evidence. By conditioning penalties on the agent's information state, we create environments that reward systematic investigation over pattern-matching — a capability with direct transfer value to real-world MLOps debugging.

The environment is deployed as an OpenEnv-compatible Docker container on Hugging Face Spaces with full API documentation, a live diagnostic dashboard, and bit-exact reproducible baselines.

---

*Built for the Meta PyTorch OpenEnv Hackathon x Scaler School of Technology, 2026.*
