# PyTorch Training Run Debugger — Explained Simply

> This file explains the entire project as if you're 10 years old. No jargon. Just simple language.

---

## What Is This Project?

Imagine you're a doctor, but instead of fixing sick people, you fix **sick computers that are trying to learn**.

When computers learn (this is called "Machine Learning" or ML), they look at thousands of examples — like pictures of cats and dogs — and slowly get better at telling them apart. This learning process is called **training**.

But sometimes, training goes wrong. The computer makes mistakes, gets confused, or learns the wrong things. When that happens, a human engineer has to figure out what went wrong and fix it — just like a doctor diagnosing a patient.

**This project builds a practice hospital for AI doctors.** It creates fake "sick training runs" with known problems, and then an AI agent (the doctor) has to:

1. **Investigate** — Look at clues (like checking temperature or blood pressure)
2. **Diagnose** — Figure out what's wrong
3. **Fix** — Apply the right treatment
4. **Verify** — Check if the patient recovered

---

## Why Does This Matter?

Real companies like Meta, Google, and OpenAI spend millions of dollars training AI models. When training breaks, engineers waste hours (sometimes days!) figuring out what went wrong. Each hour of broken training can cost **$2-$8 per GPU** — and some companies use thousands of GPUs at once.

If we could train an AI to automatically find and fix these problems, it would save enormous amounts of time and money.

This project is a **training ground** where AI agents can practice debugging — like a flight simulator for pilots, but for ML engineers.

---

## How Does It Work? (The Big Picture)

Think of it like a detective game with 6 mystery cases:

### The Game Rules

1. **The computer shows you a broken training run** — You see charts showing how the training is going (spoiler: it's going badly!)
2. **You can investigate** — You have 5 different "magnifying glasses" to look at different parts of the problem
3. **You figure out what's wrong** — You pick from a list of 6 possible problems
4. **You fix it** — You apply the right fix
5. **You restart and check** — You restart the training and see if it works now
6. **You submit your answer** — "I think the problem was X"

If you're right, you get points. If you're wrong, you lose points. If you investigate smartly, you get bonus points. If you ignore evidence and do something silly, you get penalty points.

---

## The 6 Mystery Cases (Tasks)

### Easy Cases (Like finding a broken window)

**Case 1: Learning Rate Too High (task_001)**
> Imagine you're learning to ride a bike, but someone set the speed to 100 mph. You'd crash immediately!

That's what happens here. The computer is learning too fast and everything explodes. The numbers go crazy and become "NaN" (Not a Number — like dividing by zero).

**Clues:** Every part of the computer shows "EXPLODING!" when you check the gradients (the direction signals that guide learning).

**Fix:** Turn down the speed (reduce the learning rate from 0.1 to 0.001).

---

**Case 2: Vanishing Gradients (task_002)**
> Now imagine you're whispering instructions to someone 100 rooms away. By the time the message reaches them, it's too quiet to hear.

The learning signals get weaker and weaker as they travel through the computer's brain layers. The deeper layers get almost zero signal — so they can't learn anything.

**Clues:** Deeper layers show "VANISHING!" gradients. The loss curve is flat — nothing is being learned.

**Fix:** Increase the learning rate so the signals are louder.

---

### Medium Cases (Like finding a hidden leak)

**Case 3: Data Leakage (task_003)**
> Imagine taking a math test, but the answer key is mixed into your practice problems. You'd score 100% — but you didn't actually learn anything!

The training data and test data got mixed together. The computer looks amazing on tests, but it's just memorizing answers — it hasn't actually learned.

**Clues:** Suspiciously high test scores from the very start. When you check the data, you find a "class overlap score" above 0.5 — meaning lots of test answers leaked into the training set.

**Trick:** There's a misleading note saying "we upgraded the model architecture" — making you think the high scores are from a better model, not leaked data.

**Fix:** Clean the data pipeline to remove the overlap.

---

**Case 4: Overfitting (task_004)**
> Imagine memorizing every single answer to last year's exam, but then failing this year's exam because the questions are slightly different.

The computer has memorized the training data perfectly (train loss near zero!) but fails on new data it hasn't seen before (validation loss keeps rising).

**Clues:** Training loss drops to almost zero while validation loss goes up — the classic "train-val divergence."

**Fix:** Add regularization (weight decay) — this is like telling the computer "don't memorize, understand the patterns instead."

---

### Hard Cases (Like solving a mystery with fake clues)

**Case 5: BatchNorm Eval Mode (task_005)**
> Imagine a student who studies perfectly at home but freezes during the actual exam because they switched into "test mode" too early.

The computer's model has a special feature called BatchNorm that behaves differently during training vs testing. Someone accidentally left it in "test mode" during training. This causes subtle, slow degradation — not an obvious crash.

**The Trap:** This case has **red herrings** — fake clues designed to mislead you:
- One layer's gradient suddenly spikes (but it's not actually exploding)
- GPU memory is at 91% (looks scary, but it's not the problem)
- One layer has near-vanishing gradients (but that's normal for this layer)
- An error log warns about GPU memory (irrelevant to the real problem)

**Clues:** When you check the model modes, you find all layers are in "eval" (test) mode instead of "train" mode. That's the real problem.

**Why it's hard:** Most agents see the gradient spike and immediately try to fix gradients — falling for the trap. The smart agent checks model modes and finds the real issue.

---

**Case 6: Code Bug (task_006)**
> Imagine a recipe that says "bake for 30 minutes" but someone accidentally changed it to "bake for 0 minutes." The oven runs, but nothing gets cooked.

There's an actual bug in the Python code. The agent sees the source code and has to find the buggy line and fix it. There are 4 possible bugs:

1. **eval_mode** — `model.eval()` instead of `model.train()` (wrong mode)
2. **detach_loss** — `loss.detach()` before `.backward()` (disconnects the learning signal)
3. **zero_grad_missing** — Forgot to clear old gradients (gradients pile up incorrectly)
4. **inplace_relu** — `inplace=True` on ReLU (corrupts the computation graph)

**Why it's hard:** The agent must actually READ code and understand what each line does — not just look at numbers and charts.

---

## The Scoring System

### Rewards (Points You Earn)

Think of it like a video game:

| What You Do | Points | Why |
|-------------|--------|-----|
| Take any action | **-0.01** | Every move costs a tiny bit (encourages efficiency) |
| Investigate something for the first time | **+0.05** | Looking at clues is good! |
| Correct diagnosis | **+0.50** | You found the answer! |
| Fix works and training recovers | **+0.40** | Your fix actually helped! |

### Penalties (Points You Lose)

| What You Do | Points | Why |
|-------------|--------|-----|
| Do something invalid | **-0.05** | You tried something that's not allowed |
| Wrong code fix | **-0.10** | Your code fix didn't work |
| Wrong diagnosis | **-0.30** | You guessed wrong |

### The Special Penalty: Context-Gated Penalty

This is the **coolest part** of the project. Here's how it works:

> You check the gradients and see they're all normal. Then you add gradient clipping anyway (a fix for gradient problems). But wait — YOU ALREADY KNOW the gradients are fine! You're ignoring your own evidence!

**Penalty: -0.20 points**

But if you add gradient clipping BEFORE checking gradients? No penalty — you haven't seen any evidence yet, so it's a reasonable guess.

This teaches the AI: **"Don't ignore what you've already learned."**

---

### The Grader (Final Score)

At the end of each case, a grader gives you a score from **0.0 to 1.0**:

- **1.0** = Perfect — investigated, fixed, restarted, and diagnosed correctly
- **0.5-0.8** = Partial — got some things right, missed others
- **0.0** = Failed — wrong diagnosis, no fix, or ran out of steps

The grader looks at the WHOLE story of what you did, not just the final answer.

---

## How the Code Is Organized

```
ML Debugger/
│
├── ml_training_debugger/          ← The brain of the project
│   ├── models.py                  ← Data shapes (what observations and actions look like)
│   ├── scenarios.py               ← Creates the 6 mystery cases with random parameters
│   ├── pytorch_engine.py          ← Real PyTorch model that gets "sick" (fault injection)
│   ├── simulation.py              ← Generates fake training charts (loss curves, accuracy)
│   ├── reward_engine.py           ← Calculates points for each action
│   ├── graders.py                 ← Final scoring (0.0 to 1.0) at episode end
│   ├── code_templates.py          ← The buggy code snippets for Task 6
│   └── client.py                  ← Helper for connecting to the environment
│
├── server/                        ← The web server
│   ├── app.py                     ← Main server with all API endpoints
│   ├── environment.py             ← The game logic (reset, step, state)
│   └── _baseline_results.py       ← Stores grader results
│
├── tests/                         ← 183 tests making sure everything works
│
├── baseline_heuristic.py          ← A simple robot that plays the game using rules
├── baseline_inference.py          ← A smart AI (GPT-4) that plays the game
├── Dockerfile                     ← Instructions to package everything in a container
├── openenv.yaml                   ← Configuration file for the OpenEnv framework
└── README.md                      ← Technical documentation
```

---

## How a Game Session Works (Step by Step)

Let's walk through a complete game:

### Step 1: Start a New Game
```
Agent: "Start task_001 please"
Environment: "Here's your broken training run:"
  - Loss history: [2.3, 3.5, 8.2, 45.0, inf, inf, inf, ...]  ← Yikes, numbers exploding!
  - Error log: "Loss is NaN at epoch 12"
  - Available actions: [inspect_gradients, inspect_data_batch, ...]
```

### Step 2: Investigate
```
Agent: "Let me inspect the gradients"
Environment: "Here's what I found:"
  - conv1: mean_norm=51.1, is_exploding=True
  - conv2: mean_norm=91.3, is_exploding=True
  - conv3: mean_norm=111.8, is_exploding=True
  - fc: mean_norm=37.7, is_exploding=True
  Reward: +0.04 (step penalty + investigation bonus)
```

### Step 3: Fix
```
Agent: "Reduce learning rate to 0.001"
Environment: "Config updated. learning_rate = 0.001"
  Reward: -0.01 (step penalty only)
```

### Step 4: Restart
```
Agent: "Restart the training run"
Environment: "Training restarted. Convergence detected!"
  Reward: +0.39 (step penalty + convergence bonus)
```

### Step 5: Diagnose
```
Agent: "The problem was lr_too_high"
Environment: "CORRECT! Episode complete."
  Reward: +0.49 (step penalty + correct diagnosis)
  Final grader score: 1.0 ← Perfect!
```

---

## What Makes This Project Special?

### 1. It Uses REAL PyTorch
This isn't fake data. When you inspect gradients, you're looking at real numbers computed by a real neural network using `torch.autograd`. The model has ~50,000 parameters and runs real forward/backward passes. This matters because the hackathon is organized by **Meta (the company that makes PyTorch)**.

### 2. Context-Gated Rewards
No other OpenEnv environment does this. The reward system tracks what the agent has learned and penalizes it for ignoring evidence. This teaches AI to reason like a real engineer — gather evidence first, then act.

### 3. Code-Level Debugging (Task 6)
The agent reads actual Python code and submits line-by-line fixes. This tests code understanding — not just number crunching. Meta cares about this because they want AI that can debug PyTorch code.

### 4. Red Herrings in Hard Tasks
Task 5 deliberately plants misleading clues. This separates agents that follow rigid patterns from agents that can reason through ambiguity — exactly like real debugging.

### 5. Progressive Information Reveal
The agent starts with limited information and must actively choose what to investigate. Each inspection reveals new data. This makes it a genuine investigation — not just a classification task.

---

## The Two Baselines (Robot Players)

### Baseline 1: The Rule-Following Robot (`baseline_heuristic.py`)
This robot follows a fixed checklist:
1. Check gradients → if exploding, fix learning rate
2. Check data → if leaking, patch data
3. Check model modes → if eval, fix mode
4. Check code → if bug found, fix it
5. If nothing works, guess "overfitting"

**Scores:** Perfect on easy/medium tasks, but only 0.35 on Task 5 because its fixed order means it tries to fix gradients before checking model modes — falling for the red herring.

### Baseline 2: The Smart AI (`baseline_inference.py`)
This uses GPT-4 to reason about the evidence. It reads the observations, thinks about what to do, and makes decisions. It should score higher on hard tasks because it can reason, not just follow rules.

---

## The Technology Stack

| Component | What It Is | Why We Use It |
|-----------|-----------|---------------|
| **Python 3.12** | Programming language | Modern, fast, supports type hints |
| **PyTorch (CPU)** | Machine learning framework | Real neural networks, real gradients (Meta's framework!) |
| **FastAPI** | Web framework | Fast, modern, auto-generates docs |
| **OpenEnv** | RL environment framework | Standard interface for AI agents (step/reset/state) |
| **Pydantic** | Data validation | Ensures all data is properly typed |
| **Plotly.js** | Charting library | Live dashboard with interactive charts |
| **Docker** | Containerization | Package everything so it runs anywhere |

---

## How to Think About This Project

**Analogy 1: Medical Training Simulator**
Medical students practice on mannequins before treating real patients. This project is a mannequin for AI debugging — the "patients" have known problems, and the "doctor" (AI agent) learns to diagnose them.

**Analogy 2: Escape Room**
Each task is like an escape room. You're locked in with clues scattered around. Some clues are helpful, some are red herrings. You need to investigate systematically, not randomly try everything.

**Analogy 3: Car Mechanic School**
A car comes in making weird noises. The mechanic can:
- Check the engine (inspect_gradients)
- Check the fuel (inspect_data_batch)
- Check the gearbox (inspect_model_modes)
- Read the error codes (inspect_code)
Then they fix the right part and test-drive it to confirm.

---

## Summary

| Question | Answer |
|----------|--------|
| **What?** | A practice environment where AI agents learn to debug broken PyTorch training runs |
| **Why?** | Real ML debugging costs companies millions. Training AI to do it has huge value. |
| **How?** | 6 mystery cases with real PyTorch models, progressive clue reveal, and smart scoring |
| **What's special?** | Real PyTorch internals, context-gated rewards, code-level debugging, red herrings |
| **Who's it for?** | AI researchers building smarter debugging agents |
| **Built with?** | Python, PyTorch, FastAPI, OpenEnv, Pydantic, Docker |
| **For what event?** | Meta PyTorch OpenEnv Hackathon x Scaler School of Technology |
