#!/usr/bin/env python3
"""LLM baseline agent using OpenAI GPT-4o.

Optional — requires OPENAI_API_KEY environment variable.
Uses temperature=0.0 and seed=42 for near-deterministic behavior.
Spec reference: Section 17.

Usage:
    OPENAI_API_KEY=... python baseline_inference.py [--url http://localhost:7860]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

from ml_training_debugger.models import MLTrainingAction
from server.environment import MLTrainingEnvironment

ALL_TASKS = [
    "task_001",
    "task_002",
    "task_003",
    "task_004",
    "task_005",
    "task_006",
]

SYSTEM_PROMPT = """You are an expert ML engineer debugging a PyTorch training run.
You are interacting with an environment that simulates a broken training job.

Available actions (respond with JSON):
- {"action_type": "inspect_gradients"} - View gradient statistics per layer
- {"action_type": "inspect_data_batch"} - View data batch statistics
- {"action_type": "inspect_model_modes"} - View model layer modes (train/eval)
- {"action_type": "inspect_model_weights"} - View model weight statistics
- {"action_type": "inspect_code"} - View PyTorch training code
- {"action_type": "modify_config", "target": "<field>", "value": <val>} - Change a hyperparameter
- {"action_type": "add_callback"} - Add gradient clipping/scheduler
- {"action_type": "patch_data_loader"} - Fix data pipeline issues
- {"action_type": "fix_model_mode"} - Call model.train()
- {"action_type": "fix_code", "line": <int>, "replacement": "<code>"} - Fix a code line
- {"action_type": "restart_run"} - Restart training (requires a fix first)
- {"action_type": "mark_diagnosed", "diagnosis": "<cause>"} - Submit diagnosis

Valid diagnoses: lr_too_high, vanishing_gradients, data_leakage, overfitting, batchnorm_eval_mode, code_bug

Strategy:
1. First investigate by inspecting gradients, data, and model modes
2. Form a hypothesis based on the evidence
3. Apply the correct fix
4. Restart training to verify
5. Submit your diagnosis

Respond with ONLY a valid JSON action object, no explanation."""


def run_llm_episode(task_id: str, client: OpenAI) -> float:
    """Run one LLM agent episode."""
    env = MLTrainingEnvironment()
    obs = env.reset(seed=42, episode_id=f"llm_{task_id}", task_id=task_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"New episode started. Observation:\n{json.dumps(obs.model_dump(), indent=2, default=str)[:3000]}"},
    ]

    for step in range(20):
        if obs.done:
            break

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
            seed=42,
            max_tokens=200,
        )

        action_text = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": action_text})

        try:
            action_data = json.loads(action_text)
            action = MLTrainingAction(**action_data)
        except (json.JSONDecodeError, Exception) as e:
            messages.append({"role": "user", "content": f"Invalid action: {e}. Try again with valid JSON."})
            continue

        obs = env.step(action)
        obs_summary = {
            "reward": obs.reward,
            "done": obs.done,
            "step": obs.episode_state.step_count,
            "available_actions": obs.available_actions,
            "error_log": obs.error_log,
        }
        if obs.gradient_stats:
            obs_summary["gradient_stats"] = [
                {"layer": g.layer_name, "mean_norm": round(g.mean_norm, 4), "exploding": g.is_exploding, "vanishing": g.is_vanishing}
                for g in obs.gradient_stats
            ]
        if obs.data_batch_stats:
            obs_summary["data_overlap"] = obs.data_batch_stats.class_overlap_score
        if obs.model_mode_info:
            obs_summary["model_modes"] = obs.model_mode_info
        if obs.code_snippet:
            obs_summary["code"] = obs.code_snippet.code[:500]

        messages.append({"role": "user", "content": f"Observation:\n{json.dumps(obs_summary, indent=2, default=str)}"})

    session = env._get_session()
    return session.last_score if session and session.last_score is not None else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM baseline agent (GPT-4o)")
    parser.add_argument("--url", default="http://localhost:7860")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    scores: dict[str, float] = {}

    for task_id in ALL_TASKS:
        try:
            score = run_llm_episode(task_id, client)
            scores[task_id] = round(score, 4)
            print(f"  {task_id}: {score:.4f}", file=sys.stderr)
        except Exception as e:
            print(f"  {task_id}: ERROR — {e}", file=sys.stderr)
            scores[task_id] = 0.0

    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
