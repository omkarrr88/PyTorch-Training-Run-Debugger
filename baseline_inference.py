#!/usr/bin/env python3
"""LLM baseline agent using Google Gemini (via OpenAI-compatible SDK).

Requires GEMINI_API_KEY environment variable (or pass via --api-key).
Uses temperature=0.0 for near-deterministic behavior.
Usage:
    GEMINI_API_KEY=... python baseline_inference.py
    python baseline_inference.py --api-key YOUR_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Load .env file if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

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
    "task_007",
]

SYSTEM_PROMPT = """You are an expert ML engineer debugging a PyTorch training run.
You are interacting with an environment that simulates a broken training job.

Available actions (respond with JSON only, no explanation):
- {"action_type": "inspect_gradients"} - View gradient statistics per layer
- {"action_type": "inspect_data_batch"} - View data batch statistics and confusion matrix
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

Valid diagnoses: lr_too_high, vanishing_gradients, data_leakage, overfitting, batchnorm_eval_mode, code_bug, scheduler_misconfigured

Strategy:
1. First investigate by inspecting gradients, data, model modes, and code
2. Form a hypothesis based on the evidence gathered
3. Apply the correct fix for the identified root cause
4. Restart training to verify the fix works
5. Submit your diagnosis

IMPORTANT: Respond with ONLY a valid JSON action object. No explanation, no markdown, no code blocks."""


def run_llm_episode(task_id: str, client: OpenAI, model_name: str) -> float:
    """Run one LLM agent episode."""
    env = MLTrainingEnvironment()
    obs = env.reset(seed=42, episode_id=f"llm_{task_id}", task_id=task_id)

    initial_obs = {
        "training_loss_history": obs.training_loss_history[:5],
        "val_accuracy_history": obs.val_accuracy_history[:5],
        "current_config": obs.current_config.model_dump(),
        "error_log": obs.error_log,
        "available_actions": obs.available_actions,
        "notes": obs.notes,
        "gpu_memory_used_gb": obs.gpu_memory_used_gb,
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"New episode started for a broken PyTorch training run.\n\nInitial observation:\n{json.dumps(initial_obs, indent=2, default=str)}",
        },
    ]

    for step in range(25):
        if obs.done:
            break

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=300,
            )
            action_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    Step {step}: API error — {e}", file=sys.stderr)
            break

        # Clean up common LLM formatting issues
        action_text = action_text.strip("`").strip()
        if action_text.startswith("json"):
            action_text = action_text[4:].strip()

        messages.append({"role": "assistant", "content": action_text})

        try:
            action_data = json.loads(action_text)
            action = MLTrainingAction(**action_data)
        except (json.JSONDecodeError, Exception) as e:
            messages.append(
                {
                    "role": "user",
                    "content": f"Invalid action format: {e}. Respond with ONLY valid JSON.",
                }
            )
            continue

        obs = env.step(action)

        obs_summary: dict = {
            "reward": obs.reward,
            "done": obs.done,
            "step": obs.episode_state.step_count,
            "available_actions": obs.available_actions,
        }
        if obs.error_log:
            obs_summary["error_log"] = obs.error_log
        if obs.gradient_stats:
            obs_summary["gradient_stats"] = [
                {
                    "layer": g.layer_name,
                    "mean_norm": round(g.mean_norm, 4),
                    "exploding": g.is_exploding,
                    "vanishing": g.is_vanishing,
                }
                for g in obs.gradient_stats
            ]
        if obs.data_batch_stats:
            obs_summary["data_overlap"] = obs.data_batch_stats.class_overlap_score
            obs_summary["duplicate_ratio"] = obs.data_batch_stats.duplicate_ratio
        if obs.model_mode_info:
            obs_summary["model_modes"] = obs.model_mode_info
        if obs.code_snippet:
            obs_summary["code"] = obs.code_snippet.code[:600]
            obs_summary["hint"] = obs.code_snippet.hint

        messages.append(
            {
                "role": "user",
                "content": f"Observation after your action:\n{json.dumps(obs_summary, indent=2, default=str)}",
            }
        )

    session = env._get_session()
    return session.last_score if session and session.last_score is not None else 0.0


PROVIDERS = {
    "groq": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "cerebras": {
        "env_key": "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1",
        "default_model": "llama3.1-8b",
    },
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
        "default_model": "gpt-4o",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM baseline agent")
    parser.add_argument("--url", default="http://localhost:7860")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument(
        "--provider",
        default="groq",
        choices=list(PROVIDERS.keys()),
        help="LLM provider (default: groq)",
    )
    parser.add_argument("--model", default=None, help="Model name (auto-detected from provider)")
    args = parser.parse_args()

    prov = PROVIDERS[args.provider]
    api_key = args.api_key or os.environ.get(prov["env_key"])
    if not api_key:
        print(f"Error: Set {prov['env_key']} env var or pass --api-key")
        sys.exit(1)

    model_name = args.model or prov["default_model"]
    client_kwargs: dict = {"api_key": api_key}
    if prov["base_url"]:
        client_kwargs["base_url"] = prov["base_url"]
    client = OpenAI(**client_kwargs)

    scores: dict[str, float] = {}
    print(f"Running LLM baseline with {args.provider}/{model_name}...", file=sys.stderr)

    for task_id in ALL_TASKS:
        try:
            score = run_llm_episode(task_id, client, model_name)
            scores[task_id] = round(score, 4)
            print(f"  {task_id}: {score:.4f}", file=sys.stderr)
        except Exception as e:
            print(f"  {task_id}: ERROR — {e}", file=sys.stderr)
            scores[task_id] = 0.0

    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
