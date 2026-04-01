#!/usr/bin/env python3
"""Inference script for the PyTorch Training Run Debugger.

Runs an LLM agent against all 7 tasks using the OpenAI client.
Connects to the environment server via WebSocket.

Required environment variables (set by hackathon evaluator):
    API_BASE_URL  — OpenAI-compatible API endpoint
    MODEL_NAME    — Model to use (e.g., "gpt-4o", "llama-3.3-70b")
    HF_TOKEN      — Hugging Face token (used as API key if OPENAI_API_KEY not set)

Optional:
    OPENAI_API_KEY — API key (takes precedence over HF_TOKEN)
    ENV_URL        — Environment server URL (default: http://localhost:7860)

Usage:
    API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o OPENAI_API_KEY=sk-... python inference.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("Error: websockets package not installed. Run: pip install websockets", file=sys.stderr)
    sys.exit(1)

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


def _build_obs_summary(obs_data: dict) -> dict:
    """Build a compact observation summary for the LLM context."""
    obs = obs_data.get("observation", obs_data)
    summary: dict = {
        "reward": obs_data.get("reward"),
        "done": obs_data.get("done"),
        "available_actions": obs.get("available_actions", []),
    }
    if obs.get("error_log"):
        summary["error_log"] = obs["error_log"]
    if obs.get("gradient_stats"):
        summary["gradient_stats"] = [
            {
                "layer": g.get("layer_name", ""),
                "mean_norm": round(g.get("mean_norm", 0), 4),
                "exploding": g.get("is_exploding", False),
                "vanishing": g.get("is_vanishing", False),
            }
            for g in obs["gradient_stats"]
        ]
    if obs.get("data_batch_stats"):
        dbs = obs["data_batch_stats"]
        summary["data_overlap"] = dbs.get("class_overlap_score", 0)
        summary["duplicate_ratio"] = dbs.get("duplicate_ratio", 0)
    if obs.get("model_mode_info"):
        summary["model_modes"] = obs["model_mode_info"]
    if obs.get("code_snippet"):
        cs = obs["code_snippet"]
        summary["code"] = cs.get("code", "")[:600]
        summary["hint"] = cs.get("hint", "")
    return summary


async def run_llm_episode(
    task_id: str, ws_url: str, client: OpenAI, model_name: str
) -> float:
    """Run one LLM agent episode via WebSocket. Returns grader score."""
    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
        # Reset
        await ws.send(json.dumps({
            "type": "reset",
            "data": {"task_id": task_id, "seed": 42},
        }))
        resp = json.loads(await ws.recv())
        obs_data = resp.get("data", resp)
        obs = obs_data.get("observation", obs_data)

        initial_obs = {
            "training_loss_history": obs.get("training_loss_history", [])[:5],
            "val_accuracy_history": obs.get("val_accuracy_history", [])[:5],
            "current_config": obs.get("current_config", {}),
            "error_log": obs.get("error_log"),
            "available_actions": obs.get("available_actions", []),
            "notes": obs.get("notes"),
        }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "New episode started for a broken PyTorch training run.\n\n"
                    f"Initial observation:\n{json.dumps(initial_obs, indent=2, default=str)}"
                ),
            },
        ]

        last_score = 0.0

        for step in range(25):
            if obs_data.get("done"):
                break

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=300,
                    timeout=30,
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
            except json.JSONDecodeError as e:
                messages.append({
                    "role": "user",
                    "content": f"Invalid JSON: {e}. Respond with ONLY valid JSON.",
                })
                continue

            # Send action via WebSocket
            await ws.send(json.dumps({"type": "step", "data": action_data}))
            resp = json.loads(await ws.recv())
            obs_data = resp.get("data", resp)

            summary = _build_obs_summary(obs_data)
            messages.append({
                "role": "user",
                "content": f"Observation after your action:\n{json.dumps(summary, indent=2, default=str)}",
            })

    # Get grader score via HTTP POST
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")
    try:
        req = urllib.request.Request(f"{env_url}/grader", method="POST")
        grader_resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        last_score = grader_resp.get("score", 0.0) or 0.0
    except Exception as e:
        print(f"    Grader request failed: {e}", file=sys.stderr)

    return last_score


async def _run_with_timeout(
    task_id: str, ws_url: str, client: OpenAI, model_name: str
) -> float:
    """Run episode with per-task timeout (150s = 2.5 min per task)."""
    try:
        return await asyncio.wait_for(
            run_llm_episode(task_id, ws_url, client, model_name),
            timeout=150,
        )
    except asyncio.TimeoutError:
        print(f"    {task_id}: TIMEOUT (>150s)", file=sys.stderr)
        return 0.0


def main() -> None:
    # Read hackathon-required environment variables
    api_base_url = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")

    if not api_base_url:
        print("Error: API_BASE_URL environment variable is required.", file=sys.stderr)
        print(
            "Usage: API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o "
            "OPENAI_API_KEY=sk-... python inference.py",
            file=sys.stderr,
        )
        sys.exit(1)

    if not model_name:
        print("Error: MODEL_NAME environment variable is required.", file=sys.stderr)
        sys.exit(1)

    if not api_key:
        print("Error: OPENAI_API_KEY or HF_TOKEN environment variable is required.", file=sys.stderr)
        sys.exit(1)

    # Environment server URL (default localhost, override for remote)
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")
    ws_proto = "wss" if env_url.startswith("https") else "ws"
    host = env_url.replace("https://", "").replace("http://", "")
    ws_url = f"{ws_proto}://{host}/ws"

    client = OpenAI(api_key=api_key, base_url=api_base_url)

    print(f"Running inference with model={model_name}", file=sys.stderr)
    print(f"  LLM API: {api_base_url}", file=sys.stderr)
    print(f"  Environment: {ws_url}", file=sys.stderr)

    scores: dict[str, float] = {}
    start_time = time.time()

    for task_id in ALL_TASKS:
        task_start = time.time()
        try:
            score = asyncio.run(_run_with_timeout(task_id, ws_url, client, model_name))
            scores[task_id] = round(score, 4)
            elapsed = time.time() - task_start
            print(f"  {task_id}: {score:.4f} ({elapsed:.1f}s)", file=sys.stderr)
        except Exception as e:
            print(f"  {task_id}: ERROR — {e}", file=sys.stderr)
            scores[task_id] = 0.0

    total_time = time.time() - start_time
    if total_time > 1100:
        print(
            f"\nWARNING: Total time {total_time:.0f}s approaching 20-minute limit",
            file=sys.stderr,
        )
    print(f"\nTotal time: {total_time:.1f}s", file=sys.stderr)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
