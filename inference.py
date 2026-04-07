#!/usr/bin/env python3
"""Inference script for the PyTorch Training Run Debugger.

Runs an LLM agent against the environment using the OpenAI client
and the standard OpenEnv GenericEnvClient (env.reset / env.step).
Emits structured [START]/[STEP]/[END] logs to stdout as required by
the hackathon evaluator.

Required environment variables (set by hackathon evaluator):
    API_BASE_URL  — OpenAI-compatible API endpoint
    MODEL_NAME    — Model to use (e.g., "gpt-4o", "llama-3.3-70b")
    HF_TOKEN      — Hugging Face token (used as API key if OPENAI_API_KEY not set)

Optional:
    OPENAI_API_KEY — API key (takes precedence over HF_TOKEN)
    ENV_URL        — Environment server URL (default: http://localhost:7860)
    TASK_NAME      — Task to run (default: task_001)
    IMAGE_NAME     — Docker image name (if set, uses from_docker_image)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai", flush=True)
    sys.exit(1)

from openenv.core import GenericAction, GenericEnvClient

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "")
TASK_NAME = os.environ.get("TASK_NAME", "task_001")
BENCHMARK = "pytorch-training-debugger"

MAX_STEPS = 25
# Max achievable reward: +0.50 (diagnosis) +0.40 (convergence) +5*0.05 (investigations)
# minus step penalties. Use 1.15 as the theoretical ceiling for normalization.
MAX_TOTAL_REWARD = 1.15
SUCCESS_SCORE_THRESHOLD = 0.5
TEMPERATURE = 0.0
MAX_TOKENS = 300
FALLBACK_ACTION = '{"action_type": "inspect_gradients"}'

# ---------------------------------------------------------------------------
# Structured logging — [START]/[STEP]/[END] format (hackathon requirement)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt for the LLM agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert ML engineer debugging a PyTorch training run.
You are interacting with an environment that simulates a broken training job.

Available actions (respond with JSON only, no explanation):
- {"action_type": "inspect_gradients"} - View gradient statistics per layer
- {"action_type": "inspect_data_batch"} - View data batch statistics
- {"action_type": "inspect_model_modes"} - View model layer modes (train/eval)
- {"action_type": "inspect_model_weights"} - View model weight statistics
- {"action_type": "inspect_code"} - View PyTorch training code
- {"action_type": "modify_config", "target": "<field>", "value": <val>}
- {"action_type": "add_callback"} - Add gradient clipping/scheduler
- {"action_type": "patch_data_loader"} - Fix data pipeline issues
- {"action_type": "fix_model_mode"} - Call model.train()
- {"action_type": "fix_code", "line": <int>, "replacement": "<code>"}
- {"action_type": "restart_run"} - Restart training (requires a fix first)
- {"action_type": "mark_diagnosed", "diagnosis": "<cause>"} - Submit diagnosis

Valid diagnoses: lr_too_high, vanishing_gradients, data_leakage, \
overfitting, batchnorm_eval_mode, code_bug, scheduler_misconfigured

IMPORTANT: Respond with ONLY a valid JSON action object."""


def _build_obs_summary(obs: dict) -> dict:
    """Build a compact observation summary for the LLM context."""
    summary: dict = {"available_actions": obs.get("available_actions", [])}
    if obs.get("error_log"):
        summary["error_log"] = obs["error_log"]
    if obs.get("training_loss_history"):
        summary["loss_trend"] = obs["training_loss_history"][:5]
    if obs.get("val_accuracy_history"):
        summary["val_acc_trend"] = obs["val_accuracy_history"][:5]
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
    if obs.get("model_weight_stats"):
        summary["weight_stats"] = [
            {
                "layer": w.get("layer_name", ""),
                "norm": round(w.get("weight_norm", 0), 4),
            }
            for w in obs["model_weight_stats"]
        ]
    if obs.get("code_snippet"):
        cs = obs["code_snippet"]
        summary["code"] = cs.get("code", "")[:600]
        summary["hint"] = cs.get("hint", "")
    if obs.get("notes"):
        summary["notes"] = obs["notes"]
    return summary


def get_model_message(
    client: OpenAI,
    step: int,
    last_obs_summary: dict,
    last_reward: float,
    history: List[str],
) -> str:
    """Get next action from the LLM."""
    history_ctx = "\n".join(history[-5:]) if history else "No previous steps."
    user_content = (
        f"Step {step}. Last reward: {last_reward:+.2f}\n"
        f"Recent history:\n{history_ctx}\n\n"
        f"Current observation:\n"
        f"{json.dumps(last_obs_summary, indent=2, default=str)}\n\n"
        "What action should you take next? Respond with JSON only."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else FALLBACK_ACTION
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return FALLBACK_ACTION


def parse_action(raw: str) -> str:
    """Clean up LLM output to extract JSON action string."""
    text = raw.strip().strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return FALLBACK_ACTION


async def main() -> None:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        if not API_KEY:
            raise RuntimeError("OPENAI_API_KEY or HF_TOKEN required.")

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        # Connect to environment via standard OpenEnv client
        if IMAGE_NAME:
            env = await GenericEnvClient.from_docker_image(IMAGE_NAME)
        else:
            env = GenericEnvClient(base_url=ENV_URL, message_timeout_s=120.0)
            await env.connect()

        result = await env.reset(task_id=TASK_NAME, seed=42)
        obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs_summary = _build_obs_summary(obs)
            raw = get_model_message(client, step, obs_summary, last_reward, history)
            action_str = parse_action(raw)

            action = GenericAction(json.loads(action_str))
            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = (
                obs.get("notes")
                if "invalid" in str(obs.get("notes", "")).lower()
                else None
            )

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.01), 0.99)  # clamp to (0, 1) exclusive
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Unhandled error: {exc}", flush=True)

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
