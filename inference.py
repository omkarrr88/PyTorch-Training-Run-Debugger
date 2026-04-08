#!/usr/bin/env python3
"""Inference script for the PyTorch Training Run Debugger.

Required environment variables (injected by evaluator):
    API_BASE_URL   — LiteLLM proxy endpoint
    API_KEY        — LiteLLM proxy key
    MODEL_NAME     — Model to use
    IMAGE_NAME     — Docker image for the environment (optional)
    ENV_URL        — Environment URL (HF Space)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import List, Optional

from openai import OpenAI
from openenv.core import GenericAction, GenericEnvClient

# ---------------------------------------------------------------------------
# Configuration — use evaluator-injected env vars EXACTLY as documented
# ---------------------------------------------------------------------------
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
ENV_URL = os.environ.get("ENV_URL", "https://ujjwalpardeshi-pytorch-training-debugger.hf.space")
BENCHMARK = "pytorch-training-debugger"

MAX_STEPS = 25
SUCCESS_SCORE_THRESHOLD = 0.5
TEMPERATURE = 0.0
MAX_TOKENS = 300

# All tasks to run — evaluator expects ALL tasks with graders
ALL_TASK_IDS = ["task_001", "task_002", "task_003", "task_004", "task_005", "task_006", "task_007"]

# ---------------------------------------------------------------------------
# Structured logging — matches evaluator's expected format
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    clean_action = action.replace("\n", " ").replace("\r", " ")
    print(
        f"[STEP] step={step} action={clean_action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
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
    """Get next action from the LLM with retry logic."""
    history_ctx = "\n".join(history[-5:]) if history else "No previous steps."
    user_content = (
        f"Step {step}. Last reward: {last_reward:+.2f}\n"
        f"Recent history:\n{history_ctx}\n\n"
        f"Current observation:\n"
        f"{json.dumps(last_obs_summary, indent=2, default=str)}\n\n"
        "What action should you take next? Respond with JSON only."
    )

    max_retries = 3
    for attempt in range(max_retries):
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
            if text:
                return text
        except Exception as exc:
            print(f"[DEBUG] Model request failed (attempt {attempt+1}): {exc}", flush=True)
            if attempt < max_retries - 1:
                import time
                time.sleep((attempt + 1) * 2)
            else:
                raise
    return '{"action_type": "inspect_gradients"}'


def parse_action(raw: str) -> str:
    """Clean up LLM output to extract JSON action string."""
    text = raw.strip().strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return '{"action_type": "inspect_gradients"}'


async def run_task(env: GenericEnvClient, client: OpenAI, task_id: str) -> None:
    """Run a single task episode with [START]/[END] logging."""
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id, seed=42)
        obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs_summary = _build_obs_summary(obs)
            raw = get_model_message(client, step, obs_summary, last_reward, history)
            action_str = parse_action(raw)

            action = GenericAction(**json.loads(action_str))
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

        # Score: clamp strictly between 0 and 1
        total_reward = sum(rewards)
        score = round(min(max(total_reward, 0.01), 0.99), 2)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)
        score = 0.01

    finally:
        log_end(task=task_id, success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    # Optional: run specific task or all tasks
    target_task = os.getenv("TASK_NAME")
    tasks_to_run = [target_task] if target_task else ALL_TASK_IDS

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] API_KEY={'set' if API_KEY else 'NOT SET'} (source={'API_KEY' if os.getenv('API_KEY') else 'HF_TOKEN' if os.getenv('HF_TOKEN') else 'NONE'})", flush=True)
    print(f"[DEBUG] Tasks to run: {tasks_to_run}", flush=True)

    completed_tasks: set = set()
    env = None
    try:
        if IMAGE_NAME:
            env = await GenericEnvClient.from_docker_image(IMAGE_NAME)
        else:
            env = GenericEnvClient(
                base_url=ENV_URL,
                message_timeout_s=120.0,
            )
            await env.connect()

        for task_id in tasks_to_run:
            await run_task(env, client, task_id)
            completed_tasks.add(task_id)

    except Exception as exc:
        print(f"[DEBUG] Fatal error: {exc}", flush=True)

    finally:
        # Emit [START]/[END] for any tasks that didn't run (e.g. env connection failed)
        for task_id in tasks_to_run:
            if task_id not in completed_tasks:
                log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
                log_end(task=task_id, success=False, steps=0, score=0.01, rewards=[])
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
