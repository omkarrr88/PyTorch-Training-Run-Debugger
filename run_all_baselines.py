#!/usr/bin/env python3
"""Run heuristic + multiple LLM baselines and show comparison table.

Usage:
    python3 run_all_baselines.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load .env
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from baseline_heuristic import ALL_TASKS
from baseline_heuristic import run_heuristic_episode
from baseline_inference import PROVIDERS, run_llm_episode

try:
    from openai import OpenAI
except ImportError:
    print("Error: pip install openai")
    sys.exit(1)


def run_heuristic() -> dict[str, float]:
    scores = {}
    for task_id in ALL_TASKS:
        scores[task_id] = round(run_heuristic_episode(task_id), 4)
    return scores


def run_llm_provider(provider_name: str, model: str | None = None) -> dict[str, float]:
    prov = PROVIDERS[provider_name]
    api_key = os.environ.get(prov["env_key"])
    if not api_key:
        return {t: -1.0 for t in ALL_TASKS}  # -1 = no key

    model_name = model or prov["default_model"]
    client_kwargs: dict = {"api_key": api_key}
    if prov["base_url"]:
        client_kwargs["base_url"] = prov["base_url"]
    client = OpenAI(**client_kwargs)

    scores: dict[str, float] = {}
    for task_id in ALL_TASKS:
        try:
            score = run_llm_episode(task_id, client, model_name)
            scores[task_id] = round(score, 4)
            print(f"  [{provider_name}/{model_name}] {task_id}: {score:.4f}", file=sys.stderr)
        except Exception as e:
            err_str = str(e)[:80]
            print(f"  [{provider_name}/{model_name}] {task_id}: ERROR — {err_str}", file=sys.stderr)
            scores[task_id] = 0.0
    return scores


def main() -> None:
    print("Running all baselines...\n", file=sys.stderr)

    results: dict[str, dict[str, float]] = {}

    # Run heuristic first (fast, deterministic)
    print("--- Heuristic baseline ---", file=sys.stderr)
    results["Heuristic"] = run_heuristic()
    print(f"  Done: {json.dumps(results['Heuristic'])}", file=sys.stderr)

    # Run LLM providers sequentially (avoids thread hang issues)
    llm_runs = [
        ("Cerebras/Llama-3.1-8B", "cerebras", "llama3.1-8b"),
        ("Groq/Llama-3.1-8B", "groq", "llama-3.1-8b-instant"),
    ]

    for label, provider, model in llm_runs:
        print(f"\n--- {label} ---", file=sys.stderr)
        try:
            results[label] = run_llm_provider(provider, model)
        except Exception as e:
            print(f"  {label}: FAILED — {e}", file=sys.stderr)
            results[label] = {t: 0.0 for t in ALL_TASKS}

    # Print comparison table
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON TABLE")
    print("=" * 80)

    headers = list(results.keys())
    print(f"\n{'Task':<12}", end="")
    for h in headers:
        print(f"{h:>25}", end="")
    print()
    print("-" * (12 + 25 * len(headers)))

    for task_id in ALL_TASKS:
        print(f"{task_id:<12}", end="")
        for h in headers:
            score = results[h].get(task_id, 0.0)
            if score < 0:
                print(f"{'no key':>25}", end="")
            else:
                print(f"{score:>25.4f}", end="")
        print()

    print("-" * (12 + 25 * len(headers)))

    # Averages
    print(f"{'AVERAGE':<12}", end="")
    for h in headers:
        valid = [v for v in results[h].values() if v >= 0]
        avg = sum(valid) / len(valid) if valid else 0
        print(f"{avg:>25.4f}", end="")
    print()

    # Save JSON
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
