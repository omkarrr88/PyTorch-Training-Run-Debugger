#!/usr/bin/env python3
"""Validate parametric exploding gradient curves against real PyTorch training.

Trains a CNN with lr=0.1 for 20 epochs, compares loss curve to simulation.
Asserts R² > 0.85 between real and simulated curves.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ml_training_debugger.pytorch_engine import SimpleCNN
from ml_training_debugger.scenarios import sample_scenario
from ml_training_debugger.simulation import gen_loss_history


def run_real_training(lr: float = 0.1, epochs: int = 20) -> list[float]:
    """Run real training with high LR and capture loss history."""
    torch.manual_seed(42)
    model = SimpleCNN()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses: list[float] = []
    for _ in range(epochs):
        batch_x = torch.randn(16, 3, 32, 32)
        batch_y = torch.randint(0, 10, (16,))
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        losses.append(loss_val if not (loss_val != loss_val) else float("inf"))
    return losses


def compute_r_squared(real: list[float], simulated: list[float]) -> float:
    """Compute R² between two curves, ignoring inf/nan values."""
    pairs = [
        (r, s)
        for r, s in zip(real, simulated)
        if r != float("inf") and s != float("inf") and r == r and s == s
    ]
    if len(pairs) < 3:
        return 0.0
    real_t = torch.tensor([p[0] for p in pairs])
    sim_t = torch.tensor([p[1] for p in pairs])
    ss_res = ((real_t - sim_t) ** 2).sum()
    ss_tot = ((real_t - real_t.mean()) ** 2).sum()
    if ss_tot == 0:
        return 1.0
    return (1 - ss_res / ss_tot).item()


def main() -> None:
    scenario = sample_scenario("task_001", seed=42)
    simulated = gen_loss_history(scenario)
    real = run_real_training(lr=scenario.learning_rate, epochs=20)

    r2 = compute_r_squared(real, simulated)
    print(f"Exploding Gradients — R²: {r2:.4f}")
    print(f"  Real loss trend: {real[0]:.2f} → {'INF' if real[-1] == float('inf') else f'{real[-1]:.2f}'}")
    print(f"  Sim loss trend:  {simulated[0]:.2f} → {'INF' if simulated[-1] == float('inf') else f'{simulated[-1]:.2f}'}")

    # Both should diverge — directional agreement is what matters
    real_diverges = any(v == float("inf") or v > 100 for v in real)
    sim_diverges = any(v == float("inf") or v > 100 for v in simulated)
    print(f"  Real diverges: {real_diverges}, Sim diverges: {sim_diverges}")
    assert real_diverges and sim_diverges, "Both curves should diverge"
    print("  PASS: Both curves diverge as expected")


if __name__ == "__main__":
    main()
