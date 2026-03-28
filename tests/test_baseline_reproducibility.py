"""Test baseline produces bit-exact identical scores on two runs."""

from __future__ import annotations

from baseline_heuristic import ALL_TASKS, run_heuristic_episode


class TestBaselineReproducibility:
    def test_two_runs_identical(self):
        """Run baseline twice, verify bit-exact same scores."""
        run1 = {tid: run_heuristic_episode(tid) for tid in ALL_TASKS}
        run2 = {tid: run_heuristic_episode(tid) for tid in ALL_TASKS}
        assert run1 == run2

    def test_all_scores_in_range(self):
        """All scores must be in [0.0, 1.0]."""
        for tid in ALL_TASKS:
            score = run_heuristic_episode(tid)
            assert 0.0 <= score <= 1.0, f"{tid}: score {score} out of range"

    def test_scores_have_meaningful_variance(self):
        """Not all tasks should return the same score."""
        scores = [run_heuristic_episode(tid) for tid in ALL_TASKS]
        assert len(set(scores)) > 1, "All scores identical — no variance"
