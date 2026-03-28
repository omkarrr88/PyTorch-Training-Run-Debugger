"""Shared test fixtures."""

from __future__ import annotations

import pytest

from ml_training_debugger.models import (
    EpisodeState,
    TrainingConfig,
)
from ml_training_debugger.scenarios import ScenarioParams, sample_scenario


@pytest.fixture
def fresh_state() -> EpisodeState:
    return EpisodeState()


@pytest.fixture
def sample_config() -> TrainingConfig:
    return TrainingConfig(learning_rate=0.001)


@pytest.fixture
def task_001_scenario() -> ScenarioParams:
    return sample_scenario("task_001", seed=42)


@pytest.fixture
def task_003_scenario() -> ScenarioParams:
    return sample_scenario("task_003", seed=42)


@pytest.fixture
def task_005_scenario() -> ScenarioParams:
    return sample_scenario("task_005", seed=42)
