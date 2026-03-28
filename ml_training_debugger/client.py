"""Typed EnvClient for baseline scripts.

Extends GenericEnvClient since we can't easily subclass the
abstract EnvClient without implementing all transport methods.
Used by baseline_heuristic.py.
"""

from __future__ import annotations

from openenv.core.generic_client import GenericEnvClient


class MLTrainingEnvClient(GenericEnvClient):
    """Typed client for the PyTorch Training Debugger environment.

    Wraps GenericEnvClient for convenient use in baselines.
    Actions are sent as dicts matching MLTrainingAction schema.
    Observations are received as dicts matching MLTrainingObservation schema.
    """

    pass
