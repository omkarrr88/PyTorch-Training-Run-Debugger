"""Tests for MLTrainingEnvClient."""

from ml_training_debugger.client import MLTrainingEnvClient


class TestMLTrainingEnvClient:
    def test_can_instantiate(self) -> None:
        """Client class imports and instantiates without error."""
        client = MLTrainingEnvClient(base_url="http://localhost:7860")
        assert client is not None

    def test_is_generic_env_client(self) -> None:
        from openenv.core.generic_client import GenericEnvClient

        assert issubclass(MLTrainingEnvClient, GenericEnvClient)
