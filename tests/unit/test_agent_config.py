"""Unit tests for src/agent/config.py."""

import pytest
from pydantic import ValidationError

from src.agent.config import AgentConfig, MemoryConfig, ServerConfig, load_agent_config


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.max_tool_iterations == 10
        assert cfg.temperature == 0.7
        assert cfg.memory_enabled is True

    def test_custom_values(self):
        cfg = AgentConfig(max_tool_iterations=5, temperature=0.3)
        assert cfg.max_tool_iterations == 5
        assert cfg.temperature == 0.3

    def test_invalid_temperature(self):
        with pytest.raises(ValidationError):
            AgentConfig(temperature=3.0)

    def test_invalid_iterations(self):
        with pytest.raises(ValidationError):
            AgentConfig(max_tool_iterations=0)


class TestLoadAgentConfig:
    def test_from_dict(self):
        settings = {"agent": {"max_tool_iterations": 15, "stream_responses": False}}
        cfg = load_agent_config(settings)
        assert cfg.max_tool_iterations == 15
        assert cfg.stream_responses is False

    def test_missing_section(self):
        cfg = load_agent_config({})
        assert cfg.max_tool_iterations == 10

    def test_empty_section(self):
        cfg = load_agent_config({"agent": None})
        assert cfg.temperature == 0.7


class TestServerConfig:
    def test_defaults(self):
        cfg = ServerConfig()
        assert cfg.port == 8000
        assert "*" in cfg.cors_origins


class TestMemoryConfig:
    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.enabled is True
        assert cfg.decay_interval_hours == 24
