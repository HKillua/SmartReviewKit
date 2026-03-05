"""Agent configuration model and loader.

Loads the ``agent`` section from settings.yaml into a validated Pydantic model.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Top-level agent configuration."""

    max_tool_iterations: int = Field(default=10, ge=1, le=50)
    stream_responses: bool = True
    tool_timeout: int = Field(default=30, ge=1)
    max_context_messages: int = Field(default=40, ge=5)
    max_context_tokens: int = Field(default=8000, ge=1000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    system_prompt_path: str = "config/prompts/system_prompt.txt"
    skills_dir: str = "src/agent/skills/definitions"
    memory_enabled: bool = True
    conversation_store_dir: str = "data/conversations"
    default_collection: str = "computer_network"
    auto_ingest_dir: str = ""


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_upload_size_mb: int = Field(default=50, ge=1)
    upload_dir: str = "data/uploads"


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    enabled: bool = True
    db_dir: str = "data/memory"
    profile_enabled: bool = True
    error_memory_enabled: bool = True
    knowledge_map_enabled: bool = True
    skill_memory_enabled: bool = True
    decay_interval_hours: int = Field(default=24, ge=1)


def load_agent_config(settings: dict[str, Any]) -> AgentConfig:
    """Create AgentConfig from the ``agent`` section of settings dict."""
    return AgentConfig(**(settings.get("agent") or {}))


def load_server_config(settings: dict[str, Any]) -> ServerConfig:
    """Create ServerConfig from the ``server`` section of settings dict."""
    return ServerConfig(**(settings.get("server") or {}))


def load_memory_config(settings: dict[str, Any]) -> MemoryConfig:
    """Create MemoryConfig from the ``memory`` section of settings dict."""
    return MemoryConfig(**(settings.get("memory") or {}))
