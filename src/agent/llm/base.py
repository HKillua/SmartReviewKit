"""LLM service abstract base class with tool-calling and streaming support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from src.agent.types import LlmRequest, LlmResponse, LlmStreamChunk


class LlmService(ABC):
    """Abstract interface for LLM providers that support tool calling and streaming.

    Concrete implementations handle the translation between our internal types
    and provider-specific API formats (OpenAI, Azure, DeepSeek, Ollama, etc.).
    """

    @abstractmethod
    async def send_request(self, request: LlmRequest) -> LlmResponse:
        """Send a non-streaming chat completion request."""
        ...

    @abstractmethod
    async def stream_request(self, request: LlmRequest) -> AsyncGenerator[LlmStreamChunk, None]:
        """Send a streaming chat completion request, yielding chunks."""
        ...
        yield  # pragma: no cover — makes this a valid async generator stub

    def validate_tools(self, tools: list[dict]) -> list[str]:
        """Validate tool schemas for provider compatibility.

        Returns a list of error messages (empty list means all valid).
        """
        errors: list[str] = []
        for i, tool in enumerate(tools):
            if tool.get("type") != "function":
                errors.append(f"Tool[{i}]: 'type' must be 'function', got '{tool.get('type')}'")
            fn = tool.get("function", {})
            if not fn.get("name"):
                errors.append(f"Tool[{i}]: 'function.name' is required")
            if "parameters" in fn and not isinstance(fn["parameters"], dict):
                errors.append(f"Tool[{i}]: 'function.parameters' must be a dict")
        return errors
