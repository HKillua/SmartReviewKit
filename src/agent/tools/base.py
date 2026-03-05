"""Tool abstract base class and ToolRegistry.

Follows the Vanna.ai pattern: every tool is a ``Tool[T]`` generic with a
Pydantic model ``T`` that describes the arguments.  The ToolRegistry handles
registration, schema generation (for LLM function-calling), dispatch, and
error wrapping.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from src.agent.types import ToolCallData, ToolContext, ToolResult

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Tool(ABC, Generic[T]):
    """Abstract base for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def get_args_schema(self) -> type[T]:
        """Return the Pydantic model class that describes the tool arguments."""
        ...

    @abstractmethod
    async def execute(self, context: ToolContext, args: T) -> ToolResult:
        ...

    def get_schema(self) -> dict[str, Any]:
        """Generate an OpenAI-compatible function tool schema."""
        schema_cls = self.get_args_schema()
        json_schema = schema_cls.model_json_schema()
        json_schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": json_schema,
            },
        }


class ToolRegistry:
    """Central registry for Agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get_tool(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling schemas for all registered tools."""
        return [t.get_schema() for t in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(
        self,
        tool_call: ToolCallData,
        context: ToolContext,
        *,
        timeout: float = 30.0,
    ) -> ToolResult:
        """Look up, validate, and execute a tool call.

        Any exception is caught and wrapped in a ``ToolResult(success=False)``.
        """
        try:
            tool = self.get_tool(tool_call.name)
        except KeyError as exc:
            return ToolResult(success=False, error=str(exc))

        try:
            schema_cls = tool.get_args_schema()
            args = schema_cls.model_validate(tool_call.arguments)
        except ValidationError as exc:
            return ToolResult(
                success=False,
                error=f"Invalid arguments for '{tool_call.name}': {exc}",
            )

        try:
            return await asyncio.wait_for(
                tool.execute(context, args),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_call.name}' timed out after {timeout}s",
            )
        except Exception as exc:
            logger.exception("Tool '%s' execution failed", tool_call.name)
            return ToolResult(success=False, error=f"Tool execution error: {exc}")
