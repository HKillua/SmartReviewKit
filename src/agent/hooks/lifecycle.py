"""Lifecycle hook base class — extension points around message and tool execution."""

from __future__ import annotations

from typing import Optional

from src.agent.types import Conversation, ToolContext, ToolResult


class LifecycleHook:
    """Base class with default pass-through implementations.

    Subclasses override only the hooks they need.
    """

    async def before_message(self, user_id: str, message: str) -> Optional[str]:
        """Called before a user message is processed.

        Return a modified message string, or ``None`` to keep the original.
        """
        return None

    async def after_message(self, conversation: Conversation) -> None:
        """Called after the full response has been generated and saved."""

    async def before_tool(self, tool_name: str, context: ToolContext) -> None:
        """Called before a tool is executed.

        Raise an exception to prevent execution.
        """

    async def after_tool(
        self, tool_name: str, result: ToolResult, context: ToolContext | None = None,
    ) -> Optional[ToolResult]:
        """Called after a tool finishes.

        Return a modified ``ToolResult`` or ``None`` to keep the original.
        """
        return None
