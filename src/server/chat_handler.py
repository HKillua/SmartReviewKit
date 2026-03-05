"""Chat handler — bridges Agent stream events to SSE chunks."""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from src.agent.agent import Agent
from src.server.models import ChatRequest, ChatStreamChunk

logger = logging.getLogger(__name__)


class ChatHandler:
    """Converts Agent.chat() StreamEvent generator into ChatStreamChunk objects."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    async def handle_stream(self, request: ChatRequest) -> AsyncGenerator[ChatStreamChunk, None]:
        async for event in self._agent.chat(
            message=request.message,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        ):
            yield ChatStreamChunk(
                type=event.type.value,
                content=event.content,
                tool_name=event.tool_name,
                metadata=event.metadata,
            )
