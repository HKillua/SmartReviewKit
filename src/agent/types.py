"""Agent-layer data types and contracts.

Defines the fundamental data structures for the Agent system:
- Stream events for SSE output
- LLM request/response types with tool calling support
- Tool execution types (call data, context, result)
- Conversation and message types
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

RoleType = Literal["user", "assistant", "system", "tool"]


# ---------------------------------------------------------------------------
# Stream Events
# ---------------------------------------------------------------------------

class StreamEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    DONE = "done"


class StreamEvent(BaseModel):
    type: StreamEventType
    content: Optional[str] = None
    tool_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool Layer
# ---------------------------------------------------------------------------

class ToolCallData(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolContext(BaseModel):
    user_id: str
    conversation_id: str
    request_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)


class ToolResult(BaseModel):
    success: bool
    result_for_llm: str = ""
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM Layer
# ---------------------------------------------------------------------------

class LlmMessage(BaseModel):
    role: RoleType
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCallData]] = None
    tool_call_id: Optional[str] = None


class LlmRequest(BaseModel):
    messages: list[LlmMessage]
    tools: Optional[list[dict[str, Any]]] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class LlmResponse(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCallData]] = None
    usage: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class LlmStreamChunk(BaseModel):
    delta_content: Optional[str] = None
    delta_tool_calls: Optional[list[dict[str, Any]]] = None
    finish_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Conversation Layer
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Message(BaseModel):
    role: RoleType
    content: Optional[str] = None
    timestamp: datetime = Field(default_factory=_utcnow)
    tool_calls: Optional[list[ToolCallData]] = None
    tool_call_id: Optional[str] = None


class Conversation(BaseModel):
    id: str
    user_id: str
    title: str = ""
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    schema_version: int = 1
