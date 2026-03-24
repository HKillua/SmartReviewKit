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


class ToolErrorType(str, Enum):
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    BLOCKED_BY_HOOK = "blocked_by_hook"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    TOOL_REPORTED_ERROR = "tool_reported_error"


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
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    id: str
    user_id: str
    title: str = ""
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = 2


class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    WAITING_USER = "waiting_user"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"


class RequestStatus(str, Enum):
    ACTIVE = "active"
    WAITING_USER = "waiting_user"
    CLARIFICATION_REQUIRED = "clarification_required"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class AgendaGoal(BaseModel):
    goal_id: str
    intent: str
    selected_tool: str
    segment_text: str = ""
    required: bool = True
    depends_on_user_input: bool = False
    status: GoalStatus = GoalStatus.PENDING
    result_summary: str = ""
    match_method: str = ""
    source_span: list[int] = Field(default_factory=list)


class AgendaState(BaseModel):
    request_status: RequestStatus = RequestStatus.ACTIVE
    current_goal_index: int = 0
    matched_skill: str = ""
    skill_start_index: int = -1
    skill_end_index: int = -1
    goals: list[AgendaGoal] = Field(default_factory=list)
    resume_payload: dict[str, Any] = Field(default_factory=dict)
