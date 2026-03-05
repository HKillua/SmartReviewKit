"""Unit tests for src/agent/types.py."""

import pytest
from pydantic import ValidationError

from src.agent.types import (
    Conversation,
    LlmMessage,
    LlmRequest,
    LlmResponse,
    LlmStreamChunk,
    Message,
    StreamEvent,
    StreamEventType,
    ToolCallData,
    ToolContext,
    ToolResult,
)


class TestStreamEvent:
    def test_create_text_delta(self):
        e = StreamEvent(type=StreamEventType.TEXT_DELTA, content="hello")
        assert e.type == StreamEventType.TEXT_DELTA
        assert e.content == "hello"

    def test_serialization(self):
        e = StreamEvent(type=StreamEventType.DONE, metadata={"k": "v"})
        data = e.model_dump()
        assert data["type"] == "done"
        reconstructed = StreamEvent.model_validate(data)
        assert reconstructed.type == StreamEventType.DONE


class TestToolCallData:
    def test_create(self):
        tc = ToolCallData(id="tc_1", name="knowledge_query", arguments={"query": "SQL"})
        assert tc.name == "knowledge_query"
        assert tc.arguments["query"] == "SQL"


class TestToolResult:
    def test_success(self):
        r = ToolResult(success=True, result_for_llm="found 5 results")
        assert r.success is True

    def test_failure(self):
        r = ToolResult(success=False, error="not found")
        assert r.success is False
        assert r.error == "not found"


class TestLlmMessage:
    def test_user_message(self):
        m = LlmMessage(role="user", content="hello")
        assert m.role == "user"
        assert m.tool_calls is None

    def test_assistant_with_tool_calls(self):
        tc = ToolCallData(id="1", name="tool", arguments={})
        m = LlmMessage(role="assistant", tool_calls=[tc])
        assert len(m.tool_calls) == 1


class TestLlmRequest:
    def test_defaults(self):
        r = LlmRequest(messages=[LlmMessage(role="user", content="hi")])
        assert r.temperature == 0.7
        assert r.stream is False

    def test_with_tools(self):
        r = LlmRequest(
            messages=[LlmMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "t"}}],
        )
        assert len(r.tools) == 1


class TestConversation:
    def test_create(self):
        c = Conversation(id="conv1", user_id="user1")
        assert c.id == "conv1"
        assert len(c.messages) == 0

    def test_json_round_trip(self):
        c = Conversation(id="conv1", user_id="user1")
        c.messages.append(Message(role="user", content="hello"))
        json_str = c.model_dump_json()
        restored = Conversation.model_validate_json(json_str)
        assert restored.id == "conv1"
        assert len(restored.messages) == 1
