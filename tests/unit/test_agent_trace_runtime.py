"""Unit tests for agent trace emission and knowledge-query trace linkage."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from src.agent.types import Conversation, LlmResponse, ToolCallData
from src.core.trace.trace_collector import TraceCollector
from src.core.types import RetrievalResult
from src.observability.dashboard.services.trace_service import TraceService


class _ToolArgs(BaseModel):
    query: str = Field(default="")
    collection: str = Field(default="computer_network")


class _TraceAwareTool:
    @property
    def name(self) -> str:
        return "knowledge_query"

    @property
    def description(self) -> str:
        return "fake knowledge query tool"

    def get_args_schema(self) -> type[_ToolArgs]:
        return _ToolArgs

    async def execute(self, context, args):
        from src.agent.types import ToolResult

        return ToolResult(
            success=True,
            result_for_llm="retrieved TCP knowledge",
            metadata={"query_trace_id": "query-trace-123"},
        )

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_args_schema().model_json_schema(),
            },
        }


class _ToolThenAnswerLlm:
    def __init__(self) -> None:
        self.calls = 0

    async def send_request(self, request):
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(
                content="Let me look that up.",
                tool_calls=[
                    ToolCallData(
                        id="tc_1",
                        name="knowledge_query",
                        arguments={"query": "TCP", "collection": "computer_network"},
                    )
                ],
            )
        return LlmResponse(content="TCP uses a three-way handshake.")


class _AlwaysToolLlm:
    async def send_request(self, request):
        return LlmResponse(
            content="Need tool again",
            tool_calls=[
                ToolCallData(
                    id="tc_1",
                    name="knowledge_query",
                    arguments={"query": "TCP", "collection": "computer_network"},
                )
            ],
        )


class _PlannerForceLlm:
    def __init__(self) -> None:
        self.seen_tools = []
        self.calls = 0

    async def send_request(self, request):
        self.calls += 1
        self.seen_tools.append(
            [tool.get("function", {}).get("name") for tool in request.tools or []]
        )
        if self.calls == 1:
            return LlmResponse(
                content="Need review summary",
                tool_calls=[
                    ToolCallData(
                        id="tc_1",
                        name="review_summary",
                        arguments={"topic": "DNS"},
                    )
                ],
            )
        return LlmResponse(content="Here is the final review summary.")


class _PlannerViolationLlm:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_tools = []

    async def send_request(self, request):
        self.calls += 1
        self.seen_tools.append(
            [tool.get("function", {}).get("name") for tool in request.tools or []]
        )
        if self.calls <= 2:
            return LlmResponse(content="I will answer directly without tools.")
        if self.calls == 3:
            return LlmResponse(
                content="Now I will use a tool.",
                tool_calls=[
                    ToolCallData(
                        id="tc_1",
                        name="review_summary",
                        arguments={"topic": "DNS"},
                    )
                ],
            )
        return LlmResponse(content="Done after downgrade.")


class _ReviewSummaryTool:
    @property
    def name(self) -> str:
        return "review_summary"

    @property
    def description(self) -> str:
        return "fake review summary tool"

    def get_args_schema(self):
        class _Args(BaseModel):
            topic: str = Field(default="")

        return _Args

    async def execute(self, context, args):
        from src.agent.types import ToolResult

        return ToolResult(success=True, result_for_llm=f"summary for {args.topic}")

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_args_schema().model_json_schema(),
            },
        }


class _DummyKnowledgeTool(_TraceAwareTool):
    pass


class _KnowledgePlanner:
    def plan(self, message, matched_skill=None):
        from src.agent.planner import ControlMode, PlannerDecision, TaskIntent

        return PlannerDecision(
            task_intent=TaskIntent.KNOWLEDGE_QUERY,
            confidence=0.95,
            match_method="rule",
            control_mode=ControlMode.ADVISORY,
            selected_tool="knowledge_query",
            planner_hint="Use course evidence.",
        )


class _GroundedTool(_TraceAwareTool):
    async def execute(self, context, args):
        from src.agent.types import ToolResult

        return ToolResult(
            success=True,
            result_for_llm="检索到 1 条相关内容：\n[1] `network.md`\nTCP 通过三次握手建立连接。",
            metadata={
                "grounding_capable": True,
                "query_trace_id": "query-trace-123",
                "query_trace_ids": ["query-trace-123"],
                "source_count": 1,
                "citations": [
                    {
                        "index": 1,
                        "source": "network.md",
                        "text_snippet": "TCP 通过三次握手建立连接。",
                    }
                ],
                "evidence_summary": "[1] `network.md`: TCP 通过三次握手建立连接。",
            },
        )


class _GroundedAnswerLlm(_ToolThenAnswerLlm):
    async def send_request(self, request):
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(
                content="先查知识库。",
                tool_calls=[
                    ToolCallData(
                        id="tc_1",
                        name="knowledge_query",
                        arguments={"query": "TCP", "collection": "computer_network"},
                    )
                ],
            )
        return LlmResponse(content="TCP 通过三次握手建立连接 [1]。")


@pytest.mark.asyncio
async def test_agent_done_event_contains_trace_id_and_agent_trace(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.tools.base import ToolRegistry
    from src.agent.conversation import ConversationStore

    trace_path = tmp_path / "traces.jsonl"
    collector = TraceCollector(traces_path=trace_path)

    conversation = Conversation(id="conv_1", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    registry = ToolRegistry()
    registry.register(_TraceAwareTool())

    agent = Agent(
        llm_service=_ToolThenAnswerLlm(),
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=2),
        trace_enabled=True,
        trace_collector=collector,
    )

    events = [event async for event in agent.chat("Explain TCP", "u1", "conv_1")]

    done_event = next(event for event in events if event.type.value == "done")
    tool_result_event = next(event for event in events if event.type.value == "tool_result")
    assert done_event.metadata["trace_id"]
    assert tool_result_event.metadata["query_trace_id"] == "query-trace-123"
    assert tool_result_event.metadata["tool_name"] == "knowledge_query"
    assert tool_result_event.metadata["retryable"] is False

    service = TraceService(trace_path)
    trace = service.get_trace(done_event.metadata["trace_id"])
    assert trace is not None
    assert trace["trace_type"] == "agent"
    assert trace["metadata"]["status"] == "success"
    stage_names = [stage["stage"] for stage in trace["stages"]]
    assert "llm_iteration" in stage_names
    assert "tool_execution" in stage_names
    assert "final_response" in stage_names
    tool_stage = next(stage for stage in trace["stages"] if stage["stage"] == "tool_execution")
    assert tool_stage["data"]["retryable"] is False


@pytest.mark.asyncio
async def test_agent_done_event_contains_grounding_metadata(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.tools.base import ToolRegistry

    trace_path = tmp_path / "grounding_traces.jsonl"
    collector = TraceCollector(traces_path=trace_path)

    conversation = Conversation(id="conv_grounding", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    registry = ToolRegistry()
    registry.register(_GroundedTool())

    agent = Agent(
        llm_service=_GroundedAnswerLlm(),
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=2),
        task_planner=_KnowledgePlanner(),
        trace_enabled=True,
        trace_collector=collector,
    )

    events = [event async for event in agent.chat("Explain TCP", "u1", "conv_grounding")]

    done_event = next(event for event in events if event.type.value == "done")
    assert done_event.metadata["has_evidence"] is True
    assert done_event.metadata["grounding_score"] >= 0.4
    assert done_event.metadata["grounding_policy_action"] == "normal"
    assert len(done_event.metadata["citations"]) == 1

    trace = TraceService(trace_path).get_trace(done_event.metadata["trace_id"])
    assert trace is not None
    stage_names = [stage["stage"] for stage in trace["stages"]]
    assert "answer_grounding" in stage_names


@pytest.mark.asyncio
async def test_agent_trace_records_max_iteration_error(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.tools.base import ToolRegistry
    from src.agent.conversation import ConversationStore

    trace_path = tmp_path / "traces.jsonl"
    collector = TraceCollector(traces_path=trace_path)

    conversation = Conversation(id="conv_2", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    registry = ToolRegistry()
    registry.register(_TraceAwareTool())

    agent = Agent(
        llm_service=_AlwaysToolLlm(),
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=1),
        trace_enabled=True,
        trace_collector=collector,
    )

    events = [event async for event in agent.chat("Explain TCP", "u1", "conv_2")]
    error_event = next(event for event in events if event.type.value == "error")
    done_event = next(event for event in events if event.type.value == "done")

    assert "最大工具调用轮次" in (error_event.content or "")

    trace = TraceService(trace_path).get_trace(done_event.metadata["trace_id"])
    assert trace is not None
    assert trace["metadata"]["status"] == "error"
    tool_stage = next(stage for stage in trace["stages"] if stage["stage"] == "tool_execution")
    assert tool_stage["data"]["error_type"] == ""
    error_stages = [stage for stage in trace["stages"] if stage["stage"] == "error"]
    assert error_stages
    assert "max_iterations_exceeded" in error_stages[-1]["data"]["error"]


@pytest.mark.asyncio
async def test_knowledge_query_returns_query_trace_id_and_parent_link(tmp_path: Path) -> None:
    from src.agent.tools.knowledge_query import KnowledgeQueryArgs, KnowledgeQueryTool
    from src.agent.types import ToolContext

    trace_path = tmp_path / "traces.jsonl"
    hybrid_search = MagicMock()
    hybrid_search.search.return_value = [
        RetrievalResult(
            chunk_id="chunk_1",
            score=0.95,
            text="TCP uses SYN, SYN-ACK, ACK.",
            metadata={"source_path": "network.md", "title": "TCP"},
        )
    ]

    tool = KnowledgeQueryTool(
        settings={"observability": {"trace_enabled": True}},
        hybrid_search=hybrid_search,
    )
    tool._current_collection = "computer_network"
    tool._trace_collector = TraceCollector(traces_path=trace_path)

    result = await tool.execute(
        ToolContext(
            user_id="u1",
            conversation_id="conv_1",
            metadata={"agent_trace_id": "agent-trace-xyz"},
        ),
        KnowledgeQueryArgs(query="TCP", collection="computer_network"),
    )

    query_trace_id = result.metadata["query_trace_id"]
    assert result.metadata["grounding_capable"] is True
    assert len(result.metadata["citations"]) == 1
    assert "TCP uses SYN" in result.metadata["evidence_summary"]
    trace = TraceService(trace_path).get_trace(query_trace_id)
    assert trace is not None
    assert trace["trace_type"] == "query"
    assert trace["metadata"]["source"] == "agent"
    assert trace["metadata"]["parent_agent_trace_id"] == "agent-trace-xyz"


@pytest.mark.asyncio
async def test_force_tool_first_round_only_exposes_selected_tool_schema(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.planner import ControlMode, PlannerDecision, TaskIntent
    from src.agent.tools.base import ToolRegistry

    class _StubPlanner:
        def plan(self, message, matched_skill=None):
            return PlannerDecision(
                task_intent=TaskIntent.REVIEW_SUMMARY,
                confidence=1.0,
                match_method="rule",
                control_mode=ControlMode.FORCE_TOOL,
                selected_tool="review_summary",
                planner_hint="Use review_summary first.",
            )

    trace_path = tmp_path / "planner_force.jsonl"
    collector = TraceCollector(traces_path=trace_path)
    conversation = Conversation(id="conv_force", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    registry = ToolRegistry()
    registry.register(_ReviewSummaryTool())
    registry.register(_DummyKnowledgeTool())

    llm = _PlannerForceLlm()
    agent = Agent(
        llm_service=llm,
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=3),
        task_planner=_StubPlanner(),
        trace_enabled=True,
        trace_collector=collector,
    )

    events = [event async for event in agent.chat("帮我复习 DNS", "u1", "conv_force")]
    done_event = next(event for event in events if event.type.value == "done")

    assert llm.seen_tools[0] == ["review_summary"]
    assert set(llm.seen_tools[1]) == {"review_summary", "knowledge_query"}
    assert done_event.metadata["planner_task_intent"] == "review_summary"
    assert done_event.metadata["planner_control_mode"] == "force_tool"
    trace = TraceService(trace_path).get_trace(done_event.metadata["trace_id"])
    assert trace is not None
    planner_stage = next(stage for stage in trace["stages"] if stage["stage"] == "planner_decision")
    assert planner_stage["data"]["selected_tool"] == "review_summary"


@pytest.mark.asyncio
async def test_planner_violation_retries_then_downgrades_to_advisory(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.planner import ControlMode, PlannerDecision, TaskIntent
    from src.agent.tools.base import ToolRegistry

    class _StubPlanner:
        def plan(self, message, matched_skill=None):
            return PlannerDecision(
                task_intent=TaskIntent.REVIEW_SUMMARY,
                confidence=1.0,
                match_method="rule",
                control_mode=ControlMode.FORCE_TOOL,
                selected_tool="review_summary",
                planner_hint="Use review_summary first.",
            )

    trace_path = tmp_path / "planner_violation.jsonl"
    collector = TraceCollector(traces_path=trace_path)
    conversation = Conversation(id="conv_violation", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    registry = ToolRegistry()
    registry.register(_ReviewSummaryTool())
    registry.register(_DummyKnowledgeTool())

    llm = _PlannerViolationLlm()
    agent = Agent(
        llm_service=llm,
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=4),
        task_planner=_StubPlanner(),
        trace_enabled=True,
        trace_collector=collector,
    )

    events = [event async for event in agent.chat("帮我复习 DNS", "u1", "conv_violation")]
    done_event = next(event for event in events if event.type.value == "done")

    assert llm.seen_tools[0] == ["review_summary"]
    assert llm.seen_tools[1] == ["review_summary"]
    assert set(llm.seen_tools[2]) == {"review_summary", "knowledge_query"}
    assert done_event.metadata["planner_final_control_mode"] == "advisory"
    assert done_event.metadata["planner_violation_count"] == 2

    trace = TraceService(trace_path).get_trace(done_event.metadata["trace_id"])
    assert trace is not None
    planner_violations = [
        stage for stage in trace["stages"]
        if stage["stage"] == "planner_violation"
    ]
    assert len(planner_violations) >= 2
