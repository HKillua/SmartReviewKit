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


class _DocumentIngestTool:
    @property
    def name(self) -> str:
        return "document_ingest"

    @property
    def description(self) -> str:
        return "fake document ingest tool"

    def get_args_schema(self):
        class _Args(BaseModel):
            file_path: str = Field(default="")
            collection: str = Field(default="computer_network")

        return _Args

    async def execute(self, context, args):
        from src.agent.types import ToolResult

        return ToolResult(success=True, result_for_llm="queued")

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_args_schema().model_json_schema(),
            },
        }


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


class _ReviewPassthroughTool:
    @property
    def name(self) -> str:
        return "review_summary"

    @property
    def description(self) -> str:
        return "review summary passthrough tool"

    def get_args_schema(self):
        class _Args(BaseModel):
            topic: str = Field(default="")

        return _Args

    async def execute(self, context, args):
        from src.agent.types import ToolResult

        return ToolResult(
            success=True,
            result_for_llm="### 复习摘要\n- DNS 解析包含递归查询 [1]。",
            metadata={
                "grounding_capable": True,
                "grounding_passthrough": True,
                "final_response_preferred": True,
                "source_count": 1,
                "citations": [
                    {"index": 1, "source": "dns.pdf", "text_snippet": "DNS 解析包含递归查询。"}
                ],
                "evidence_summary": "[1] `dns.pdf`: DNS 解析包含递归查询。",
            },
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


class _CaptureVisibleToolsLlm:
    def __init__(self) -> None:
        self.seen_tools: list[list[str]] = []

    async def send_request(self, request):
        self.seen_tools.append(
            [tool.get("function", {}).get("name") for tool in request.tools or []]
        )
        return LlmResponse(content="直接回答。")

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_args_schema().model_json_schema(),
            },
        }


class _ReviewPlanner:
    def plan(self, message, matched_skill=None):
        from src.agent.planner import ControlMode, PlannerDecision, TaskIntent

        return PlannerDecision(
            task_intent=TaskIntent.REVIEW_SUMMARY,
            confidence=1.0,
            match_method="rule",
            control_mode=ControlMode.FORCE_TOOL,
            selected_tool="review_summary",
            planner_hint="Use review summary first.",
        )


class _ReviewOnlyLlm:
    def __init__(self) -> None:
        self.calls = 0

    async def send_request(self, request):
        self.calls += 1
        return LlmResponse(
            content="Need summary",
            tool_calls=[
                ToolCallData(
                    id="tc_1",
                    name="review_summary",
                    arguments={"topic": "DNS"},
                )
            ],
        )


class _NoLlmExpected:
    def __init__(self) -> None:
        self.calls = 0

    async def send_request(self, request):
        self.calls += 1
        raise AssertionError("Composite execution should not call the LLM tool loop")


class _CompositeReviewTool:
    @property
    def name(self) -> str:
        return "review_summary"

    @property
    def description(self) -> str:
        return "composite review tool"

    def get_args_schema(self):
        class _Args(BaseModel):
            topic: str = Field(default="")

        return _Args

    async def execute(self, context, args):
        from src.agent.types import ToolResult

        assert context.metadata["composite_mode"] is True
        return ToolResult(
            success=True,
            result_for_llm="TCP 的复习重点包括三次握手和可靠传输机制。",
            metadata={
                "grounding_capable": True,
                "final_response_preferred": True,
                "citations": [
                    {"index": 1, "source": "tcp.pdf", "text_snippet": "TCP 包括三次握手。"}
                ],
                "evidence_summary": "[1] `tcp.pdf`: TCP 包括三次握手。",
                "source_count": 1,
            },
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


class _CompositeQuizTool:
    @property
    def name(self) -> str:
        return "quiz_generator"

    @property
    def description(self) -> str:
        return "composite quiz tool"

    def get_args_schema(self):
        class _Args(BaseModel):
            topic: str = Field(default="")
            question_type: str = Field(default="选择题")
            count: int = Field(default=3)
            difficulty: int = Field(default=3)

        return _Args

    async def execute(self, context, args):
        from src.agent.types import ToolResult

        handoff = context.metadata["composite_handoff"]
        assert handoff["latest_result_text"].startswith("TCP 的复习重点")
        return ToolResult(
            success=True,
            result_for_llm="### 第 1 题\n\nTCP 为什么需要三次握手？",
            metadata={
                "grounding_capable": True,
                "final_response_preferred": True,
                "generation_mode": "composite_handoff",
                "citations": [
                    {"index": 1, "source": "tcp.pdf", "text_snippet": "TCP 使用三次握手建立连接。"}
                ],
                "evidence_summary": "[1] `tcp.pdf`: TCP 使用三次握手建立连接。",
                "source_count": 1,
            },
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


class _CompositeQuizFailTool(_CompositeQuizTool):
    async def execute(self, context, args):
        from src.agent.types import ToolResult

        return ToolResult(success=False, error="quiz generation failed")


class _CompositePlanner:
    def __init__(self, *, fail_second: bool = False) -> None:
        self.fail_second = fail_second

    def plan(self, message, matched_skill=None):
        from src.agent.planner import ControlMode, PlannedSubtask, PlannerDecision, TaskIntent

        subtasks = [
            PlannedSubtask(
                task_intent=TaskIntent.REVIEW_SUMMARY,
                selected_tool="review_summary",
                confidence=1.0,
                source_span=(0, 2),
                segment_text="总结 TCP 的重点",
            ),
            PlannedSubtask(
                task_intent=TaskIntent.QUIZ_GENERATOR,
                selected_tool="quiz_generator",
                confidence=1.0,
                source_span=(3, 8),
                segment_text="出 3 道题",
            ),
        ]
        return PlannerDecision(
            task_intent=TaskIntent.REVIEW_SUMMARY,
            confidence=1.0,
            match_method="rule_composite",
            control_mode=ControlMode.FORCE_TOOL,
            selected_tool="review_summary",
            planner_hint="Composite execution",
            is_composite=True,
            subtasks=subtasks,
            primary_intent=TaskIntent.REVIEW_SUMMARY,
            ordering_method="rule_span_order",
        )


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
async def test_grounding_passthrough_tool_skips_conservative_rewrite(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.tools.base import ToolRegistry

    trace_path = tmp_path / "review_passthrough.jsonl"
    collector = TraceCollector(traces_path=trace_path)

    conversation = Conversation(id="conv_review", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    registry = ToolRegistry()
    registry.register(_ReviewPassthroughTool())

    llm = _ReviewOnlyLlm()
    agent = Agent(
        llm_service=llm,
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=2),
        task_planner=_ReviewPlanner(),
        trace_enabled=True,
        trace_collector=collector,
    )

    events = [event async for event in agent.chat("帮我总结 DNS", "u1", "conv_review")]

    done_event = next(event for event in events if event.type.value == "done")
    final_text = "".join(event.content or "" for event in events if event.type.value == "text_delta")
    assert llm.calls == 1
    assert "DNS 解析包含递归查询" in final_text
    assert done_event.metadata["grounding_score"] >= 0.4
    assert done_event.metadata["grounding_policy_action"] == "normal"


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
async def test_knowledge_query_trace_records_retrieval_policy(tmp_path: Path) -> None:
    from src.agent.tools.knowledge_query import KnowledgeQueryArgs, KnowledgeQueryTool
    from src.agent.types import ToolContext
    from src.core.query_engine.query_router import QueryRouter

    trace_path = tmp_path / "traces.jsonl"
    hybrid_search = MagicMock()
    hybrid_search.search.return_value = [
        RetrievalResult(
            chunk_id="chunk_1",
            score=0.95,
            text="TCP 题库练习。",
            metadata={"source_path": "question_bank.md", "title": "TCP练习"},
        )
    ]

    tool = KnowledgeQueryTool(
        settings={"observability": {"trace_enabled": True}},
        hybrid_search=hybrid_search,
        query_router=QueryRouter(),
    )
    tool._current_collection = "computer_network"
    tool._trace_collector = TraceCollector(traces_path=trace_path)

    result = await tool.execute(
        ToolContext(
            user_id="u1",
            conversation_id="conv_1",
            metadata={
                "agent_trace_id": "agent-trace-xyz",
                "planner_task_intent": "knowledge_query",
            },
        ),
        KnowledgeQueryArgs(query="出一道TCP的练习题", collection="computer_network"),
    )

    trace = TraceService(trace_path).get_trace(result.metadata["query_trace_id"])
    assert trace is not None
    assert trace["metadata"]["planner_task_intent"] == "knowledge_query"
    assert "question_bank" in trace["metadata"]["preferred_sources"]
    assert trace["metadata"]["retrieval_strategy"] == "full"
    assert trace["metadata"]["router_match_method"] == "rule"
    policy_stage = next(stage for stage in trace["stages"] if stage["stage"] == "retrieval_policy")
    assert policy_stage["data"]["preferred_sources"][0] == "question_bank"
    assert policy_stage["data"]["method"] == "rule"


@pytest.mark.asyncio
async def test_knowledge_query_trace_records_composite_metadata(tmp_path: Path) -> None:
    from src.agent.tools.knowledge_query import KnowledgeQueryArgs, KnowledgeQueryTool
    from src.agent.types import ToolContext

    trace_path = tmp_path / "composite_query_trace.jsonl"
    hybrid_search = MagicMock()
    hybrid_search.search.return_value = [
        RetrievalResult(
            chunk_id="chunk_1",
            score=0.95,
            text="DNS 是分层命名系统。",
            metadata={"source_path": "dns.pdf", "title": "DNS"},
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
            conversation_id="conv_composite_query",
            request_id="req_composite",
            metadata={
                "agent_trace_id": "agent-trace-xyz",
                "planner_task_intent": "knowledge_query",
                "composite_mode": True,
                "composite_parent_request_id": "req_composite",
                "composite_subtask_index": 0,
                "composite_subtask_intent": "knowledge_query",
            },
        ),
        KnowledgeQueryArgs(query="解释 DNS"),
    )

    trace = TraceService(trace_path).get_trace(result.metadata["query_trace_id"])
    assert trace is not None
    assert trace["metadata"]["composite_parent_request_id"] == "req_composite"
    assert trace["metadata"]["composite_subtask_index"] == 0
    assert trace["metadata"]["composite_subtask_intent"] == "knowledge_query"


@pytest.mark.asyncio
async def test_knowledge_query_uses_default_collection_from_tool_context(tmp_path: Path) -> None:
    from src.agent.tools.knowledge_query import KnowledgeQueryArgs, KnowledgeQueryTool
    from src.agent.types import ToolContext

    class _FakeSparseRetriever:
        default_collection = "api_e2e_test"

    class _FakeHybridSearch:
        def __init__(self) -> None:
            self.sparse_retriever = _FakeSparseRetriever()
            self.seen_queries: list[dict[str, Any]] = []

        def search(self, *, query, top_k, trace=None, **kwargs):
            self.seen_queries.append({"query": query, "kwargs": kwargs})
            return [
                RetrievalResult(
                    chunk_id="chunk_default_collection",
                    score=0.9,
                    text="价格199元",
                    metadata={"source_path": "blogger_intro.pdf", "title": "商品介绍"},
                )
            ]

    hybrid_search = _FakeHybridSearch()
    tool = KnowledgeQueryTool(
        settings={"observability": {"trace_enabled": False}},
        hybrid_search=hybrid_search,
    )

    result = await tool.execute(
        ToolContext(
            user_id="u1",
            conversation_id="conv_default_collection",
            metadata={"default_collection": "api_e2e_test"},
        ),
        KnowledgeQueryArgs(query="价格是多少"),
    )

    assert result.success is True
    assert result.metadata["collection"] == "api_e2e_test"
    assert hybrid_search.seen_queries


@pytest.mark.asyncio
async def test_normal_chat_hides_document_ingest_tool_schema(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.tools.base import ToolRegistry

    conversation = Conversation(id="conv_hide_ingest", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    registry = ToolRegistry()
    registry.register(_DummyKnowledgeTool())
    registry.register(_DocumentIngestTool())

    llm = _CaptureVisibleToolsLlm()
    agent = Agent(
        llm_service=llm,
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=1),
    )

    events = [event async for event in agent.chat("解释一下 TCP 三次握手", "u1", "conv_hide_ingest")]
    assert any(event.type.value == "done" for event in events)
    assert llm.seen_tools
    assert "document_ingest" not in llm.seen_tools[0]
    assert "knowledge_query" in llm.seen_tools[0]


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


@pytest.mark.asyncio
async def test_composite_plan_executes_subtasks_in_order(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.tools.base import ToolRegistry

    trace_path = tmp_path / "composite_trace.jsonl"
    collector = TraceCollector(traces_path=trace_path)
    conversation = Conversation(id="conv_composite", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    registry = ToolRegistry()
    registry.register(_CompositeReviewTool())
    registry.register(_CompositeQuizTool())

    llm = _NoLlmExpected()
    agent = Agent(
        llm_service=llm,
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=3),
        task_planner=_CompositePlanner(),
        trace_enabled=True,
        trace_collector=collector,
    )

    events = [
        event async for event in agent.chat("帮我先总结 TCP 的重点，再出 3 道题", "u1", "conv_composite")
    ]

    tool_starts = [event.tool_name for event in events if event.type.value == "tool_start"]
    assert tool_starts == ["review_summary", "quiz_generator"]
    final_text = "".join(event.content or "" for event in events if event.type.value == "text_delta")
    assert "## 复习总结" in final_text
    assert "## 练习题" in final_text
    done_event = next(event for event in events if event.type.value == "done")
    assert done_event.metadata["composite"] is True
    assert len(done_event.metadata["completed_subtasks"]) == 2
    assert done_event.metadata["failed_subtask"] == {}
    assert llm.calls == 0

    trace = TraceService(trace_path).get_trace(done_event.metadata["trace_id"])
    assert trace is not None
    stage_names = [stage["stage"] for stage in trace["stages"]]
    assert "composite_plan" in stage_names
    assert "composite_subtask_execution" in stage_names
    assert "composite_finalize" in stage_names


@pytest.mark.asyncio
async def test_composite_plan_returns_partial_success_on_failure(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.tools.base import ToolRegistry

    conversation = Conversation(id="conv_composite_fail", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    registry = ToolRegistry()
    registry.register(_CompositeReviewTool())
    registry.register(_CompositeQuizFailTool())

    agent = Agent(
        llm_service=_NoLlmExpected(),
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=3),
        task_planner=_CompositePlanner(),
    )

    events = [
        event async for event in agent.chat("帮我先总结 TCP 的重点，再出 3 道题", "u1", "conv_composite_fail")
    ]

    final_text = "".join(event.content or "" for event in events if event.type.value == "text_delta")
    assert "## 复习总结" in final_text
    assert "## 未完成项" in final_text
    done_event = next(event for event in events if event.type.value == "done")
    assert len(done_event.metadata["completed_subtasks"]) == 1
    assert done_event.metadata["failed_subtask"]["selected_tool"] == "quiz_generator"
