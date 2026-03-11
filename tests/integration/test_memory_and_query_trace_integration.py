from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.memory.session_memory import SessionMemory
from src.agent.types import Conversation, LlmResponse, ToolCallData
from src.core.trace.trace_collector import TraceCollector
from src.core.types import RetrievalResult
from src.observability.dashboard.services.trace_service import TraceService


class _ReviewAgentLlm:
    async def send_request(self, request):
        system = request.messages[0].content or ""
        if request.tools:
            return LlmResponse(
                content="让我先整理复习要点。",
                tool_calls=[
                    ToolCallData(
                        id="tc_review",
                        name="review_summary",
                        arguments={"topic": "DNS"},
                    )
                ],
            )
        if "擅长总结考点" in system:
            return LlmResponse(content="### DNS 复习要点\n- DNS 解析流程包含递归查询 [1]。")
        return LlmResponse(content="DNS 复习要点 [1]")


class _QuizEvaluatorAgentLlm:
    async def send_request(self, request):
        system = request.messages[0].content or ""
        if request.tools:
            return LlmResponse(
                content="让我先判题。",
                tool_calls=[
                    ToolCallData(
                        id="tc_eval",
                        name="quiz_evaluator",
                        arguments={
                            "question": "TCP 为什么需要三次握手？",
                            "user_answer": "为了确认双方收发能力正常。",
                            "correct_answer": "为了同步双方状态并确认收发能力。",
                            "topic": "TCP",
                            "concepts": ["TCP"],
                        },
                    )
                ],
            )
        if "只输出合法 JSON" in system:
            return LlmResponse(
                content='{"verdict":"correct","score":100,"explanation":"基础解析","key_concepts":["TCP"]}'
            )
        if "只增强解析" in system:
            return LlmResponse(content="根据课程资料，TCP 需要三次握手来同步状态 [1]。")
        return LlmResponse(content="判题完成。")


@pytest.mark.asyncio
async def test_review_summary_generates_agent_query_and_memory_traces(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.memory.enhancer import MemoryRecordHook
    from src.agent.tools.base import ToolRegistry
    from src.agent.tools.review_summary import ReviewSummaryTool

    trace_path = tmp_path / "traces.jsonl"
    collector = TraceCollector(traces_path=trace_path)
    session_mem = SessionMemory(str(tmp_path))

    hybrid_search = MagicMock()
    hybrid_search.search.return_value = [
        RetrievalResult(
            chunk_id="chunk_dns",
            score=0.95,
            text="DNS 解析流程包含递归查询和迭代查询。",
            metadata={"source_path": "dns.md", "title": "DNS", "page": 3},
        )
    ]

    registry = ToolRegistry()
    registry.register(
        ReviewSummaryTool(
            hybrid_search=hybrid_search,
            llm_service=_ReviewAgentLlm(),
            trace_enabled=True,
            trace_collector=collector,
        )
    )

    conversation = Conversation(id="conv_review_trace", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    memory_hook = MemoryRecordHook(
        session_memory=session_mem,
        extraction_mode="rule",
        trace_enabled=True,
        trace_collector=collector,
    )

    agent = Agent(
        llm_service=_ReviewAgentLlm(),
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=2),
        lifecycle_hooks=[memory_hook],
        trace_enabled=True,
        trace_collector=collector,
    )

    events = [event async for event in agent.chat("帮我总结 DNS 解析流程的复习要点。", "u1", "conv_review_trace")]
    await agent.flush()

    done_event = next(event for event in events if event.type.value == "done")
    service = TraceService(trace_path)
    agent_traces = service.list_traces(trace_type="agent")
    query_traces = service.list_traces(trace_type="query")
    memory_traces = service.list_traces(trace_type="memory")

    assert len(agent_traces) == 1
    assert len(query_traces) == 1
    assert len(memory_traces) == 1
    assert done_event.metadata["query_trace_ids"] == [query_traces[0]["trace_id"]]
    assert query_traces[0]["metadata"]["source"] == "review_summary"
    assert query_traces[0]["metadata"]["parent_agent_trace_id"] == agent_traces[0]["trace_id"]
    assert memory_traces[0]["metadata"]["conversation_id"] == "conv_review_trace"


@pytest.mark.asyncio
async def test_quiz_evaluator_evidence_enhanced_generates_linked_traces(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.memory.enhancer import MemoryRecordHook
    from src.agent.tools.base import ToolRegistry
    from src.agent.tools.quiz_evaluator import QuizEvaluatorTool

    trace_path = tmp_path / "quiz_traces.jsonl"
    collector = TraceCollector(traces_path=trace_path)
    session_mem = SessionMemory(str(tmp_path))

    hybrid_search = MagicMock()
    hybrid_search.search.return_value = [
        RetrievalResult(
            chunk_id="chunk_tcp",
            score=0.93,
            text="TCP 通过三次握手同步双方状态。",
            metadata={"source_path": "tcp.md", "title": "TCP", "page": 8},
        )
    ]

    registry = ToolRegistry()
    registry.register(
        QuizEvaluatorTool(
            llm_service=_QuizEvaluatorAgentLlm(),
            hybrid_search=hybrid_search,
            trace_enabled=True,
            trace_collector=collector,
        )
    )

    conversation = Conversation(id="conv_eval_trace", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    memory_hook = MemoryRecordHook(
        session_memory=session_mem,
        extraction_mode="rule",
        trace_enabled=True,
        trace_collector=collector,
    )

    agent = Agent(
        llm_service=_QuizEvaluatorAgentLlm(),
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=2),
        lifecycle_hooks=[memory_hook],
        trace_enabled=True,
        trace_collector=collector,
    )

    events = [event async for event in agent.chat("请帮我判这道 TCP 题。", "u1", "conv_eval_trace")]
    await agent.flush()

    done_event = next(event for event in events if event.type.value == "done")
    service = TraceService(trace_path)
    agent_traces = service.list_traces(trace_type="agent")
    query_traces = service.list_traces(trace_type="query")
    memory_traces = service.list_traces(trace_type="memory")

    assert len(agent_traces) == 1
    assert len(query_traces) == 1
    assert len(memory_traces) == 1
    assert done_event.metadata["evaluation_mode"] == "evidence_enhanced"
    assert done_event.metadata["query_trace_ids"] == [query_traces[0]["trace_id"]]
    assert query_traces[0]["metadata"]["source"] == "quiz_evaluator"
    assert query_traces[0]["metadata"]["parent_agent_trace_id"] == agent_traces[0]["trace_id"]
