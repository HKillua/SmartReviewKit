"""Integration test for agent trace and query trace linkage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.types import Conversation, LlmResponse, ToolCallData
from src.core.trace.trace_collector import TraceCollector
from src.core.types import RetrievalResult
from src.observability.dashboard.services.trace_service import TraceService


class _ToolCallingLlm:
    def __init__(self) -> None:
        self.calls = 0

    async def send_request(self, request):
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(
                content="Searching the knowledge base.",
                tool_calls=[
                    ToolCallData(
                        id="tc_1",
                        name="knowledge_query",
                        arguments={"query": "TCP", "collection": "computer_network"},
                    )
                ],
            )
        return LlmResponse(content="TCP uses SYN, SYN-ACK, and ACK.")


@pytest.mark.asyncio
async def test_agent_and_query_traces_are_linked(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.tools.base import ToolRegistry
    from src.agent.tools.knowledge_query import KnowledgeQueryTool

    trace_path = tmp_path / "traces.jsonl"
    collector = TraceCollector(traces_path=trace_path)

    hybrid_search = MagicMock()
    hybrid_search.search.return_value = [
        RetrievalResult(
            chunk_id="chunk_1",
            score=0.95,
            text="TCP uses SYN, SYN-ACK, and ACK.",
            metadata={"source_path": "network.md", "title": "TCP"},
        )
    ]

    knowledge_tool = KnowledgeQueryTool(
        settings={"observability": {"trace_enabled": True}},
        hybrid_search=hybrid_search,
    )
    knowledge_tool._current_collection = "computer_network"
    knowledge_tool._trace_collector = collector

    registry = ToolRegistry()
    registry.register(knowledge_tool)

    conversation = Conversation(id="conv_integration", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    agent = Agent(
        llm_service=_ToolCallingLlm(),
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=2),
        trace_enabled=True,
        trace_collector=collector,
    )

    events = [event async for event in agent.chat("Explain TCP", "u1", "conv_integration")]
    done_event = next(event for event in events if event.type.value == "done")

    service = TraceService(trace_path)
    agent_traces = service.list_traces(trace_type="agent")
    query_traces = service.list_traces(trace_type="query")

    assert len(agent_traces) == 1
    assert len(query_traces) == 1
    assert agent_traces[0]["trace_id"] == done_event.metadata["trace_id"]
    assert query_traces[0]["metadata"]["parent_agent_trace_id"] == agent_traces[0]["trace_id"]

    tool_stages = [
        stage for stage in agent_traces[0]["stages"]
        if stage.get("stage") == "tool_execution"
    ]
    assert tool_stages
    assert tool_stages[0]["data"]["query_trace_id"] == query_traces[0]["trace_id"]
