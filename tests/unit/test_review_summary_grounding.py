from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.trace.trace_collector import TraceCollector
from src.observability.dashboard.services.trace_service import TraceService


def _result(text: str, **metadata):
    return SimpleNamespace(
        text=text,
        metadata=metadata,
        chunk_id=metadata.get("chunk_id", "chunk-1"),
        score=metadata.get("score", 0.9),
    )


@pytest.mark.asyncio
async def test_review_summary_returns_evidence_metadata() -> None:
    from src.agent.tools.review_summary import ReviewSummaryArgs, ReviewSummaryTool
    from src.agent.types import LlmResponse, ToolContext

    hybrid_search = MagicMock()
    hybrid_search.search = MagicMock(
        return_value=[
            _result(
                "DNS 解析包含递归查询和迭代查询。",
                source_path="dns.pdf",
                title="DNS",
                page=3,
            )
        ]
    )
    llm = AsyncMock()
    llm.send_request = AsyncMock(
        return_value=LlmResponse(content="### DNS 复习要点\n- DNS 解析包含递归查询 [1]。")
    )

    tool = ReviewSummaryTool(hybrid_search=hybrid_search, llm_service=llm)
    result = await tool.execute(
        ToolContext(user_id="u1", conversation_id="c1"),
        ReviewSummaryArgs(topic="DNS"),
    )

    assert result.success is True
    assert result.metadata["grounding_capable"] is True
    assert result.metadata["grounding_passthrough"] is True
    assert result.metadata["final_response_preferred"] is True
    assert len(result.metadata["citations"]) == 1
    assert "dns.pdf" in result.metadata["evidence_summary"]


@pytest.mark.asyncio
async def test_review_summary_returns_real_query_trace_id(tmp_path) -> None:
    from src.agent.tools.review_summary import ReviewSummaryArgs, ReviewSummaryTool
    from src.agent.types import LlmResponse, ToolContext

    trace_path = tmp_path / "traces.jsonl"
    collector = TraceCollector(traces_path=trace_path)
    hybrid_search = MagicMock()
    hybrid_search.search = MagicMock(
        return_value=[
            _result(
                "DNS 解析包含递归查询和迭代查询。",
                source_path="dns.pdf",
                title="DNS",
                page=3,
            )
        ]
    )
    llm = AsyncMock()
    llm.send_request = AsyncMock(
        return_value=LlmResponse(content="### DNS 复习要点\n- DNS 解析包含递归查询 [1]。")
    )

    tool = ReviewSummaryTool(
        hybrid_search=hybrid_search,
        llm_service=llm,
        trace_enabled=True,
        trace_collector=collector,
    )
    result = await tool.execute(
        ToolContext(
            user_id="u1",
            conversation_id="c1",
            metadata={"agent_trace_id": "agent-trace-1"},
        ),
        ReviewSummaryArgs(topic="DNS"),
    )

    query_trace_id = result.metadata["query_trace_id"]
    assert query_trace_id
    assert result.metadata["query_trace_ids"] == [query_trace_id]
    trace = TraceService(trace_path).get_trace(query_trace_id)
    assert trace is not None
    assert trace["metadata"]["source"] == "review_summary"
    assert trace["metadata"]["parent_agent_trace_id"] == "agent-trace-1"
