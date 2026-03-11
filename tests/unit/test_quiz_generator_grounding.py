from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _result(text: str, **metadata):
    return SimpleNamespace(
        text=text,
        metadata=metadata,
        chunk_id=metadata.get("chunk_id", "chunk-1"),
        score=metadata.get("score", 0.9),
    )


@pytest.mark.asyncio
async def test_quiz_generator_question_bank_mode_returns_citations() -> None:
    from src.agent.tools.quiz_generator import QuizGeneratorArgs, QuizGeneratorTool
    from src.agent.types import ToolContext

    hybrid_search = MagicMock()
    hybrid_search.search = MagicMock(
        return_value=[
            _result(
                "TCP 是什么？",
                source_path="question_bank.md",
                answer="传输控制协议",
                explanation="TCP 是传输层的可靠协议。",
                options={"A": "传输控制协议", "B": "用户数据报协议"},
                tags=["TCP"],
            )
        ]
    )

    tool = QuizGeneratorTool(hybrid_search=hybrid_search, llm_service=AsyncMock())
    result = await tool.execute(
        ToolContext(user_id="u1", conversation_id="c1"),
        QuizGeneratorArgs(topic="TCP", count=1),
    )

    assert result.success is True
    assert result.metadata["generation_mode"] == "question_bank"
    assert result.metadata["grounding_capable"] is True
    assert result.metadata["final_response_preferred"] is True
    assert len(result.metadata["citations"]) == 1


@pytest.mark.asyncio
async def test_quiz_generator_rag_backed_mode_returns_citations() -> None:
    from src.agent.tools.quiz_generator import QuizGeneratorArgs, QuizGeneratorTool
    from src.agent.types import LlmResponse, ToolContext

    def _search(*, query, top_k, filters=None):
        if filters:
            return []
        return [
            _result(
                "TCP 通过三次握手建立连接。",
                source_path="network.md",
                title="TCP",
            )
        ]

    hybrid_search = MagicMock()
    hybrid_search.search = MagicMock(side_effect=_search)

    llm = AsyncMock()
    llm.send_request = AsyncMock(
        return_value=LlmResponse(
            content='[{"question":"TCP 为什么需要三次握手？","answer":"为了确认双方收发能力和初始序号。","explanation":"三次握手可以同步双方状态。","concepts":["TCP"]}]'
        )
    )

    tool = QuizGeneratorTool(hybrid_search=hybrid_search, llm_service=llm)
    result = await tool.execute(
        ToolContext(user_id="u1", conversation_id="c1"),
        QuizGeneratorArgs(topic="TCP", count=1),
    )

    assert result.success is True
    assert result.metadata["generation_mode"] == "rag_backed"
    assert len(result.metadata["citations"]) == 1
    assert "TCP" in result.result_for_llm


@pytest.mark.asyncio
async def test_quiz_generator_returns_insufficient_evidence_without_llm_fallback() -> None:
    from src.agent.tools.quiz_generator import QuizGeneratorArgs, QuizGeneratorTool
    from src.agent.types import ToolContext

    hybrid_search = MagicMock()
    hybrid_search.search = MagicMock(return_value=[])
    llm = AsyncMock()

    tool = QuizGeneratorTool(hybrid_search=hybrid_search, llm_service=llm)
    result = await tool.execute(
        ToolContext(user_id="u1", conversation_id="c1"),
        QuizGeneratorArgs(topic="一个不存在的冷门主题", count=2),
    )

    assert result.success is True
    assert result.metadata["generation_mode"] == "insufficient_evidence"
    assert result.metadata["final_response_preferred"] is True
    assert result.metadata["citations"] == []
    llm.send_request.assert_not_called()
