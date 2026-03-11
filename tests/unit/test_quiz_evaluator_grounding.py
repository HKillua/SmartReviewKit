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
async def test_quiz_evaluator_evidence_enhanced_mode_returns_citations() -> None:
    from src.agent.tools.quiz_evaluator import QuizEvaluatorArgs, QuizEvaluatorTool
    from src.agent.types import LlmResponse, ToolContext

    llm = AsyncMock()
    llm.send_request = AsyncMock(
        side_effect=[
            LlmResponse(
                content='{"verdict":"correct","score":100,"explanation":"基础解析","key_concepts":["TCP"]}'
            ),
            LlmResponse(content="根据课程资料，TCP 需要三次握手来同步状态并确认双方能力 [1]。"),
        ]
    )
    hybrid_search = MagicMock()
    hybrid_search.search = MagicMock(
        return_value=[
            _result(
                "TCP 通过三次握手同步双方状态。",
                source_path="tcp.pdf",
                title="TCP",
                page=8,
            )
        ]
    )

    tool = QuizEvaluatorTool(llm_service=llm, hybrid_search=hybrid_search)
    result = await tool.execute(
        ToolContext(user_id="u1", conversation_id="c1"),
        QuizEvaluatorArgs(
            question="TCP 为什么需要三次握手？",
            user_answer="为了确认双方收发能力正常。",
            correct_answer="为了同步双方状态并确认收发能力。",
            topic="TCP",
            concepts=["TCP"],
        ),
    )

    assert result.success is True
    assert result.metadata["evaluation_mode"] == "evidence_enhanced"
    assert result.metadata["grounding_passthrough"] is True
    assert len(result.metadata["citations"]) == 1
    assert "[1]" in result.result_for_llm


@pytest.mark.asyncio
async def test_quiz_evaluator_direct_mode_keeps_scoring_without_evidence() -> None:
    from src.agent.tools.quiz_evaluator import QuizEvaluatorArgs, QuizEvaluatorTool
    from src.agent.types import LlmResponse, ToolContext

    llm = AsyncMock()
    llm.send_request = AsyncMock(
        return_value=LlmResponse(
            content='{"verdict":"incorrect","score":40,"explanation":"答案不完整","key_concepts":["TCP"]}'
        )
    )
    hybrid_search = MagicMock()
    hybrid_search.search = MagicMock(return_value=[])

    tool = QuizEvaluatorTool(llm_service=llm, hybrid_search=hybrid_search)
    result = await tool.execute(
        ToolContext(user_id="u1", conversation_id="c1"),
        QuizEvaluatorArgs(
            question="TCP 为什么需要三次握手？",
            user_answer="为了快。",
            correct_answer="为了同步双方状态并确认收发能力。",
            topic="TCP",
            concepts=["TCP"],
        ),
    )

    assert result.success is True
    assert result.metadata["evaluation_mode"] == "direct_no_evidence"
    assert result.metadata["citations"] == []
    assert "未绑定课程资料证据" in result.result_for_llm
