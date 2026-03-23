from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from src.agent.config import AgentConfig
from src.agent.conversation import ConversationStore
from src.agent.pacing import compute_pacing_from_conversation
from src.agent.planner import ControlMode, PlannerDecision, TaskIntent
from src.agent.quiz_batch import (
    build_quiz_batch_alignment,
    extract_recent_quiz_bundle,
)
from src.agent.tools.base import Tool, ToolRegistry
from src.agent.types import Conversation, Message, ToolCallData, ToolContext, ToolResult


_QUIZ_MARKDOWN = """以下是 2 道选择题：

### 第 1 题

TCP 是面向连接的吗？

<details>
<summary>🔑 查看答案与解析</summary>

**答案**: 是

**解析**: TCP 是面向连接协议。

**涉及知识点**: TCP, 面向连接
</details>

### 第 2 题

UDP 是否可靠？

<details>
<summary>🔑 查看答案与解析</summary>

**答案**: 否

**解析**: UDP 不提供可靠传输。

**涉及知识点**: UDP, 可靠传输
</details>
"""


class _BatchQuizArgs(BaseModel):
    items: list[dict] = Field(default_factory=list)
    alignment_mode: str = Field(default="")
    alignment_status: str = Field(default="")
    split_confidence: float = Field(default=0.0)
    clarification_reason: str = Field(default="")


class _BatchQuizTool(Tool[_BatchQuizArgs]):
    @property
    def name(self) -> str:
        return "quiz_evaluator"

    @property
    def description(self) -> str:
        return "fake batch quiz evaluator"

    def get_args_schema(self) -> type[_BatchQuizArgs]:
        return _BatchQuizArgs

    async def execute(self, context: ToolContext, args: _BatchQuizArgs) -> ToolResult:
        count = len(args.items or [])
        return ToolResult(
            success=True,
            result_for_llm=f"本次判改题数: {count}",
            metadata={
                "tool_output_kind": "final_answer",
                "final_response_preferred": True,
                "grounding_passthrough": True,
                "batch_evaluation": count > 1,
                "question_count": count,
                "batch_results": [
                    {
                        "index": int(item.get("index", 0) or 0),
                        "question": str(item.get("question", "") or ""),
                        "verdict": "correct",
                        "score": 100,
                        "concepts": list(item.get("concepts", []) or []),
                        "query_trace_ids": [],
                        "source_count": 0,
                    }
                    for item in args.items
                ],
                "alignment_mode": args.alignment_mode,
                "alignment_status": args.alignment_status or "aligned",
                "split_confidence": float(args.split_confidence or 1.0),
                "score_aggregation": "mean_rounded",
                "source_count": 0,
            },
        )


class _DirectQuizPlanner:
    def plan(self, *args, **kwargs):
        return PlannerDecision(
            task_intent=TaskIntent.QUIZ_EVALUATOR,
            confidence=1.0,
            match_method="test",
            control_mode=ControlMode.FORCE_TOOL,
            selected_tool="quiz_evaluator",
            planner_hint="这是测试用批量判题任务。",
        )


class _PassiveLlm:
    async def send_request(self, request):
        raise AssertionError("direct final-answer quiz path should not call llm")


@pytest.mark.asyncio
async def test_extract_recent_quiz_bundle_parses_generator_markdown() -> None:
    bundle = extract_recent_quiz_bundle([{"role": "assistant", "content": _QUIZ_MARKDOWN}])

    assert len(bundle) == 2
    assert bundle[0].index == 1
    assert bundle[0].question == "TCP 是面向连接的吗？"
    assert bundle[0].correct_answer == "是"
    assert bundle[1].concepts == ["UDP", "可靠传输"]


@pytest.mark.asyncio
async def test_batch_alignment_uses_recent_quiz_context_for_numbered_answers() -> None:
    alignment = await build_quiz_batch_alignment(
        message="请帮我批改：\n第1题：是\n第2题：否",
        recent_messages=[{"role": "assistant", "content": _QUIZ_MARKDOWN}],
        llm_service=None,
        max_items=5,
    )

    assert alignment.is_aligned is True
    assert alignment.alignment_mode == "recent_quiz_context"
    assert [item.index for item in alignment.items] == [1, 2]
    assert alignment.items[0].question == "TCP 是面向连接的吗？"
    assert alignment.items[1].correct_answer == "否"


@pytest.mark.asyncio
async def test_batch_alignment_uses_llm_for_free_text_split() -> None:
    llm = AsyncMock()
    llm.send_request = AsyncMock(
        return_value=SimpleNamespace(
            content=(
                '{"items": ['
                '{"index": 1, "question_text": "", "answer_text": "是", "answer_confidence": 0.92, "alignment_notes": "映射到第一题"},'
                '{"index": 2, "question_text": "", "answer_text": "否", "answer_confidence": 0.88, "alignment_notes": "映射到第二题"}'
                '], "unmatched_segments": [], "clarification_needed": false}'
            ),
            error=None,
        )
    )

    alignment = await build_quiz_batch_alignment(
        message="请帮我批改，我觉得第一题应该是是，第二题应该是否。",
        recent_messages=[{"role": "assistant", "content": _QUIZ_MARKDOWN}],
        llm_service=llm,
        max_items=5,
    )

    assert alignment.is_aligned is True
    assert alignment.alignment_mode == "hybrid_split"
    assert [item.question for item in alignment.items] == [
        "TCP 是面向连接的吗？",
        "UDP 是否可靠？",
    ]


@pytest.mark.asyncio
async def test_quiz_evaluator_batch_returns_structured_metadata() -> None:
    from src.agent.tools.quiz_evaluator import QuizEvaluatorArgs, QuizEvaluatorTool
    from src.agent.types import LlmResponse

    llm = AsyncMock()
    llm.send_request = AsyncMock(
        side_effect=[
            LlmResponse(
                content='{"verdict":"correct","score":100,"explanation":"答对了","key_concepts":["TCP"]}'
            ),
            LlmResponse(
                content='{"verdict":"incorrect","score":30,"explanation":"答错了","key_concepts":["UDP"]}'
            ),
        ]
    )
    tool = QuizEvaluatorTool(llm_service=llm, hybrid_search=None)
    result = await tool.execute(
        ToolContext(user_id="u1", conversation_id="c1"),
        QuizEvaluatorArgs(
            items=[
                {
                    "index": 1,
                    "question": "TCP 是面向连接的吗？",
                    "user_answer": "是",
                    "correct_answer": "是",
                    "concepts": ["TCP"],
                },
                {
                    "index": 2,
                    "question": "UDP 是否可靠？",
                    "user_answer": "是",
                    "correct_answer": "否",
                    "concepts": ["UDP"],
                },
            ],
            alignment_mode="explicit_message",
            alignment_status="aligned",
            split_confidence=1.0,
        ),
    )

    assert result.success is True
    assert result.metadata["batch_evaluation"] is True
    assert result.metadata["question_count"] == 2
    assert result.metadata["alignment_mode"] == "explicit_message"
    assert result.metadata["alignment_status"] == "aligned"
    assert len(result.metadata["batch_results"]) == 2
    assert result.metadata["verdict"] == "partial"
    assert "本次判改题数: 2" in result.result_for_llm


def test_compute_pacing_from_batch_results_metadata() -> None:
    conversation = Conversation(
        id="conv_batch_pacing",
        user_id="u1",
        messages=[
            Message(
                role="assistant",
                tool_calls=[ToolCallData(id="quiz_batch", name="quiz_evaluator", arguments={})],
            ),
            Message(
                role="tool",
                tool_call_id="quiz_batch",
                content="批量判题完成",
                metadata={
                    "batch_results": [
                        {"index": 1, "verdict": "incorrect"},
                        {"index": 2, "verdict": "incorrect"},
                    ]
                },
            ),
        ],
    )

    pacing_level, pacing_reason = compute_pacing_from_conversation(conversation)

    assert pacing_level == "decelerate"
    assert "连续两次判题错误" in pacing_reason


@pytest.mark.asyncio
async def test_agent_done_metadata_includes_batch_quiz_fields() -> None:
    from src.agent.agent import Agent

    conversation = Conversation(id="conv_batch_done", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    registry = ToolRegistry()
    registry.register(_BatchQuizTool())

    agent = Agent(
        llm_service=_PassiveLlm(),
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=1),
        task_planner=_DirectQuizPlanner(),
    )

    message = (
        "请帮我批改：\n"
        "第1题 题目：TCP 是面向连接的吗？ 我的答案：是 正确答案：是\n"
        "第2题 题目：UDP 是否可靠？ 我的答案：是 正确答案：否"
    )
    events = [event async for event in agent.chat(message, "u1", "conv_batch_done")]
    done_event = next(event for event in events if event.type.value == "done")

    assert done_event.metadata["batch_evaluation"] is True
    assert done_event.metadata["question_count"] == 2
    assert done_event.metadata["alignment_mode"] == "explicit_message"
    assert done_event.metadata["alignment_status"] == "aligned"
    assert len(done_event.metadata["batch_results"]) == 2
