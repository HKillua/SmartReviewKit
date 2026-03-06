"""Quiz evaluation tool — judges user answers and updates memory."""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.tools.base import Tool
from src.agent.types import ToolContext, ToolResult
from src.agent.utils.sanitizer import sanitize_user_input

logger = logging.getLogger(__name__)

EVAL_PROMPT_TEMPLATE = """请评判以下题目的用户答案。

题目类型: {question_type}
题目: {question}
正确答案: {correct_answer}
用户答案: {user_answer}

请按以下 JSON 格式输出（不要有多余文本）:
{{
  "verdict": "correct" | "incorrect" | "partial",
  "score": 0-100,
  "explanation": "详细解析",
  "key_concepts": ["涉及的核心知识点"]
}}"""


class QuizEvaluatorArgs(BaseModel):
    question: str = Field(..., description="题目内容")
    user_answer: str = Field(..., description="用户的答案")
    correct_answer: str = Field(..., description="正确答案")
    question_type: str = Field(default="选择题", description="题型")
    topic: str = Field(default="", description="题目所属主题")
    concepts: list[str] = Field(default_factory=list, description="涉及的知识点")


class QuizEvaluatorTool(Tool[QuizEvaluatorArgs]):
    """Evaluate a user's answer, provide feedback, and update memory stores."""

    def __init__(
        self,
        llm_service: Any = None,
        error_memory: Any = None,
        knowledge_map: Any = None,
    ) -> None:
        self._llm = llm_service
        self._error_memory = error_memory
        self._knowledge_map = knowledge_map

    @property
    def name(self) -> str:
        return "quiz_evaluator"

    @property
    def description(self) -> str:
        return "评判用户的答案正确性，给出详细解析，并更新学习记忆"

    def get_args_schema(self) -> type[QuizEvaluatorArgs]:
        return QuizEvaluatorArgs

    async def execute(self, context: ToolContext, args: QuizEvaluatorArgs) -> ToolResult:
        if self._llm is None:
            is_correct = args.user_answer.strip().lower() == args.correct_answer.strip().lower()
            verdict = "correct" if is_correct else "incorrect"
            return ToolResult(
                success=True,
                result_for_llm=(
                    f"判定: {'✅ 正确' if is_correct else '❌ 错误'}\n"
                    f"正确答案: {args.correct_answer}\n"
                    "(LLM 不可用，无法提供详细解析)"
                ),
            )

        try:
            from src.agent.types import LlmMessage, LlmRequest
            from src.agent.utils.json_helpers import safe_parse_json

            prompt = EVAL_PROMPT_TEMPLATE.format(
                question_type=sanitize_user_input(args.question_type, max_length=50),
                question=sanitize_user_input(args.question),
                correct_answer=sanitize_user_input(args.correct_answer),
                user_answer=sanitize_user_input(args.user_answer),
            )

            req = LlmRequest(
                messages=[
                    LlmMessage(role="system", content="你是一位严谨的课程考试评判员，只输出合法 JSON。"),
                    LlmMessage(role="user", content=prompt),
                ],
                temperature=0.1,
            )
            resp = await self._llm.send_request(req)
            if resp.error:
                return ToolResult(success=False, error=f"评判失败: {resp.error}")

            result = safe_parse_json(resp.content or "")
            if result is None:
                result = {"verdict": "unknown", "explanation": resp.content, "key_concepts": []}

            verdict = result.get("verdict", "unknown").lower().strip()
            result["verdict"] = verdict
            is_correct = verdict == "correct"

            # Update memory
            await self._update_memory(context, args, result, is_correct)

            icons = {"correct": "✅ 正确", "incorrect": "❌ 错误", "partial": "⚠️ 部分正确"}
            return ToolResult(
                success=True,
                result_for_llm=(
                    f"判定: {icons.get(verdict, verdict)}\n"
                    f"得分: {result.get('score', 'N/A')}/100\n\n"
                    f"**解析**: {result.get('explanation', '')}\n\n"
                    f"**涉及知识点**: {', '.join(result.get('key_concepts', args.concepts))}"
                ),
                metadata={"verdict": verdict, "score": result.get("score")},
            )
        except Exception as exc:
            logger.exception("QuizEvaluatorTool failed")
            return ToolResult(success=False, error=f"评判异常: {exc}")

    async def _update_memory(
        self, context: ToolContext, args: QuizEvaluatorArgs, result: dict, is_correct: bool
    ) -> None:
        concepts = result.get("key_concepts", args.concepts) or args.concepts

        if not is_correct and self._error_memory:
            try:
                from src.agent.memory.error_memory import ErrorRecord
                record = ErrorRecord(
                    user_id=context.user_id,
                    question=args.question,
                    question_type=args.question_type,
                    topic=args.topic,
                    concepts=concepts,
                    user_answer=args.user_answer,
                    correct_answer=args.correct_answer,
                    explanation=result.get("explanation", ""),
                    error_type="conceptual",
                )
                await self._error_memory.add_error(context.user_id, record)
            except Exception:
                logger.warning("Failed to save error to ErrorMemory")

        if self._knowledge_map and concepts:
            for concept in concepts:
                try:
                    await self._knowledge_map.update_mastery(
                        context.user_id, concept, correct=is_correct
                    )
                except Exception:
                    logger.warning("Failed to update KnowledgeMapMemory for %s", concept)
