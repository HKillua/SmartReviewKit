"""Quiz generation tool — creates practice questions from the knowledge base."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.tools.base import Tool
from src.agent.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)

QUIZ_PROMPT_TEMPLATE = """请根据以下知识内容，生成 {count} 道{question_type}，难度级别 {difficulty}/5。

知识内容:
{context}

{weak_points_hint}

要求:
1. 每道题用 JSON 对象表示，包含字段: question, options (选择题), answer, explanation, concepts
2. 所有题目放在一个 JSON 数组中
3. concepts 是该题涉及的知识点列表
4. 难度应与级别匹配: 1=基础概念, 2=理解应用, 3=综合分析, 4=设计优化, 5=前沿拓展

请输出 JSON 数组（不要有多余文本）："""


class QuizGeneratorArgs(BaseModel):
    topic: str = Field(..., description="出题主题")
    question_type: str = Field(
        default="选择题",
        description="题型：选择题、填空题、简答题、SQL题",
    )
    count: int = Field(default=3, ge=1, le=10, description="题目数量")
    difficulty: int = Field(default=3, ge=1, le=5, description="难度级别 1-5")


class QuizGeneratorTool(Tool[QuizGeneratorArgs]):
    """Generate quiz questions on a topic from the knowledge base."""

    def __init__(
        self,
        hybrid_search: Any = None,
        llm_service: Any = None,
        error_memory: Any = None,
        knowledge_map: Any = None,
    ) -> None:
        self._search = hybrid_search
        self._llm = llm_service
        self._error_memory = error_memory
        self._knowledge_map = knowledge_map

    @property
    def name(self) -> str:
        return "quiz_generator"

    @property
    def description(self) -> str:
        return "基于知识库内容生成指定题型、数量和难度的练习题"

    def get_args_schema(self) -> type[QuizGeneratorArgs]:
        return QuizGeneratorArgs

    async def execute(self, context: ToolContext, args: QuizGeneratorArgs) -> ToolResult:
        if self._search is None:
            return ToolResult(success=False, error="知识检索服务未初始化")

        # Retrieve knowledge
        try:
            results = self._search.search(query=args.topic, top_k=6)
        except Exception as exc:
            return ToolResult(success=False, error=f"知识检索失败: {exc}")

        if not results:
            return ToolResult(success=True, result_for_llm="未找到相关知识，无法生成题目。")

        knowledge_text = "\n\n".join(f"[{i}] {r.text[:500]}" for i, r in enumerate(results, 1))

        # Weak points hint
        weak_hint = ""
        if self._error_memory:
            try:
                weak = await self._error_memory.get_weak_concepts(context.user_id)
                if weak:
                    weak_hint = f"该学生薄弱知识点: {', '.join(weak[:8])}，请适当侧重。"
            except Exception:
                pass

        prompt = QUIZ_PROMPT_TEMPLATE.format(
            count=args.count,
            question_type=args.question_type,
            difficulty=args.difficulty,
            context=knowledge_text,
            weak_points_hint=weak_hint,
        )

        if self._llm is None:
            return ToolResult(success=False, error="LLM 服务未初始化，无法生成题目")

        try:
            from src.agent.types import LlmMessage, LlmRequest
            req = LlmRequest(
                messages=[
                    LlmMessage(role="system", content="你是一位课程出题专家，只输出合法 JSON。"),
                    LlmMessage(role="user", content=prompt),
                ],
                temperature=0.5,
            )
            resp = await self._llm.send_request(req)
            if resp.error:
                return ToolResult(success=False, error=f"LLM 生成失败: {resp.error}")

            raw = (resp.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            try:
                questions = json.loads(raw)
            except json.JSONDecodeError:
                return ToolResult(success=True, result_for_llm=resp.content or "")

            return ToolResult(
                success=True,
                result_for_llm=json.dumps(questions, ensure_ascii=False, indent=2),
                metadata={"question_count": len(questions)},
            )
        except Exception as exc:
            logger.exception("QuizGeneratorTool LLM call failed")
            return ToolResult(success=False, error=f"出题失败: {exc}")
