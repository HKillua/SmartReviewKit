"""Quiz generation tool — creates practice questions from the knowledge base."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.tools.base import Tool
from src.agent.types import ToolContext, ToolResult
from src.agent.utils.sanitizer import sanitize_user_input

logger = logging.getLogger(__name__)

QUIZ_PROMPT_TEMPLATE = """请根据以下知识内容，生成 {count} 道{question_type}，难度级别 {difficulty}/5。

知识内容:
{context}

{weak_points_hint}

要求:
1. 每道题用 JSON 对象表示，包含字段: question, options (选择题才需要), answer, explanation, concepts
2. 所有题目放在一个 JSON 数组中
3. concepts 是该题涉及的知识点列表
4. 难度应与级别匹配: 1=基础概念, 2=理解应用, 3=综合分析, 4=设计优化, 5=前沿拓展
5. **explanation 字段必须包含详细的解题思路和知识点分析**，至少 2-3 句话，要说明：
   - 为什么正确答案是对的
   - 其他选项（如选择题）为什么是错的
   - 涉及的核心原理或概念

请输出 JSON 数组（不要有多余文本）："""


def _format_questions(questions: list[dict], question_type: str) -> str:
    """Format quiz questions as Markdown with answers and explanations."""
    parts: list[str] = []
    for i, q in enumerate(questions, 1):
        text = q.get("question", "")
        parts.append(f"### 第 {i} 题\n\n{text}")

        options = q.get("options")
        if options:
            if isinstance(options, dict):
                for key, val in options.items():
                    parts.append(f"- **{key}.** {val}")
            elif isinstance(options, list):
                for idx, val in enumerate(options):
                    label = chr(ord("A") + idx)
                    parts.append(f"- **{label}.** {val}")
            parts.append("")

        answer = q.get("answer", "")
        parts.append(f"<details>\n<summary>🔑 查看答案与解析</summary>\n")
        parts.append(f"**答案**: {answer}\n")

        explanation = q.get("explanation", "")
        if explanation:
            parts.append(f"**解析**: {explanation}\n")

        concepts = q.get("concepts", [])
        if concepts:
            parts.append(f"**涉及知识点**: {', '.join(concepts)}")

        parts.append("</details>\n")

    header = f"以下是 {len(questions)} 道{question_type}：\n"
    return header + "\n".join(parts)


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

        try:
            results = await asyncio.to_thread(self._search.search, query=args.topic, top_k=6)
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
            question_type=sanitize_user_input(args.question_type, max_length=50),
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

            from src.agent.utils.json_helpers import safe_parse_json
            questions = safe_parse_json(resp.content or "")
            if questions is None:
                return ToolResult(success=True, result_for_llm=resp.content or "")

            formatted = _format_questions(questions, args.question_type)
            return ToolResult(
                success=True,
                result_for_llm=formatted,
                metadata={"question_count": len(questions)},
            )
        except Exception as exc:
            logger.exception("QuizGeneratorTool LLM call failed")
            return ToolResult(success=False, error=f"出题失败: {exc}")
