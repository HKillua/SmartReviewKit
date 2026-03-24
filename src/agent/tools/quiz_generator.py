"""Quiz generation tool — creates practice questions from the knowledge base."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.grounding import build_evidence_summary
from src.agent.tools.base import Tool
from src.agent.types import ToolContext, ToolResult
from src.agent.utils.sanitizer import sanitize_user_input
from src.core.response.citation_generator import CitationGenerator, sanitize_retrieval_text

logger = logging.getLogger(__name__)


def _composite_handoff(context: ToolContext) -> dict[str, Any]:
    value = context.metadata.get("composite_handoff", {})
    return value if isinstance(value, dict) else {}


def _handoff_context_text(context: ToolContext) -> str:
    handoff = _composite_handoff(context)
    aggregate = handoff.get("aggregate_evidence", {})
    if isinstance(aggregate, dict):
        evidence_summary = str(aggregate.get("evidence_summary", "") or "").strip()
        if evidence_summary:
            return evidence_summary
    latest_result = str(handoff.get("latest_result_text", "") or "").strip()
    return latest_result


def _handoff_citations(context: ToolContext) -> list[dict[str, Any]]:
    handoff = _composite_handoff(context)
    aggregate = handoff.get("aggregate_evidence", {})
    if isinstance(aggregate, dict):
        citations = aggregate.get("citations", [])
        if isinstance(citations, list) and citations:
            return [dict(citation) for citation in citations if isinstance(citation, dict)]
    latest = handoff.get("latest_citations", [])
    if isinstance(latest, list):
        return [dict(citation) for citation in latest if isinstance(citation, dict)]
    return []


def _response_profile(context: ToolContext) -> str:
    value = str(context.metadata.get("response_profile", "") or "").strip().lower()
    return value if value in {"balanced_fast", "quality_first"} else "quality_first"


def _should_wait_for_user(context: ToolContext) -> bool:
    if bool(context.metadata.get("agenda_next_goal_depends_on_user_input", False)):
        return True
    matched_skill = str(context.metadata.get("matched_skill", "") or "").strip()
    return matched_skill in {"quiz_drill", "error_review"}


def _quiz_bundle_from_questions(
    questions: list[dict[str, Any]],
    *,
    question_type: str,
    topic: str,
) -> list[dict[str, Any]]:
    bundle: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        concepts = [
            str(value).strip()
            for value in question.get("concepts", [])
            if str(value).strip()
        ]
        bundle.append(
            {
                "index": index,
                "question": str(question.get("question", "") or "").strip(),
                "correct_answer": str(question.get("answer", "") or "").strip(),
                "question_type": question_type,
                "topic": topic,
                "concepts": concepts,
            }
        )
    return bundle


def _apply_completion_metadata(
    metadata: dict[str, Any],
    *,
    context: ToolContext,
    quiz_bundle: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    updated = dict(metadata)
    wait_for_user = _should_wait_for_user(context)
    updated["completion_hint"] = "wait_user" if wait_for_user else "step_done"
    if quiz_bundle:
        updated["resume_payload"] = {"quiz_bundle": quiz_bundle}
    return updated

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

FAST_QUIZ_PROMPT_TEMPLATE = """请根据以下课程证据，生成 {count} 道{question_type}，难度级别 {difficulty}/5。

知识内容:
{context}

{weak_points_hint}

要求:
1. 每道题用 JSON 对象表示，字段为: question, options (选择题才需要), answer, explanation, concepts
2. 所有题目放在一个 JSON 数组中
3. explanation 保持精炼，1-2 句话说明考查点即可
4. concepts 只保留最核心的 1-3 个知识点
5. 题目尽量短、聚焦，不要生成过长题干

请直接输出 JSON 数组："""


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


def _extract_quiz_snippet(text: str, *, limit: int = 90) -> str:
    cleaned = sanitize_retrieval_text(text).replace("\n", " ").strip()
    if not cleaned:
        return ""
    for separator in ("。", "！", "？", ". "):
        if separator in cleaned:
            head = cleaned.split(separator, 1)[0].strip()
            if head:
                return head[:limit]
    return cleaned[:limit]


class QuizGeneratorArgs(BaseModel):
    topic: str = Field(..., description="出题主题")
    question_type: str = Field(
        default="选择题",
        description="题型：选择题、填空题、简答题、SQL题",
    )
    count: int = Field(default=3, ge=1, le=10, description="题目数量")
    difficulty: int = Field(default=3, ge=1, le=5, description="难度级别 1-5")


FALLBACK_QUIZ_PROMPT = """请根据你的知识，生成 {count} 道关于 "{topic}" 的{question_type}，难度级别 {difficulty}/5。

{weak_points_hint}

要求:
1. 每道题用 JSON 对象表示，包含字段: question, options (选择题才需要), answer, explanation, concepts
2. 所有题目放在一个 JSON 数组中
3. concepts 是该题涉及的知识点列表
4. **explanation 字段必须包含详细的解题思路和知识点分析**，至少 2-3 句话
5. 注意：此题目基于通用知识生成，未关联课程课件

请输出 JSON 数组（不要有多余文本）："""


class QuizGeneratorTool(Tool[QuizGeneratorArgs]):
    """Generate quiz questions on a topic from the knowledge base."""

    def __init__(
        self,
        hybrid_search: Any = None,
        source_aware_search: Any = None,
        llm_service: Any = None,
        error_memory: Any = None,
        knowledge_map: Any = None,
        query_router: Any = None,
    ) -> None:
        self._search = hybrid_search
        self._source_aware_search = source_aware_search
        self._llm = llm_service
        self._error_memory = error_memory
        self._knowledge_map = knowledge_map
        self._router = query_router
        self._citation_generator = CitationGenerator(snippet_max_length=220)

    @property
    def name(self) -> str:
        return "quiz_generator"

    @property
    def description(self) -> str:
        return "基于知识库内容生成指定题型、数量和难度的练习题"

    def get_args_schema(self) -> type[QuizGeneratorArgs]:
        return QuizGeneratorArgs

    def _ensure_source_aware_search(self) -> Any | None:
        if self._source_aware_search is not None:
            return self._source_aware_search
        if self._search is None:
            return None
        from src.core.query_engine.source_aware_search import SourceAwareSearch

        self._source_aware_search = SourceAwareSearch(
            hybrid_search=self._search,
            query_router=self._router,
        )
        return self._source_aware_search

    async def execute(self, context: ToolContext, args: QuizGeneratorArgs) -> ToolResult:
        weak_hint = await self._get_weak_hint(context.user_id)
        handoff_context = _handoff_context_text(context)
        handoff_citations = _handoff_citations(context)

        # --- Tier 1: try question bank ---
        qb_results = await self._search_question_bank(args.topic, args.count)
        if qb_results and len(qb_results) >= args.count:
            question_bank_questions: list[dict[str, Any]] = []
            for result in qb_results[: args.count]:
                question: dict[str, Any] = {
                    "question": sanitize_retrieval_text(result.text)[:800],
                    "answer": result.metadata.get("answer", ""),
                    "explanation": result.metadata.get("explanation", ""),
                    "concepts": result.metadata.get("tags", []),
                }
                options = result.metadata.get("options")
                if options:
                    question["options"] = options
                question_bank_questions.append(question)
            formatted = "以下题目来自课程题库：\n\n" + _format_questions(
                question_bank_questions,
                args.question_type,
            )
            citations = [
                citation.to_dict()
                for citation in self._citation_generator.generate(qb_results[: args.count])
            ]
            return ToolResult(
                success=True,
                result_for_llm=formatted,
                metadata=_apply_completion_metadata(
                    {
                    "question_count": min(len(qb_results), args.count),
                    "source": "question_bank",
                    "generation_mode": "question_bank",
                    "grounding_capable": True,
                    "citations": citations,
                    "evidence_summary": build_evidence_summary(citations),
                    "source_count": len(citations),
                    "final_response_preferred": True,
                    },
                    context=context,
                    quiz_bundle=_quiz_bundle_from_questions(
                        question_bank_questions,
                        question_type=args.question_type,
                        topic=args.topic,
                    ),
                ),
            )

        # --- Tier 2: RAG context + LLM generation ---
        response_profile = _response_profile(context)
        top_k = 3 if response_profile == "balanced_fast" else 6
        results = await self._search_all(args.topic, top_k=top_k)
        if results:
            return await self._generate_from_context(
                results,
                args,
                weak_hint,
                response_profile=response_profile,
                supplemental_context=handoff_context,
                supplemental_citations=handoff_citations,
            )

        if handoff_context:
            return await self._generate_from_handoff_context(
                handoff_context,
                args,
                weak_hint,
                citations=handoff_citations,
            )

        return ToolResult(
            success=True,
            result_for_llm=(
                f"当前知识库中关于“{sanitize_user_input(args.topic)}”的课程证据不足，"
                "暂不生成正式题目。请缩小范围（如指定章节、协议或知识点），"
                "或先导入相关课件后再试。"
            ),
            metadata=_apply_completion_metadata(
                {
                "question_count": 0,
                "generation_mode": "insufficient_evidence",
                "grounding_capable": True,
                "citations": [],
                "evidence_summary": "",
                "source_count": 0,
                "final_response_preferred": True,
                },
                context=context,
            ),
        )

    async def _search_question_bank(self, topic: str, count: int) -> list:
        if self._search is None:
            return []
        try:
            source_aware = self._ensure_source_aware_search()
            if source_aware is not None:
                normalized = await asyncio.to_thread(
                    source_aware.search,
                    query=topic,
                    task_intent="quiz_generator",
                    top_k=count + 2,
                    allowed_sources=["question_bank"],
                )
                return list(normalized.results)
            results = await asyncio.to_thread(
                self._search.search,
                query=topic,
                top_k=count + 2,
                filters={"source_type": "question_bank"},
            )
            return results if isinstance(results, list) else []
        except Exception:
            logger.debug("Question bank search failed, continuing", exc_info=True)
            return []

    async def _search_all(self, topic: str, top_k: int = 6) -> list:
        if self._search is None:
            return []
        try:
            source_aware = self._ensure_source_aware_search()
            if source_aware is not None:
                normalized = await asyncio.to_thread(
                    source_aware.search,
                    query=topic,
                    task_intent="quiz_generator",
                    top_k=top_k,
                    fast_mode=top_k <= 3,
                )
                return list(normalized.results)
            results = await asyncio.to_thread(
                self._search.search, query=topic, top_k=top_k,
            )
            return results if isinstance(results, list) else []
        except Exception as exc:
            logger.warning("Knowledge search failed: %s", exc)
            return []

    def _format_existing_questions(self, results: list, question_type: str) -> str:
        """Format question bank retrieval results as quiz output."""
        questions: list[dict] = []
        for r in results:
            q: dict = {
                "question": sanitize_retrieval_text(r.text)[:800],
                "answer": r.metadata.get("answer", ""),
                "explanation": r.metadata.get("explanation", ""),
                "concepts": r.metadata.get("tags", []),
            }
            options = r.metadata.get("options")
            if options:
                q["options"] = options
            questions.append(q)
        return "以下题目来自课程题库：\n\n" + _format_questions(questions, question_type)

    def _build_fast_questions(self, results: list, args: "QuizGeneratorArgs") -> list[dict]:
        topic = sanitize_user_input(args.topic, max_length=80)
        questions: list[dict] = []
        fallback_point = f"{topic} 的核心概念与适用场景"
        for index in range(args.count):
            result = results[index % len(results)]
            snippet = _extract_quiz_snippet(getattr(result, "text", "")) or fallback_point
            if "选择" in args.question_type:
                questions.append(
                    {
                        "question": f"关于 {topic}，以下哪一项最符合课程资料中的描述？（第 {index + 1} 题）",
                        "options": {
                            "A": snippet,
                            "B": f"{topic} 与该知识点无关",
                            "C": f"{topic} 只在物理层中使用",
                            "D": f"{topic} 不涉及任何状态或控制机制",
                        },
                        "answer": "A",
                        "explanation": f"课程证据指出：{snippet}",
                        "concepts": [topic],
                    }
                )
            else:
                questions.append(
                    {
                        "question": f"请简述 {topic} 的一个核心知识点。（第 {index + 1} 题）",
                        "answer": snippet,
                        "explanation": f"可结合课程证据作答：{snippet}",
                        "concepts": [topic],
                    }
                )
        return questions

    async def _generate_from_context(
        self,
        results: list,
        args: "QuizGeneratorArgs",
        weak_hint: str,
        *,
        response_profile: str = "quality_first",
        supplemental_context: str = "",
        supplemental_citations: Optional[list[dict[str, Any]]] = None,
    ) -> ToolResult:
        if response_profile == "balanced_fast":
            questions = self._build_fast_questions(results, args)
            citations = [
                citation.to_dict()
                for citation in self._citation_generator.generate(results)
            ]
            if supplemental_citations:
                citations.extend(supplemental_citations)
            formatted = _format_questions(questions, args.question_type)
            return ToolResult(
                success=True,
                result_for_llm=formatted,
                metadata=_apply_completion_metadata(
                    {
                    "question_count": len(questions),
                    "source": "rag_backed",
                    "generation_mode": "rag_backed",
                    "grounding_capable": True,
                    "citations": citations,
                    "evidence_summary": build_evidence_summary(citations),
                    "source_count": len(citations),
                    "final_response_preferred": True,
                    },
                    context=context,
                    quiz_bundle=_quiz_bundle_from_questions(
                        questions,
                        question_type=args.question_type,
                        topic=args.topic,
                    ),
                ),
            )

        result_chars = 220 if response_profile == "balanced_fast" else 500
        supplemental_chars = 500 if response_profile == "balanced_fast" else 1200
        knowledge_text = "\n\n".join(
            f"[{i}] {sanitize_retrieval_text(r.text)[:result_chars]}"
            for i, r in enumerate(results, 1)
        )
        if supplemental_context:
            knowledge_text += f"\n\n[Supplemental]\n{supplemental_context[:supplemental_chars]}"
        prompt_template = (
            FAST_QUIZ_PROMPT_TEMPLATE if response_profile == "balanced_fast" else QUIZ_PROMPT_TEMPLATE
        )
        prompt = prompt_template.format(
            count=args.count,
            question_type=sanitize_user_input(args.question_type, max_length=50),
            difficulty=args.difficulty,
            context=knowledge_text,
            weak_points_hint=weak_hint,
        )
        citations = [
            citation.to_dict()
            for citation in self._citation_generator.generate(results)
        ]
        if supplemental_citations:
            citations.extend(supplemental_citations)
        return await self._call_llm_for_quiz(
            prompt,
            args.question_type,
            source="rag_backed",
            citations=citations,
            response_profile=response_profile,
            topic=args.topic,
            context=context,
        )

    async def _generate_from_handoff_context(
        self,
        context_text: str,
        args: "QuizGeneratorArgs",
        weak_hint: str,
        *,
        citations: Optional[list[dict[str, Any]]] = None,
    ) -> ToolResult:
        prompt = QUIZ_PROMPT_TEMPLATE.format(
            count=args.count,
            question_type=sanitize_user_input(args.question_type, max_length=50),
            difficulty=args.difficulty,
            context=context_text[:2400],
            weak_points_hint=weak_hint,
        )
        return await self._call_llm_for_quiz(
            prompt,
            args.question_type,
            source="composite_handoff",
            citations=citations or [],
            response_profile="quality_first",
            topic=args.topic,
            context=context,
        )

    async def _generate_from_llm_knowledge(self, args: "QuizGeneratorArgs", weak_hint: str) -> ToolResult:
        if self._llm is None:
            return ToolResult(success=False, error="LLM 服务未初始化，无法生成题目")
        prompt = FALLBACK_QUIZ_PROMPT.format(
            count=args.count,
            topic=sanitize_user_input(args.topic),
            question_type=sanitize_user_input(args.question_type, max_length=50),
            difficulty=args.difficulty,
            weak_points_hint=weak_hint,
        )
        return await self._call_llm_for_quiz(
            prompt,
            args.question_type,
            source="llm_fallback",
            response_profile="quality_first",
            topic=args.topic,
            context=context,
        )

    async def _call_llm_for_quiz(
        self,
        prompt: str,
        question_type: str,
        source: str = "rag_backed",
        citations: Optional[list[dict[str, Any]]] = None,
        response_profile: str = "quality_first",
        topic: str = "",
        context: ToolContext | None = None,
    ) -> ToolResult:
        if self._llm is None:
            return ToolResult(success=False, error="LLM 服务未初始化")
        try:
            from src.agent.types import LlmMessage, LlmRequest
            req = LlmRequest(
                messages=[
                    LlmMessage(role="system", content="你是一位课程出题专家，只输出合法 JSON。"),
                    LlmMessage(role="user", content=prompt),
                ],
                temperature=0.3 if response_profile == "balanced_fast" else 0.5,
                max_tokens=780 if response_profile == "balanced_fast" else None,
            )
            resp = await self._llm.send_request(req)
            if resp.error:
                return ToolResult(success=False, error=f"LLM 生成失败: {resp.error}")

            from src.agent.utils.json_helpers import safe_parse_json
            questions = safe_parse_json(resp.content or "")
            if questions is None:
                return ToolResult(
                    success=True,
                    result_for_llm=resp.content or "",
                    metadata=_apply_completion_metadata(
                        {
                        "question_count": 0,
                        "source": source,
                        "generation_mode": source,
                        "grounding_capable": source != "insufficient_evidence",
                        "citations": citations or [],
                        "evidence_summary": build_evidence_summary(citations or []),
                        "source_count": len(citations or []),
                        "final_response_preferred": True,
                        },
                        context=context or ToolContext(user_id="", conversation_id=""),
                    ),
                )

            formatted = _format_questions(questions, question_type)
            return ToolResult(
                success=True,
                result_for_llm=formatted,
                metadata=_apply_completion_metadata(
                    {
                    "question_count": len(questions),
                    "source": source,
                    "generation_mode": source,
                    "grounding_capable": True,
                    "citations": citations or [],
                    "evidence_summary": build_evidence_summary(citations or []),
                    "source_count": len(citations or []),
                    "final_response_preferred": True,
                    },
                    context=context or ToolContext(user_id="", conversation_id=""),
                    quiz_bundle=_quiz_bundle_from_questions(
                        questions,
                        question_type=question_type,
                        topic=topic,
                    ),
                ),
            )
        except Exception as exc:
            logger.exception("QuizGeneratorTool LLM call failed")
            return ToolResult(success=False, error=f"出题失败: {exc}")

    async def _get_weak_hint(self, user_id: str) -> str:
        if not self._error_memory:
            return ""
        try:
            weak = await self._error_memory.get_weak_concepts(user_id)
            if weak:
                return f"该学生薄弱知识点: {', '.join(weak[:8])}，请适当侧重。"
        except Exception:
            pass
        return ""
