"""Quiz evaluation tool — judges user answers and updates memory."""

from __future__ import annotations

import logging
import asyncio
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.grounding import build_evidence_summary
from src.agent.tools.base import Tool
from src.agent.types import ToolContext, ToolResult
from src.agent.utils.sanitizer import sanitize_user_input
from src.core.response.citation_generator import CitationGenerator
from src.core.trace.trace_collector import TraceCollector
from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


def _collection_name(search: Any) -> str:
    value = getattr(search, "default_collection", "")
    return value if isinstance(value, str) else ""


def _composite_handoff(context: ToolContext) -> dict[str, Any]:
    value = context.metadata.get("composite_handoff", {})
    return value if isinstance(value, dict) else {}


def _handoff_evidence_summary(context: ToolContext) -> str:
    handoff = _composite_handoff(context)
    aggregate = handoff.get("aggregate_evidence", {})
    if isinstance(aggregate, dict):
        evidence_summary = str(aggregate.get("evidence_summary", "") or "").strip()
        if evidence_summary:
            return evidence_summary
    return str(handoff.get("latest_evidence_summary", "") or "").strip()


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


def _int_metadata(value: Any, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default

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

EXPLANATION_ENHANCEMENT_PROMPT = """你已经完成了题目评分，请基于课程证据增强解析。

题目: {question}
正确答案: {correct_answer}
用户答案: {user_answer}
判定: {verdict}
得分: {score}
原始解析: {base_explanation}
涉及知识点: {key_concepts}

课程证据:
{evidence_summary}

要求:
1. 在不改变 verdict 和 score 的前提下，重写更具体、可追溯的解析
2. 尽量在关键句后保留 `[1]`、`[2]` 这类来源编号
3. 只能基于给定证据增强解释，不要编造课程资料之外的新结论
4. 直接输出解析正文，不要输出 JSON
"""


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
        hybrid_search: Any = None,
        trace_enabled: bool = False,
        trace_collector: TraceCollector | None = None,
    ) -> None:
        self._llm = llm_service
        self._error_memory = error_memory
        self._knowledge_map = knowledge_map
        self._search = hybrid_search
        self._citation_generator = CitationGenerator(snippet_max_length=220)
        self._trace_enabled = trace_enabled
        self._trace_collector = trace_collector or (TraceCollector() if trace_enabled else None)

    @property
    def name(self) -> str:
        return "quiz_evaluator"

    @property
    def description(self) -> str:
        return "评判用户的答案正确性，给出详细解析，并更新学习记忆"

    def get_args_schema(self) -> type[QuizEvaluatorArgs]:
        return QuizEvaluatorArgs

    async def execute(self, context: ToolContext, args: QuizEvaluatorArgs) -> ToolResult:
        composite_mode = bool(context.metadata.get("composite_mode", False))
        composite_parent_request_id = str(
            context.metadata.get("composite_parent_request_id", "") or ""
        ).strip()
        composite_subtask_index = _int_metadata(
            context.metadata.get("composite_subtask_index", -1),
            default=-1,
        )
        composite_subtask_intent = str(
            context.metadata.get("composite_subtask_intent", "") or ""
        ).strip()
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
                metadata={
                    "verdict": verdict,
                    "evaluation_mode": "direct_no_evidence",
                    "grounding_capable": True,
                    "citations": [],
                    "evidence_summary": "",
                    "source_count": 0,
                    "query_trace_id": "",
                    "query_trace_ids": [],
                    "final_response_preferred": True,
                    "grounding_passthrough": True,
                    "composite_parent_request_id": composite_parent_request_id if composite_mode else "",
                    "composite_subtask_index": composite_subtask_index if composite_mode else -1,
                    "composite_subtask_intent": composite_subtask_intent if composite_mode else "",
                },
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
            concepts = result.get("key_concepts", args.concepts) or args.concepts
            explanation = str(result.get("explanation", "") or "")
            citations: list[dict] = []
            evidence_summary = ""
            evaluation_mode = "direct_no_evidence"
            query_trace: TraceContext | None = None

            evidence_results, query_trace = await self._retrieve_supporting_evidence(
                context,
                args,
                concepts,
            )
            if evidence_results:
                citations = [
                    citation.to_dict()
                    for citation in self._citation_generator.generate(evidence_results)
                ]
                evidence_summary = build_evidence_summary(citations)
                enhanced = await self._enhance_explanation_with_evidence(
                    args=args,
                    verdict=verdict,
                    score=result.get("score"),
                    base_explanation=explanation,
                    key_concepts=concepts,
                    evidence_summary=evidence_summary,
                )
                if enhanced:
                    explanation = enhanced
                    evaluation_mode = "evidence_enhanced"
            else:
                handoff_summary = _handoff_evidence_summary(context)
                if handoff_summary:
                    citations = _handoff_citations(context)
                    evidence_summary = handoff_summary
                    enhanced = await self._enhance_explanation_with_evidence(
                        args=args,
                        verdict=verdict,
                        score=result.get("score"),
                        base_explanation=explanation,
                        key_concepts=concepts,
                        evidence_summary=evidence_summary,
                    )
                    if enhanced:
                        explanation = enhanced
                        evaluation_mode = "handoff_evidence_enhanced"
            if not evidence_results and not evidence_summary and explanation:
                explanation += "\n\n> 说明：以上解析未绑定课程资料证据，请结合课件复核。"
            result["explanation"] = explanation

            # Update memory
            await self._update_memory(context, args, result, is_correct)

            icons = {"correct": "✅ 正确", "incorrect": "❌ 错误", "partial": "⚠️ 部分正确"}
            return ToolResult(
                success=True,
                result_for_llm=(
                    f"判定: {icons.get(verdict, verdict)}\n"
                    f"得分: {result.get('score', 'N/A')}/100\n\n"
                    f"**解析**: {result.get('explanation', '')}\n\n"
                    f"**涉及知识点**: {', '.join(concepts)}"
                ),
                metadata={
                    "verdict": verdict,
                    "score": result.get("score"),
                    "evaluation_mode": evaluation_mode,
                    "grounding_capable": True,
                    "citations": citations,
                    "evidence_summary": evidence_summary,
                    "source_count": len(citations),
                    "query_trace_id": query_trace.trace_id if query_trace is not None else "",
                    "query_trace_ids": [query_trace.trace_id] if query_trace is not None else [],
                    "final_response_preferred": True,
                    "grounding_passthrough": True,
                    "composite_parent_request_id": composite_parent_request_id if composite_mode else "",
                    "composite_subtask_index": composite_subtask_index if composite_mode else -1,
                    "composite_subtask_intent": composite_subtask_intent if composite_mode else "",
                },
            )
        except Exception as exc:
            logger.exception("QuizEvaluatorTool failed")
            return ToolResult(success=False, error=f"评判异常: {exc}")

    async def _retrieve_supporting_evidence(
        self,
        context: ToolContext,
        args: QuizEvaluatorArgs,
        concepts: list[str],
    ) -> tuple[list, TraceContext | None]:
        if self._search is None:
            return [], None
        search_query = " ".join(
            part for part in [args.topic, " ".join(concepts[:4]), args.question[:120]] if part
        ).strip()
        if not search_query:
            return [], None
        query_trace: TraceContext | None = None
        if self._trace_enabled and self._trace_collector is not None:
            query_trace = TraceContext(trace_type="query")
            query_trace.metadata.update(
                {
                    "query": search_query[:200],
                    "top_k": 4,
                    "collection": _collection_name(self._search),
                    "source": "quiz_evaluator",
                    "parent_agent_trace_id": context.metadata.get("agent_trace_id", ""),
                    "topic": args.topic[:120],
                    "concept_count": len(concepts),
                    "evaluation_mode_candidate": "evidence_enhanced",
                }
            )
            if context.metadata.get("composite_mode", False):
                query_trace.metadata.update(
                    {
                        "composite_parent_request_id": str(
                            context.metadata.get("composite_parent_request_id", "") or ""
                        ),
                        "composite_subtask_index": int(
                            _int_metadata(
                                context.metadata.get("composite_subtask_index", -1),
                                default=-1,
                            )
                        ),
                        "composite_subtask_intent": str(
                            context.metadata.get("composite_subtask_intent", "") or ""
                        ),
                    }
                )
        try:
            results = await asyncio.to_thread(
                self._search.search,
                query=search_query,
                top_k=4,
                trace=query_trace,
            )
            result_list = results if isinstance(results, list) else []
            if query_trace is not None:
                query_trace.metadata["result_count"] = len(result_list)
                self._trace_collector.collect(query_trace)
            return result_list, query_trace
        except Exception as exc:
            logger.warning("QuizEvaluator evidence retrieval failed", exc_info=True)
            if query_trace is not None:
                query_trace.record_stage(
                    "error",
                    {"phase": "search", "error": str(exc)[:300]},
                )
                self._trace_collector.collect(query_trace)
            return [], query_trace

    async def _enhance_explanation_with_evidence(
        self,
        *,
        args: QuizEvaluatorArgs,
        verdict: str,
        score: Any,
        base_explanation: str,
        key_concepts: list[str],
        evidence_summary: str,
    ) -> str:
        if self._llm is None or not evidence_summary:
            return ""
        try:
            from src.agent.types import LlmMessage, LlmRequest

            prompt = EXPLANATION_ENHANCEMENT_PROMPT.format(
                question=sanitize_user_input(args.question),
                correct_answer=sanitize_user_input(args.correct_answer),
                user_answer=sanitize_user_input(args.user_answer),
                verdict=verdict,
                score=score,
                base_explanation=sanitize_user_input(base_explanation, max_length=1200),
                key_concepts=", ".join(key_concepts),
                evidence_summary=evidence_summary,
            )
            req = LlmRequest(
                messages=[
                    LlmMessage(role="system", content="你是一位严谨的课程助教，只增强解析，不改变判分结论。"),
                    LlmMessage(role="user", content=prompt),
                ],
                temperature=0.2,
            )
            resp = await self._llm.send_request(req)
            if resp.error:
                return ""
            return (resp.content or "").strip()
        except Exception:
            logger.warning("QuizEvaluator evidence enhancement failed", exc_info=True)
            return ""

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
