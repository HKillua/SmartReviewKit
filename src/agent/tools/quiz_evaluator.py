"""Quiz evaluation tool — judges user answers and updates memory."""

from __future__ import annotations

import logging
import asyncio
from typing import Any, Optional
import re

from pydantic import BaseModel, Field

from src.agent.grounding import build_evidence_summary
from src.agent.quiz_batch import QuizBatchAlignment, QuizBatchItem, build_quiz_batch_alignment
from src.agent.tools.base import Tool
from src.agent.types import LlmMessage, LlmRequest, ToolContext, ToolResult
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


def _response_profile(context: ToolContext) -> str:
    value = str(context.metadata.get("response_profile", "") or "").strip().lower()
    return value if value in {"balanced_fast", "quality_first"} else "quality_first"


def _evidence_brief(citations: list[dict[str, Any]], *, limit: int = 2) -> str:
    lines: list[str] = []
    for citation in citations[:limit]:
        snippet = str(citation.get("text_snippet", "") or "").replace("\n", " ").strip()
        source = str(citation.get("source", "未知来源") or "未知来源")
        index = citation.get("index", "?")
        if snippet:
            lines.append(f"- [{index}] {source}: {snippet[:120]}")
    return "\n".join(lines)


_CITATION_REF_RE = re.compile(r"\[(\d+)\]")

EVAL_PROMPT_TEMPLATE = """请评判以下题目的用户答案。

题目类型: {question_type}
题目: {question}
正确答案(如未提供则为空): {correct_answer}
用户答案: {user_answer}

补充要求:
1. 如果没有提供正确答案，请根据题目和课程常识自行判断参考答案，再评判用户答案
2. 评分要保守，不要因为缺少标准答案就直接给满分

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
    question: str = Field(default="", description="题目内容")
    user_answer: str = Field(default="", description="用户的答案")
    correct_answer: str = Field(default="", description="正确答案")
    question_type: str = Field(default="选择题", description="题型")
    topic: str = Field(default="", description="题目所属主题")
    concepts: list[str] = Field(default_factory=list, description="涉及的知识点")
    items: list[dict[str, Any]] = Field(default_factory=list, description="批量判改条目")
    alignment_mode: str = Field(default="", description="题答对齐模式")
    alignment_status: str = Field(default="", description="题答对齐状态")
    split_confidence: float = Field(default=0.0, description="拆题置信度")
    clarification_reason: str = Field(default="", description="需要澄清时的原因")


class QuizEvaluatorTool(Tool[QuizEvaluatorArgs]):
    """Evaluate a user's answer, provide feedback, and update memory stores."""

    def __init__(
        self,
        llm_service: Any = None,
        error_memory: Any = None,
        knowledge_map: Any = None,
        hybrid_search: Any = None,
        source_aware_search: Any = None,
        trace_enabled: bool = False,
        trace_collector: TraceCollector | None = None,
    ) -> None:
        self._llm = llm_service
        self._error_memory = error_memory
        self._knowledge_map = knowledge_map
        self._search = hybrid_search
        self._source_aware_search = source_aware_search
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

    def _ensure_source_aware_search(self) -> Any | None:
        if self._source_aware_search is not None:
            return self._source_aware_search
        if self._search is None:
            return None
        from src.core.query_engine.source_aware_search import SourceAwareSearch

        self._source_aware_search = SourceAwareSearch(hybrid_search=self._search)
        return self._source_aware_search

    async def execute(self, context: ToolContext, args: QuizEvaluatorArgs) -> ToolResult:
        response_profile = _response_profile(context)
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
        try:
            alignment = await self._resolve_batch_alignment(context, args)
            batch_items = alignment.items
            if not alignment.is_aligned:
                return self._clarification_result(
                    alignment=alignment,
                    composite_mode=composite_mode,
                    composite_parent_request_id=composite_parent_request_id,
                    composite_subtask_index=composite_subtask_index,
                    composite_subtask_intent=composite_subtask_intent,
                )

            semaphore = asyncio.Semaphore(3)

            async def _runner(index: int, item: QuizBatchItem) -> dict[str, Any]:
                async with semaphore:
                    return await self._evaluate_single_item(
                        context=context,
                        item=item,
                        item_index=index,
                        response_profile=response_profile,
                    )

            item_results = await asyncio.gather(
                *[
                    _runner(index, item)
                    for index, item in enumerate(batch_items, start=1)
                ]
            )
            aggregated = self._aggregate_batch(item_results)
            rendered = self._render_batch_result(aggregated, alignment)
            metadata = {
                "verdict": aggregated["verdict"],
                "score": aggregated["score"],
                "evaluation_mode": aggregated["evaluation_mode"],
                "grounding_capable": True,
                "citations": aggregated["citations"],
                "evidence_summary": aggregated["evidence_summary"],
                "source_count": len(aggregated["citations"]),
                "query_trace_id": aggregated["query_trace_ids"][0] if aggregated["query_trace_ids"] else "",
                "query_trace_ids": aggregated["query_trace_ids"],
                "final_response_preferred": True,
                "grounding_passthrough": True,
                "tool_output_kind": "final_answer",
                "completion_hint": "step_done",
                "batch_evaluation": len(batch_items) > 1,
                "question_count": len(batch_items),
                "batch_results": aggregated["batch_results"],
                "alignment_mode": alignment.alignment_mode,
                "alignment_status": alignment.alignment_status,
                "split_confidence": alignment.split_confidence,
                "score_aggregation": "mean_rounded",
                "composite_parent_request_id": composite_parent_request_id if composite_mode else "",
                "composite_subtask_index": composite_subtask_index if composite_mode else -1,
                "composite_subtask_intent": composite_subtask_intent if composite_mode else "",
            }
            return ToolResult(success=True, result_for_llm=rendered, metadata=metadata)
        except Exception as exc:
            logger.exception("QuizEvaluatorTool failed")
            return ToolResult(success=False, error=f"评判异常: {exc}")

    async def _resolve_batch_alignment(
        self,
        context: ToolContext,
        args: QuizEvaluatorArgs,
    ) -> QuizBatchAlignment:
        resume_payload = context.metadata.get("agenda_resume_payload", {})
        resume_quiz_bundle: list[dict[str, Any]] = []
        if isinstance(resume_payload, dict):
            candidate = resume_payload.get("quiz_bundle", [])
            if isinstance(candidate, list):
                resume_quiz_bundle = candidate
        items = self._normalize_items(args)
        if args.alignment_status == "aligned" and items:
            return QuizBatchAlignment(
                items=items,
                alignment_mode=args.alignment_mode or "explicit_message",
                alignment_status="aligned",
                split_confidence=float(args.split_confidence or 1.0),
            )
        if args.alignment_status in {"clarification_required", "partial_alignment"}:
            return QuizBatchAlignment(
                items=items,
                alignment_mode=args.alignment_mode or "explicit_message",
                alignment_status=args.alignment_status,
                split_confidence=float(args.split_confidence or 0.0),
                clarification_reason=args.clarification_reason,
            )
        if items:
            return QuizBatchAlignment(
                items=items,
                alignment_mode=args.alignment_mode or "explicit_message",
                alignment_status="aligned",
                split_confidence=float(args.split_confidence or 1.0),
            )
        latest_user = next(
            (
                str(message.get("content", "") or "")
                for message in reversed(context.recent_messages)
                if str(message.get("role", "")) == "user" and str(message.get("content", "") or "").strip()
            ),
            "",
        )
        return await build_quiz_batch_alignment(
            message=latest_user,
            recent_messages=context.recent_messages,
            quiz_bundle=resume_quiz_bundle,
            llm_service=self._llm,
            max_items=5,
        )

    @staticmethod
    def _normalize_items(args: QuizEvaluatorArgs) -> list[QuizBatchItem]:
        normalized: list[QuizBatchItem] = []
        for raw in args.items:
            if not isinstance(raw, dict):
                continue
            item = QuizBatchItem(
                question=str(raw.get("question", "") or ""),
                user_answer=str(raw.get("user_answer", "") or ""),
                correct_answer=str(raw.get("correct_answer", "") or ""),
                question_type=str(raw.get("question_type", "") or "选择题"),
                topic=str(raw.get("topic", "") or ""),
                concepts=[str(value) for value in raw.get("concepts", []) if str(value).strip()],
                index=int(raw.get("index", 0) or 0),
                alignment_notes=str(raw.get("alignment_notes", "") or ""),
                answer_confidence=float(raw.get("answer_confidence", 1.0) or 1.0),
            )
            if item.question or item.user_answer:
                normalized.append(item)
        if normalized:
            normalized.sort(key=lambda item: (item.index or 999, item.question))
            return normalized
        if args.question or args.user_answer or args.correct_answer:
            return [
                QuizBatchItem(
                    question=args.question,
                    user_answer=args.user_answer,
                    correct_answer=args.correct_answer,
                    question_type=args.question_type,
                    topic=args.topic,
                    concepts=list(args.concepts),
                    index=1,
                )
            ]
        return []

    def _clarification_result(
        self,
        *,
        alignment: QuizBatchAlignment,
        composite_mode: bool,
        composite_parent_request_id: str,
        composite_subtask_index: int,
        composite_subtask_intent: str,
    ) -> ToolResult:
        hints = [
            "请按“第1题：... 第2题：...”逐条作答。",
            "如果方便，也可以把原题一起贴上来。",
        ]
        item_count = len(alignment.items)
        prefix = "我还不能稳定对齐这次的多题作答。"
        if item_count == 1:
            prefix = "我还不能稳定识别这道题的题目与作答对应关系。"
        reason = alignment.clarification_reason or "当前消息里的题目边界或题答对应关系不够清晰。"
        text = f"{prefix}\n\n原因：{reason}\n\n建议：\n- " + "\n- ".join(hints)
        return ToolResult(
            success=True,
            result_for_llm=text,
            metadata={
                "evaluation_mode": "alignment_clarification",
                "grounding_capable": False,
                "citations": [],
                "evidence_summary": "",
                "source_count": 0,
                "query_trace_id": "",
                "query_trace_ids": [],
                "final_response_preferred": True,
                "grounding_passthrough": True,
                "tool_output_kind": "final_answer",
                "completion_hint": "clarify",
                "batch_evaluation": item_count > 1,
                "question_count": item_count,
                "batch_results": [
                    {
                        "index": item.index,
                        "question": item.question,
                        "verdict": "clarification_required",
                        "score": None,
                        "concepts": list(item.concepts),
                        "query_trace_ids": [],
                        "source_count": 0,
                    }
                    for item in alignment.items
                ],
                "alignment_mode": alignment.alignment_mode,
                "alignment_status": "clarification_required",
                "split_confidence": alignment.split_confidence,
                "score_aggregation": "mean_rounded",
                "composite_parent_request_id": composite_parent_request_id if composite_mode else "",
                "composite_subtask_index": composite_subtask_index if composite_mode else -1,
                "composite_subtask_intent": composite_subtask_intent if composite_mode else "",
            },
        )

    async def _evaluate_single_item(
        self,
        *,
        context: ToolContext,
        item: QuizBatchItem,
        item_index: int,
        response_profile: str,
    ) -> dict[str, Any]:
        prompt = EVAL_PROMPT_TEMPLATE.format(
            question_type=sanitize_user_input(item.question_type, max_length=50),
            question=sanitize_user_input(item.question),
            correct_answer=sanitize_user_input(item.correct_answer or "未提供"),
            user_answer=sanitize_user_input(item.user_answer),
        )

        result: dict[str, Any] | None = None
        if self._llm is not None:
            req = LlmRequest(
                messages=[
                    LlmMessage(role="system", content="你是一位严谨的课程考试评判员，只输出合法 JSON。"),
                    LlmMessage(role="user", content=prompt),
                ],
                temperature=0.1,
                max_tokens=320 if response_profile == "balanced_fast" else None,
            )
            try:
                if response_profile == "balanced_fast":
                    resp = await asyncio.wait_for(self._llm.send_request(req), timeout=12.0)
                else:
                    resp = await self._llm.send_request(req)
            except TimeoutError:
                resp = None
            if resp is not None and not resp.error:
                from src.agent.utils.json_helpers import safe_parse_json

                result = safe_parse_json(resp.content or "")

        if result is None:
            result = self._fallback_result(item)

        verdict = str(result.get("verdict", "unknown") or "unknown").lower().strip()
        is_correct = verdict == "correct"
        score = int(result.get("score", 0) or 0)
        concepts = [str(value) for value in (result.get("key_concepts", item.concepts) or item.concepts) if str(value).strip()]
        explanation = str(result.get("explanation", "") or "")
        citations: list[dict[str, Any]] = []
        evidence_summary = ""
        evaluation_mode = "direct_no_evidence"
        query_trace: TraceContext | None = None

        evidence_results, query_trace = await self._retrieve_supporting_evidence(
            context,
            item,
            concepts,
        )
        if evidence_results:
            citations = [
                citation.to_dict()
                for citation in self._citation_generator.generate(evidence_results)
            ]
            evidence_summary = build_evidence_summary(citations)
            enhanced = ""
            if self._should_enhance_explanation(
                response_profile=response_profile,
                verdict=verdict,
                base_explanation=explanation,
            ):
                if response_profile == "balanced_fast":
                    enhanced = self._build_fast_evidence_explanation(
                        base_explanation=explanation,
                        citations=citations,
                    )
                else:
                    enhanced = await self._enhance_explanation_with_evidence(
                        item=item,
                        verdict=verdict,
                        score=score,
                        base_explanation=explanation,
                        key_concepts=concepts,
                        evidence_summary=evidence_summary,
                        response_profile=response_profile,
                    )
            if enhanced:
                explanation = enhanced
                evaluation_mode = "evidence_enhanced"
        else:
            handoff_summary = _handoff_evidence_summary(context)
            if handoff_summary:
                citations = _handoff_citations(context)
                evidence_summary = handoff_summary
                enhanced = ""
                if self._should_enhance_explanation(
                    response_profile=response_profile,
                    verdict=verdict,
                    base_explanation=explanation,
                ):
                    if response_profile == "balanced_fast":
                        enhanced = self._build_fast_evidence_explanation(
                            base_explanation=explanation,
                            citations=citations,
                        )
                    else:
                        enhanced = await self._enhance_explanation_with_evidence(
                            item=item,
                            verdict=verdict,
                            score=score,
                            base_explanation=explanation,
                            key_concepts=concepts,
                            evidence_summary=evidence_summary,
                            response_profile=response_profile,
                        )
                if enhanced:
                    explanation = enhanced
                    evaluation_mode = "handoff_evidence_enhanced"
        if not evidence_results and not evidence_summary and explanation:
            explanation += "\n\n> 说明：以上解析未绑定课程资料证据，请结合课件复核。"

        await self._update_memory(context, item, {"key_concepts": concepts, "explanation": explanation}, is_correct)

        return {
            "index": item.index or item_index,
            "question": item.question,
            "user_answer": item.user_answer,
            "correct_answer": item.correct_answer,
            "verdict": verdict,
            "score": score,
            "explanation": explanation,
            "concepts": concepts,
            "citations": citations,
            "evidence_summary": evidence_summary,
            "evaluation_mode": evaluation_mode,
            "query_trace_ids": [query_trace.trace_id] if query_trace is not None else [],
            "source_count": len(citations),
        }

    @staticmethod
    def _remap_citations(
        text: str,
        citations: list[dict[str, Any]],
        *,
        start_index: int,
    ) -> tuple[str, list[dict[str, Any]], int]:
        if not citations:
            return text, [], start_index
        mapping: dict[int, int] = {}
        updated: list[dict[str, Any]] = []
        next_index = start_index
        for citation in citations:
            old_index = int(citation.get("index", 0) or 0) or 1
            if old_index not in mapping:
                mapping[old_index] = next_index
                next_index += 1
            mapped = dict(citation)
            mapped["index"] = mapping[old_index]
            updated.append(mapped)
        updated_text = _CITATION_REF_RE.sub(
            lambda match: f"[{mapping.get(int(match.group(1)), int(match.group(1)))}]",
            text,
        )
        return updated_text, updated, next_index

    def _aggregate_batch(self, item_results: list[dict[str, Any]]) -> dict[str, Any]:
        if not item_results:
            return {
                "verdict": "partial",
                "score": 0,
                "evaluation_mode": "direct_no_evidence",
                "citations": [],
                "evidence_summary": "",
                "query_trace_ids": [],
                "batch_results": [],
            }
        aggregated_citations: list[dict[str, Any]] = []
        next_index = 1
        query_trace_ids: list[str] = []
        modes = {str(item.get("evaluation_mode", "") or "") for item in item_results if str(item.get("evaluation_mode", "") or "")}
        batch_results: list[dict[str, Any]] = []
        for item in sorted(item_results, key=lambda row: int(row.get("index", 0) or 0)):
            explanation, remapped, next_index = self._remap_citations(
                str(item.get("explanation", "") or ""),
                list(item.get("citations", []) or []),
                start_index=next_index,
            )
            item["explanation"] = explanation
            item["citations"] = remapped
            aggregated_citations.extend(remapped)
            for trace_id in item.get("query_trace_ids", []) or []:
                if trace_id and trace_id not in query_trace_ids:
                    query_trace_ids.append(trace_id)
            batch_results.append(
                {
                    "index": int(item.get("index", 0) or 0),
                    "question": str(item.get("question", "") or ""),
                    "verdict": str(item.get("verdict", "") or ""),
                    "score": int(item.get("score", 0) or 0),
                    "concepts": list(item.get("concepts", []) or []),
                    "query_trace_ids": list(item.get("query_trace_ids", []) or []),
                    "source_count": int(item.get("source_count", 0) or 0),
                }
            )
        scores = [int(item.get("score", 0) or 0) for item in item_results]
        avg_score = round(sum(scores) / len(scores))
        verdicts = {str(item.get("verdict", "") or "") for item in item_results}
        if verdicts == {"correct"}:
            overall_verdict = "correct"
        elif verdicts == {"incorrect"}:
            overall_verdict = "incorrect"
        else:
            overall_verdict = "partial"
        evaluation_mode = modes.pop() if len(modes) == 1 else "batch_mixed"
        return {
            "verdict": overall_verdict,
            "score": avg_score,
            "evaluation_mode": evaluation_mode,
            "citations": aggregated_citations,
            "evidence_summary": build_evidence_summary(aggregated_citations) if aggregated_citations else "",
            "query_trace_ids": query_trace_ids,
            "batch_results": batch_results,
            "items": item_results,
        }

    @staticmethod
    def _render_batch_result(
        aggregated: dict[str, Any],
        alignment: QuizBatchAlignment,
    ) -> str:
        icons = {"correct": "✅ 正确", "incorrect": "❌ 错误", "partial": "⚠️ 部分正确"}
        lines = [
            f"本次判改题数: {len(aggregated.get('items', []))}",
            f"整体判定: {icons.get(aggregated.get('verdict', ''), aggregated.get('verdict', ''))}",
            f"平均得分: {aggregated.get('score', 0)}/100",
        ]
        if alignment.alignment_mode:
            lines.append(f"对齐方式: {alignment.alignment_mode}")
        if alignment.split_confidence:
            lines.append(f"拆题置信度: {alignment.split_confidence:.2f}")
        for item in aggregated.get("items", []):
            lines.append("")
            lines.append(f"### 第 {int(item.get('index', 0) or 0)} 题")
            lines.append(f"判定: {icons.get(item.get('verdict', ''), item.get('verdict', ''))}")
            lines.append(f"得分: {int(item.get('score', 0) or 0)}/100")
            lines.append(f"题目: {item.get('question', '')}")
            explanation = str(item.get("explanation", "") or "")
            if explanation:
                lines.append(f"解析: {explanation}")
            concepts = list(item.get("concepts", []) or [])
            if concepts:
                lines.append(f"涉及知识点: {', '.join(concepts)}")
        weak_concepts = []
        for item in aggregated.get("items", []):
            if str(item.get("verdict", "")) != "correct":
                weak_concepts.extend(list(item.get("concepts", []) or []))
        weak_concepts = list(dict.fromkeys([value for value in weak_concepts if value]))
        if weak_concepts:
            lines.append("")
            lines.append(f"总体建议: 优先复习 {', '.join(weak_concepts[:5])}。")
        else:
            lines.append("")
            lines.append("总体建议: 当前这组题掌握较好，可以继续挑战更高阶题目。")
        return "\n".join(lines).strip()

    @staticmethod
    def _fallback_result(item: QuizBatchItem) -> dict[str, Any]:
        user_answer = " ".join((item.user_answer or "").split())
        correct_answer = " ".join((item.correct_answer or "").split())
        if not user_answer:
            return {
                "verdict": "incorrect",
                "score": 0,
                "explanation": "未提供有效作答内容，无法给分。",
                "key_concepts": list(item.concepts or []),
            }
        if correct_answer and user_answer.lower() == correct_answer.lower():
            return {
                "verdict": "correct",
                "score": 100,
                "explanation": "答案与标准答案一致。",
                "key_concepts": list(item.concepts or []),
            }
        return {
            "verdict": "partial",
            "score": 60,
            "explanation": "答案包含部分合理要点，但仍缺少更完整的课程依据和关键细节。",
            "key_concepts": list(item.concepts or []),
        }

    @staticmethod
    def _build_fast_evidence_explanation(
        *,
        base_explanation: str,
        citations: list[dict[str, Any]],
    ) -> str:
        normalized = (base_explanation or "").strip() or "已结合课程证据补充本题解析。"
        brief = _evidence_brief(citations)
        if not brief:
            return normalized
        return f"{normalized}\n\n课程证据：\n{brief}"

    async def _retrieve_supporting_evidence(
        self,
        context: ToolContext,
        item: QuizBatchItem,
        concepts: list[str],
    ) -> tuple[list, TraceContext | None]:
        if self._search is None:
            return [], None
        search_query = " ".join(
            part for part in [item.topic, " ".join(concepts[:4]), item.question[:120]] if part
        ).strip()
        if not search_query:
            return [], None
        query_trace: TraceContext | None = None
        top_k = 3 if _response_profile(context) == "balanced_fast" else 4
        if self._trace_enabled and self._trace_collector is not None:
            query_trace = TraceContext(trace_type="query")
            query_trace.metadata.update(
                {
                    "query": search_query[:200],
                    "top_k": top_k,
                    "collection": _collection_name(self._search),
                    "source": "quiz_evaluator",
                    "parent_agent_trace_id": context.metadata.get("agent_trace_id", ""),
                    "topic": item.topic[:120],
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
            source_aware = self._ensure_source_aware_search()
            if source_aware is not None:
                normalized = await asyncio.to_thread(
                    source_aware.search,
                    query=search_query,
                    task_intent="quiz_evaluator",
                    top_k=top_k,
                    trace=query_trace,
                    fast_mode=_response_profile(context) == "balanced_fast",
                )
                result_list = list(normalized.results)
                if query_trace is not None:
                    query_trace.metadata.update(normalized.routing_metadata)
            else:
                results = await asyncio.to_thread(
                    self._search.search,
                    query=search_query,
                    top_k=top_k,
                    trace=query_trace,
                    fast_mode=_response_profile(context) == "balanced_fast",
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
        item: QuizBatchItem,
        verdict: str,
        score: Any,
        base_explanation: str,
        key_concepts: list[str],
        evidence_summary: str,
        response_profile: str = "quality_first",
    ) -> str:
        if self._llm is None or not evidence_summary:
            return ""
        try:
            from src.agent.types import LlmMessage, LlmRequest

            prompt = EXPLANATION_ENHANCEMENT_PROMPT.format(
                question=sanitize_user_input(item.question),
                correct_answer=sanitize_user_input(item.correct_answer or "未提供"),
                user_answer=sanitize_user_input(item.user_answer),
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
                max_tokens=700 if response_profile == "balanced_fast" else None,
            )
            resp = await self._llm.send_request(req)
            if resp.error:
                return ""
            return (resp.content or "").strip()
        except Exception:
            logger.warning("QuizEvaluator evidence enhancement failed", exc_info=True)
            return ""

    @staticmethod
    def _should_enhance_explanation(
        *,
        response_profile: str,
        verdict: str,
        base_explanation: str,
    ) -> bool:
        if response_profile != "balanced_fast":
            return True
        normalized = " ".join((base_explanation or "").split())
        if verdict != "correct":
            return True
        return len(normalized) < 120

    async def _update_memory(
        self, context: ToolContext, item: QuizBatchItem, result: dict, is_correct: bool
    ) -> None:
        concepts = result.get("key_concepts", item.concepts) or item.concepts

        if not is_correct and self._error_memory:
            try:
                from src.agent.memory.error_memory import ErrorRecord
                record = ErrorRecord(
                    user_id=context.user_id,
                    question=item.question,
                    question_type=item.question_type,
                    topic=item.topic,
                    concepts=concepts,
                    user_answer=item.user_answer,
                    correct_answer=item.correct_answer,
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
