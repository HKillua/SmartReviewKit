"""Review summary tool — generates structured exam-point summaries from knowledge base."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.grounding import build_evidence_summary
from src.agent.tools.base import Tool
from src.agent.types import ToolContext, ToolResult
from src.agent.utils.sanitizer import sanitize_user_input
from src.core.response.citation_generator import CitationGenerator, sanitize_retrieval_text
from src.core.trace.trace_collector import TraceCollector
from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


def _collection_name(search: Any) -> str:
    value = getattr(search, "default_collection", "")
    return value if isinstance(value, str) else ""


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
    return str(handoff.get("latest_result_text", "") or "").strip()


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


def _extract_summary_snippet(text: str, *, limit: int = 140) -> str:
    cleaned = sanitize_retrieval_text(text).replace("\n", " ").strip()
    if not cleaned:
        return ""
    for separator in ("。", "！", "？", ". "):
        if separator in cleaned:
            head = cleaned.split(separator, 1)[0].strip()
            if head:
                return head[:limit]
    return cleaned[:limit]

REVIEW_PROMPT_TEMPLATE = """请根据以下知识库检索内容，生成一份结构化的考点复习摘要。

主题: {topic}
{chapter_line}

检索到的知识内容:
{context}

{weak_points_section}

要求:
1. 按 "核心概念 → 重要定理/规则 → 易错点 → 与其他章节的关联" 组织
2. 每个考点配一句简短解释
3. 用 ⚠️ 标注薄弱知识点（如果有的话）
4. 使用 Markdown 格式输出
5. 尽量在关键结论后保留 `[1]`、`[2]` 这类来源编号
6. 只总结证据覆盖到的知识点；如果当前资料只覆盖了部分内容，要明确说明

请生成考点复习摘要："""


class ReviewSummaryArgs(BaseModel):
    topic: str = Field(..., description="复习主题或关键词")
    chapter: Optional[str] = Field(default=None, description="章节名称（可选）")
    include_weak_points: bool = Field(default=True, description="是否标注薄弱知识点")


class ReviewSummaryTool(Tool[ReviewSummaryArgs]):
    """Generate a structured review summary for a given topic from the knowledge base."""

    def __init__(
        self,
        hybrid_search: Any = None,
        source_aware_search: Any = None,
        llm_service: Any = None,
        error_memory: Any = None,
        knowledge_map: Any = None,
        trace_enabled: bool = False,
        trace_collector: TraceCollector | None = None,
    ) -> None:
        self._search = hybrid_search
        self._source_aware_search = source_aware_search
        self._llm = llm_service
        self._error_memory = error_memory
        self._knowledge_map = knowledge_map
        self._citation_generator = CitationGenerator(snippet_max_length=220)
        self._trace_enabled = trace_enabled
        self._trace_collector = trace_collector or (TraceCollector() if trace_enabled else None)

    @property
    def name(self) -> str:
        return "review_summary"

    @property
    def description(self) -> str:
        return "按主题/章节从知识库生成结构化考点复习摘要，可标注薄弱知识点"

    def get_args_schema(self) -> type[ReviewSummaryArgs]:
        return ReviewSummaryArgs

    def _ensure_source_aware_search(self) -> Any | None:
        if self._source_aware_search is not None:
            return self._source_aware_search
        if self._search is None:
            return None
        from src.core.query_engine.source_aware_search import SourceAwareSearch

        self._source_aware_search = SourceAwareSearch(hybrid_search=self._search)
        return self._source_aware_search

    def _build_fast_summary(
        self,
        *,
        topic: str,
        results: list,
        citations: list[dict[str, Any]],
        weak_section: str,
    ) -> str:
        lines = [f"# {sanitize_user_input(topic, max_length=80)} 复习摘要", "", "## 核心概念"]
        for index, result in enumerate(results[:3], start=1):
            snippet = _extract_summary_snippet(getattr(result, "text", ""))
            if not snippet:
                continue
            lines.append(f"- {snippet} `[{index}]`")
        if len(lines) == 3:
            lines.append("- 请结合原课件继续细化本节重点。")
        lines.extend(
            [
                "",
                "## 复习提示",
                "- 先掌握定义、流程和易混点，再回看原课件核对细节。",
            ]
        )
        if weak_section:
            lines.append(f"- ⚠️ {weak_section}")
        if citations:
            lines.extend(["", "## 来源编号"])
            for citation in citations[:4]:
                lines.append(
                    f"- [{citation.get('index', '?')}] {citation.get('source', '未知来源')}"
                )
        return "\n".join(lines).strip()

    async def execute(self, context: ToolContext, args: ReviewSummaryArgs) -> ToolResult:
        query_trace: TraceContext | None = None
        normalized_metadata: dict[str, Any] = {}
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
        # Step 1: retrieve knowledge
        if self._search is None:
            return ToolResult(success=False, error="知识检索服务未初始化")

        if self._trace_enabled and self._trace_collector is not None:
            query_trace = TraceContext(trace_type="query")
            query_trace.metadata.update(
                {
                    "query": args.topic[:200],
                    "top_k": 8,
                    "collection": _collection_name(self._search),
                    "source": "review_summary",
                    "parent_agent_trace_id": context.metadata.get("agent_trace_id", ""),
                    "topic": args.topic[:120],
                    "chapter": (args.chapter or "")[:120],
                }
            )
            if composite_mode:
                query_trace.metadata.update(
                    {
                        "composite_parent_request_id": composite_parent_request_id,
                        "composite_subtask_index": composite_subtask_index,
                        "composite_subtask_intent": composite_subtask_intent,
                    }
                )

        try:
            source_aware = self._ensure_source_aware_search()
            top_k = 6 if _response_profile(context) == "balanced_fast" else 8
            if source_aware is not None:
                normalized = await asyncio.to_thread(
                    source_aware.search,
                    query=args.topic,
                    task_intent="review_summary",
                    top_k=top_k,
                    trace=query_trace,
                    fast_mode=_response_profile(context) == "balanced_fast",
                )
                results = normalized.results
                normalized_metadata = dict(normalized.routing_metadata)
            else:
                results = await asyncio.to_thread(
                    self._search.search,
                    query=args.topic,
                    top_k=top_k,
                    trace=query_trace,
                    fast_mode=_response_profile(context) == "balanced_fast",
                )
        except Exception as exc:
            logger.exception("HybridSearch failed in ReviewSummaryTool")
            if query_trace is not None:
                query_trace.record_stage(
                    "error",
                    {"phase": "search", "error": str(exc)[:300]},
                )
                self._trace_collector.collect(query_trace)
            return ToolResult(success=False, error=f"知识检索失败: {exc}")

        citations: list[dict[str, Any]] = []
        evidence_summary = ""
        knowledge_text = ""
        result_chars = 420 if _response_profile(context) == "balanced_fast" else 600
        if results:
            citations = [
                citation.to_dict()
                for citation in self._citation_generator.generate(results)
            ]
            evidence_summary = build_evidence_summary(citations)
            knowledge_text = "\n\n".join(
                f"[{i}] {sanitize_retrieval_text(r.text)[:result_chars]}" for i, r in enumerate(results, 1)
            )
        else:
            knowledge_text = _handoff_context_text(context)
            citations = _handoff_citations(context)
            evidence_summary = build_evidence_summary(citations) if citations else ""
            if query_trace is not None:
                query_trace.metadata["handoff_used"] = bool(knowledge_text)

        if query_trace is not None:
            query_trace.metadata["result_count"] = len(results)
            query_trace.metadata["effective_source_count"] = len(citations)
            if normalized_metadata:
                query_trace.metadata.update(normalized_metadata)
            self._trace_collector.collect(query_trace)

        if not knowledge_text:
            metadata = {
                "grounding_capable": True,
                "citations": [],
                "evidence_summary": "",
                "source_count": 0,
                "query_trace_id": query_trace.trace_id if query_trace is not None else "",
                "query_trace_ids": [query_trace.trace_id] if query_trace is not None else [],
                "final_response_preferred": True,
                "grounding_passthrough": True,
                **normalized_metadata,
            }
            if composite_mode:
                metadata.update(
                    {
                        "composite_parent_request_id": composite_parent_request_id,
                        "composite_subtask_index": composite_subtask_index,
                        "composite_subtask_intent": composite_subtask_intent,
                    }
                )
            return ToolResult(
                success=True,
                result_for_llm="未找到与该主题相关的知识库内容，请确认主题名称。",
                metadata=metadata,
            )

        # Step 2: get weak points from memory (if available)
        weak_section = ""
        if args.include_weak_points and self._error_memory:
            try:
                weak_concepts = await self._error_memory.get_weak_concepts(context.user_id)
                if weak_concepts:
                    weak_section = f"该学生的薄弱知识点: {', '.join(weak_concepts[:10])}"
            except Exception:
                logger.warning("Failed to retrieve weak concepts from ErrorMemory")

        chapter_line = f"章节: {sanitize_user_input(args.chapter, max_length=100)}" if args.chapter else ""

        if _response_profile(context) == "balanced_fast":
            fast_summary = self._build_fast_summary(
                topic=args.topic,
                results=results,
                citations=citations,
                weak_section=weak_section,
            )
            return ToolResult(
                success=True,
                result_for_llm=fast_summary,
                metadata={
                    "grounding_capable": True,
                    "citations": citations,
                    "evidence_summary": evidence_summary,
                    "source_count": len(citations),
                    "query_trace_id": query_trace.trace_id if query_trace is not None else "",
                    "query_trace_ids": [query_trace.trace_id] if query_trace is not None else [],
                    "final_response_preferred": True,
                    "grounding_passthrough": True,
                    "generation_mode": "review_summary_fast_path",
                    "composite_parent_request_id": composite_parent_request_id if composite_mode else "",
                    "composite_subtask_index": composite_subtask_index if composite_mode else -1,
                    "composite_subtask_intent": composite_subtask_intent if composite_mode else "",
                    **normalized_metadata,
                },
            )

        prompt = REVIEW_PROMPT_TEMPLATE.format(
            topic=sanitize_user_input(args.topic),
            chapter_line=chapter_line,
            context=knowledge_text,
            weak_points_section=weak_section,
        )

        # Step 3: call LLM to generate summary
        if self._llm is None:
            return ToolResult(
                success=True,
                result_for_llm=f"[LLM 不可用，返回原始检索结果]\n\n{knowledge_text}",
                metadata={
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
                    **normalized_metadata,
                },
            )

        try:
            from src.agent.types import LlmMessage, LlmRequest
            req = LlmRequest(
                messages=[
                    LlmMessage(role="system", content="你是一位专业的课程教师，擅长总结考点。"),
                    LlmMessage(role="user", content=prompt),
                ],
                temperature=0.3,
                max_tokens=900 if _response_profile(context) == "balanced_fast" else None,
            )
            resp = await self._llm.send_request(req)
            if resp.error:
                return ToolResult(
                    success=True,
                    result_for_llm=f"[LLM 调用失败，返回原始检索结果]\n\n{knowledge_text}",
                    metadata={
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
            return ToolResult(
                success=True,
                result_for_llm=resp.content or "",
                metadata={
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
                    **normalized_metadata,
                },
            )
        except Exception as exc:
            logger.exception("LLM call failed in ReviewSummaryTool")
            return ToolResult(
                success=True,
                result_for_llm=f"[LLM 降级，返回原始检索结果]\n\n{knowledge_text}",
                metadata={
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
                    **normalized_metadata,
                },
            )
