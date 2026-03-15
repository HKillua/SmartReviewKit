"""Knowledge retrieval tool — wraps the existing HybridSearch infrastructure."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.grounding import build_evidence_summary
from src.agent.tools.base import Tool
from src.agent.types import ToolContext, ToolResult
from src.core.response.citation_generator import (
    CitationGenerator,
    resolve_source_display,
    sanitize_retrieval_text,
)
from src.core.trace.trace_collector import TraceCollector
from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


def _int_metadata(value: Any, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


class KnowledgeQueryArgs(BaseModel):
    query: str = Field(..., description="检索查询文本")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    collection: Optional[str] = Field(default=None, description="知识库 collection 名称；未提供时使用当前默认 collection")
    content_type: Optional[str] = Field(default=None, description="按内容类型过滤: concept/definition/theorem/example/exercise/formula/summary")
    use_parent: bool = Field(default=False, description="如果匹配的是 child chunk，返回其 parent chunk 以获取更完整的上下文")


class KnowledgeQueryTool(Tool[KnowledgeQueryArgs]):
    """Retrieves relevant knowledge chunks using HybridSearch (Dense + Sparse + RRF)."""

    def __init__(
        self,
        settings: Any = None,
        hybrid_search: Any = None,
        reranker: Any = None,
        query_enhancer: Any = None,
        conflict_detector: Any = None,
        query_router: Any = None,
        source_aware_search: Any = None,
        semantic_cache: Any = None,
        trace_collector: TraceCollector | None = None,
    ) -> None:
        self._settings = settings
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._query_enhancer = query_enhancer
        self._conflict_detector = conflict_detector
        self._query_router = query_router
        self._source_aware_search = source_aware_search
        self._semantic_cache = semantic_cache
        self._initialized = hybrid_search is not None
        self._current_collection: Optional[str] = None
        self._embedding_client: Any = None
        self._init_lock = asyncio.Lock()
        self._trace_collector = trace_collector or TraceCollector()
        self._citation_generator = CitationGenerator(snippet_max_length=220)
        if hybrid_search is not None:
            sparse_retriever = getattr(hybrid_search, "sparse_retriever", None)
            self._current_collection = getattr(sparse_retriever, "default_collection", None)

        retrieval_cfg = None
        if settings and hasattr(settings, 'retrieval'):
            retrieval_cfg = settings.retrieval
        self._rewrite_enabled = bool(getattr(retrieval_cfg, 'query_rewrite_enabled', False))
        self._hyde_enabled = bool(getattr(retrieval_cfg, 'hyde_enabled', False))
        self._multi_query_enabled = bool(getattr(retrieval_cfg, 'multi_query_enabled', False))

    @property
    def name(self) -> str:
        return "knowledge_query"

    @property
    def description(self) -> str:
        return "从知识库中检索与查询相关的课程内容，返回带引用的文本片段"

    def get_args_schema(self) -> type[KnowledgeQueryArgs]:
        return KnowledgeQueryArgs

    def _trace_enabled(self) -> bool:
        observability = {}
        if isinstance(self._settings, dict):
            observability = self._settings.get("observability", {}) or {}
        elif self._settings is not None and hasattr(self._settings, "observability"):
            observability = getattr(self._settings, "observability") or {}
        if isinstance(observability, dict):
            return bool(observability.get("trace_enabled", False))
        return bool(getattr(observability, "trace_enabled", False))

    def _ensure_source_aware_search(self) -> Any | None:
        if self._source_aware_search is not None:
            return self._source_aware_search
        if self._hybrid_search is None:
            return None
        from src.core.query_engine.source_aware_search import SourceAwareSearch

        self._source_aware_search = SourceAwareSearch(
            hybrid_search=self._hybrid_search,
            query_router=self._query_router,
        )
        return self._source_aware_search

    def _ensure_initialized(self, collection: str = "computer_network") -> None:
        if self._initialized and self._current_collection == collection:
            return
        if self._settings is None:
            raise RuntimeError("KnowledgeQueryTool requires settings or a pre-built HybridSearch")

        from src.core.settings import load_settings
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.reranker import create_core_reranker
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        from src.storage.runtime import create_sparse_index

        settings = load_settings() if not hasattr(self._settings, 'embedding') else self._settings

        if self._embedding_client is None:
            self._embedding_client = EmbeddingFactory.create(settings)
        if self._reranker is None:
            self._reranker = create_core_reranker(settings=settings)

        vector_store = VectorStoreFactory.create(settings, collection_name=collection)
        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=self._embedding_client,
            vector_store=vector_store,
        )
        bm25_indexer = create_sparse_index(settings, collection=collection)
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection

        query_processor = QueryProcessor()
        self._hybrid_search = create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )
        self._current_collection = collection
        self._initialized = True
        logger.info("KnowledgeQueryTool initialized for collection: %s", collection)

    async def execute(self, context: ToolContext, args: KnowledgeQueryArgs) -> ToolResult:
        query_trace: TraceContext | None = None
        try:
            effective_collection = (
                args.collection
                or str(context.metadata.get("default_collection", "") or "").strip()
                or self._current_collection
                or "computer_network"
            )
            planner_task_intent = str(context.metadata.get("planner_task_intent", "") or "").strip()
            matched_skill = str(context.metadata.get("matched_skill", "") or "").strip()
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
            # --- Adaptive routing ---
            routing = None
            if self._query_router is not None:
                try:
                    routing = self._query_router.route(
                        args.query,
                        context.recent_messages,
                        planner_task_intent=planner_task_intent or None,
                        matched_skill=matched_skill or None,
                    )
                except Exception:
                    logger.warning("QueryRouter failed, using default retrieval", exc_info=True)

            # --- Semantic cache lookup ---
            if self._semantic_cache is not None:
                try:
                    cached = await self._semantic_cache.get(args.query, collection=effective_collection)
                    if cached is not None:
                        cached_metadata = dict(cached.metadata or {})
                        cached_metadata.setdefault("cache_hit", True)
                        cached_metadata.setdefault("grounding_capable", True)
                        if query_trace is not None:
                            cached_metadata.setdefault("query_trace_id", query_trace.trace_id)
                        return ToolResult(
                            success=True,
                            result_for_llm=cached.result + "\n\n_(来自语义缓存)_",
                            metadata=cached_metadata,
                        )
                except Exception:
                    logger.debug("Semantic cache lookup failed", exc_info=True)

            async with self._init_lock:
                self._ensure_initialized(effective_collection)

            if self._trace_enabled():
                query_trace = TraceContext(trace_type="query")
                query_trace.metadata.update(
                    {
                        "query": args.query[:200],
                        "top_k": args.top_k,
                        "collection": effective_collection,
                        "source": "agent",
                        "parent_agent_trace_id": context.metadata.get("agent_trace_id", ""),
                        "planner_task_intent": planner_task_intent,
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

            effective_query = args.query
            hyde_vector = None

            if self._query_enhancer is not None:
                if self._rewrite_enabled:
                    try:
                        if context.recent_messages and len(context.recent_messages) > 1:
                            effective_query = await self._query_enhancer.conversation_aware_rewrite(
                                args.query, context.recent_messages,
                            )
                            logger.info("Conv-aware rewrite: '%s' -> '%s'", args.query[:40], effective_query[:40])
                        else:
                            effective_query = await self._query_enhancer.rewrite(args.query)
                            logger.info("Query rewrite: '%s' -> '%s'", args.query[:40], effective_query[:40])
                    except Exception:
                        logger.warning("Query rewrite failed, using original")

                if self._hyde_enabled:
                    try:
                        hyde_vector = await self._query_enhancer.hyde_embed(args.query)
                        if hyde_vector:
                            logger.info("HyDE embedding generated for query")
                    except Exception:
                        logger.warning("HyDE failed, using standard embedding")

            if query_trace is not None:
                query_trace.metadata["effective_query"] = effective_query[:200]
                query_trace.record_stage(
                    "agent_query_setup",
                    {
                        "rewrite_enabled": self._rewrite_enabled,
                        "hyde_enabled": self._hyde_enabled,
                        "multi_query_enabled": self._multi_query_enabled,
                        "used_hyde_vector": hyde_vector is not None,
                    },
                )

            search_kwargs: dict[str, Any] = {}
            route_filter = None
            if args.content_type:
                search_kwargs["filters"] = {"content_type": args.content_type}
            elif routing is not None and routing.preferred_sources:
                route_filter = routing.to_metadata_filter()
                if route_filter:
                    search_kwargs.setdefault("filters", {}).update(route_filter)
            if hyde_vector is not None:
                search_kwargs["query_vector"] = hyde_vector

            queries = [effective_query]
            if self._query_enhancer is not None and self._multi_query_enabled:
                try:
                    sub_queries = await self._query_enhancer.decompose(args.query)
                    if len(sub_queries) > 1:
                        queries = sub_queries
                        logger.info("Multi-query decomposed into %d sub-queries", len(queries))
                except Exception:
                    logger.warning("Multi-query decompose failed")

            if query_trace is not None:
                query_trace.metadata["sub_query_count"] = len(queries)
                if routing is not None:
                    query_trace.metadata["preferred_sources"] = list(routing.preferred_sources)
                    query_trace.metadata["retrieval_strategy"] = routing.retrieval_strategy
                    query_trace.metadata["source_weights"] = dict(getattr(routing, "source_weights", {}) or {})
                    query_trace.metadata["source_raw_overfetch"] = dict(getattr(routing, "source_raw_overfetch", {}) or {})
                    query_trace.metadata["normalization_profile"] = getattr(routing, "normalization_profile", "")
                    query_trace.metadata["evidence_profile"] = getattr(routing, "evidence_profile", "")
                    query_trace.metadata["router_match_method"] = routing.match_method
                    query_trace.metadata["router_confidence"] = round(float(routing.confidence), 3)
                    query_trace.metadata["fallback_to_llm"] = bool(routing.fallback_to_llm)
                if route_filter:
                    query_trace.metadata["route_filter"] = route_filter
                if routing is not None:
                    query_trace.record_stage(
                        "retrieval_policy",
                        {
                            "planner_task_intent": planner_task_intent,
                            "matched_skill": matched_skill,
                            "preferred_sources": list(routing.preferred_sources),
                            "source_filter": route_filter or routing.source_filter,
                            "retrieval_strategy": routing.retrieval_strategy,
                            "source_weights": dict(getattr(routing, "source_weights", {}) or {}),
                            "source_raw_overfetch": dict(getattr(routing, "source_raw_overfetch", {}) or {}),
                            "normalization_profile": getattr(routing, "normalization_profile", ""),
                            "evidence_profile": getattr(routing, "evidence_profile", ""),
                            "fallback_to_llm": routing.fallback_to_llm,
                            "need_rag": routing.need_rag,
                            "confidence": round(float(routing.confidence), 3),
                            "method": routing.match_method,
                        },
                    )

            # P3+P6: run searches via asyncio.to_thread to avoid blocking
            # the event loop, and run sub-queries in parallel.
            source_aware = self._ensure_source_aware_search()

            async def _search_one(q: str) -> tuple[list, dict[str, Any]]:
                trace_arg = query_trace if len(queries) == 1 else None
                if source_aware is not None:
                    normalized = await asyncio.to_thread(
                        source_aware.search,
                        query=q,
                        task_intent="knowledge_query",
                        top_k=args.top_k,
                        trace=trace_arg,
                        filters=search_kwargs.get("filters"),
                        route_decision=routing,
                        query_vector=hyde_vector,
                    )
                    return list(normalized.results), dict(normalized.routing_metadata)
                r = await asyncio.to_thread(
                    self._hybrid_search.search,
                    query=q,
                    top_k=args.top_k,
                    trace=trace_arg,
                    **search_kwargs,
                )
                return (r if isinstance(r, list) else []), {}

            search_results = await asyncio.gather(*[_search_one(q) for q in queries])
            all_results: list = []
            normalized_metadata: dict[str, Any] = {}
            for batch, batch_metadata in search_results:
                all_results.extend(batch)
                for key, value in batch_metadata.items():
                    normalized_metadata.setdefault(key, value)

            # Deduplicate by chunk_id, keep highest score
            seen: dict[str, Any] = {}
            for r in all_results:
                if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
                    seen[r.chunk_id] = r
            results = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:args.top_k]

            if query_trace is not None and normalized_metadata:
                query_trace.metadata.update(normalized_metadata)

            if not results:
                metadata = {
                    "grounding_capable": True,
                    "citations": [],
                    "evidence_summary": "",
                    "source_count": 0,
                    "collection": effective_collection,
                    "query_trace_ids": [query_trace.trace_id] if query_trace is not None else [],
                    "final_response_preferred": True,
                }
                if composite_mode:
                    metadata.update(
                        {
                            "composite_parent_request_id": composite_parent_request_id,
                            "composite_subtask_index": composite_subtask_index,
                            "composite_subtask_intent": composite_subtask_intent,
                        }
                    )
                if query_trace is not None:
                    metadata["query_trace_id"] = query_trace.trace_id
                return ToolResult(
                    success=True,
                    result_for_llm="未找到与查询相关的知识库内容。请尝试换一种表述重新提问。",
                    metadata=metadata,
                )

            if args.use_parent and source_aware is None:
                results = self._resolve_parents(results, args.collection)

            conflict_section = ""
            if self._conflict_detector is not None and len(results) > 1:
                try:
                    from src.core.conflict.types import ConflictReport
                    report: ConflictReport = await self._conflict_detector.detect(effective_query, results)
                    if report.has_conflicts:
                        cl = [f"\n⚠️ **知识冲突检测** (共 {len(report.conflicts)} 处)："]
                        for c in report.conflicts:
                            cl.append(f"- [{c.type.value}] {c.description} (置信度: {c.confidence:.0%})")
                        cl.append(f"\n**裁决建议**: {report.resolution_summary}")
                        cl.append("")
                        conflict_section = "\n".join(cl)
                except Exception:
                    logger.warning("Conflict detection failed, skipping", exc_info=True)

            lines: list[str] = [f"检索到 {len(results)} 条相关内容：\n"]
            for i, r in enumerate(results, 1):
                source = resolve_source_display(r.metadata) or "未知来源"
                title = r.metadata.get("title", "")
                ct = r.metadata.get("content_type", "")
                ct_tag = f" [{ct}]" if ct else ""
                chunk_ref = r.chunk_id[:12]
                header = f"**[{i}]** {title}{ct_tag} — `{source}` (chunk: {chunk_ref})" if title else f"**[{i}]{ct_tag}** `{source}` (chunk: {chunk_ref})"
                lines.append(f"{header} (相关度: {r.score:.2f})")
                lines.append(sanitize_retrieval_text(r.text)[:1600])
                lines.append("")

            result_text = "\n".join(lines)
            if conflict_section:
                result_text += "\n" + conflict_section

            citations = [citation.to_dict() for citation in self._citation_generator.generate(results)]
            evidence_summary = build_evidence_summary(citations)
            query_trace_ids = [query_trace.trace_id] if query_trace is not None else []
            metadata = {
                "grounding_capable": True,
                "citations": citations,
                "evidence_summary": evidence_summary,
                "evidence_texts": [str(r.text or "")[:4000] for r in results],
                "source_count": len(citations),
                "collection": effective_collection,
                "query_trace_ids": query_trace_ids,
            }
            metadata.update(normalized_metadata)
            if composite_mode:
                metadata.update(
                    {
                        "composite_parent_request_id": composite_parent_request_id,
                        "composite_subtask_index": composite_subtask_index,
                        "composite_subtask_intent": composite_subtask_intent,
                    }
                )
            if query_trace is not None:
                metadata["query_trace_id"] = query_trace.trace_id

            if self._semantic_cache is not None:
                try:
                    await self._semantic_cache.put(
                        args.query,
                        result_text,
                        {
                            "collection": effective_collection,
                            **metadata,
                        },
                        collection=effective_collection,
                    )
                except Exception:
                    logger.debug("Semantic cache write failed", exc_info=True)

            return ToolResult(success=True, result_for_llm=result_text, metadata=metadata)

        except Exception as exc:
            logger.exception("KnowledgeQueryTool failed")
            metadata = {}
            if query_trace is not None:
                metadata["query_trace_id"] = query_trace.trace_id
                query_trace.record_stage(
                    "error",
                    {"phase": "knowledge_query", "error": str(exc)[:300]},
                )
            return ToolResult(success=False, error=f"知识检索失败: {exc}", metadata=metadata)
        finally:
            if query_trace is not None:
                self._trace_collector.collect(query_trace)

    def _resolve_parents(self, results: list, collection: str) -> list:
        """Replace child chunks with their parent chunks for richer context."""
        try:
            from src.libs.vector_store.vector_store_factory import VectorStoreFactory
            from src.core.settings import load_settings
            settings = self._settings or load_settings()
            store = VectorStoreFactory.create(settings, collection_name=collection)

            parent_ids: list[str] = []
            result_map: dict[str, Any] = {}
            for r in results:
                pid = r.metadata.get("parent_chunk_id")
                if pid and not r.metadata.get("is_parent"):
                    parent_ids.append(pid)
                    result_map[pid] = r
                else:
                    result_map[r.chunk_id] = r

            if not parent_ids:
                return results

            parent_records = store.get_by_ids(parent_ids)
            from src.core.types import RetrievalResult
            enriched = []
            seen: set[str] = set()
            for r in results:
                pid = r.metadata.get("parent_chunk_id")
                if pid and pid not in seen:
                    seen.add(pid)
                    pr = next((p for p in parent_records if p.get("id") == pid), None)
                    if pr and pr.get("text"):
                        enriched.append(RetrievalResult(
                            chunk_id=pid,
                            score=r.score,
                            text=pr["text"],
                            metadata={**pr.get("metadata", {}), "resolved_from_child": r.chunk_id},
                        ))
                        continue
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    enriched.append(r)
            return enriched
        except Exception as e:
            logger.warning(f"Parent resolution failed, returning original results: {e}")
            return results
