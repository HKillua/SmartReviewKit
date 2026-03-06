"""Knowledge retrieval tool — wraps the existing HybridSearch infrastructure."""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.tools.base import Tool
from src.agent.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


class KnowledgeQueryArgs(BaseModel):
    query: str = Field(..., description="检索查询文本")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    collection: str = Field(default="computer_network", description="知识库 collection 名称，默认 computer_network，必须使用英文")
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
    ) -> None:
        self._settings = settings
        self._hybrid_search = hybrid_search
        self._reranker = reranker
        self._query_enhancer = query_enhancer
        self._initialized = hybrid_search is not None
        self._current_collection: Optional[str] = None
        self._embedding_client: Any = None

    @property
    def name(self) -> str:
        return "knowledge_query"

    @property
    def description(self) -> str:
        return "从知识库中检索与查询相关的课程内容，返回带引用的文本片段"

    def get_args_schema(self) -> type[KnowledgeQueryArgs]:
        return KnowledgeQueryArgs

    def _ensure_initialized(self, collection: str = "computer_network") -> None:
        if self._initialized and self._current_collection == collection:
            return
        if self._settings is None:
            raise RuntimeError("KnowledgeQueryTool requires settings or a pre-built HybridSearch")

        from src.core.settings import load_settings, resolve_path
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.reranker import create_core_reranker
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

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
        bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
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
        try:
            self._ensure_initialized(args.collection)

            effective_query = args.query
            hyde_vector = None

            if self._query_enhancer is not None:
                try:
                    effective_query = await self._query_enhancer.rewrite(args.query)
                except Exception:
                    logger.warning("Query rewrite failed, using original")
                try:
                    hyde_vector = await self._query_enhancer.hyde_embed(args.query)
                except Exception:
                    logger.warning("HyDE failed, using standard embedding")

            search_kwargs: dict[str, Any] = {}
            if args.content_type:
                search_kwargs["filters"] = {"content_type": args.content_type}
            if hyde_vector is not None:
                search_kwargs["query_vector"] = hyde_vector

            # Multi-query: decompose and merge
            all_results = []
            queries = [effective_query]
            if self._query_enhancer is not None:
                try:
                    sub_queries = await self._query_enhancer.decompose(args.query)
                    if len(sub_queries) > 1:
                        queries = sub_queries
                except Exception:
                    logger.warning("Multi-query decompose failed")

            for q in queries:
                results = self._hybrid_search.search(
                    query=q,
                    top_k=args.top_k,
                    **search_kwargs,
                )
                if isinstance(results, list):
                    all_results.extend(results)

            # Deduplicate by chunk_id, keep highest score
            seen: dict[str, Any] = {}
            for r in all_results:
                if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
                    seen[r.chunk_id] = r
            results = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:args.top_k]

            if not results:
                return ToolResult(
                    success=True,
                    result_for_llm="未找到与查询相关的知识库内容。请尝试换一种表述重新提问。",
                )

            if args.use_parent:
                results = self._resolve_parents(results, args.collection)

            lines: list[str] = [f"检索到 {len(results)} 条相关内容：\n"]
            for i, r in enumerate(results, 1):
                source = r.metadata.get("source_path", "未知来源")
                title = r.metadata.get("title", "")
                ct = r.metadata.get("content_type", "")
                ct_tag = f" [{ct}]" if ct else ""
                chunk_ref = r.chunk_id[:12]
                header = f"**[{i}]** {title}{ct_tag} — `{source}` (chunk: {chunk_ref})" if title else f"**[{i}]{ct_tag}** `{source}` (chunk: {chunk_ref})"
                lines.append(f"{header} (相关度: {r.score:.2f})")
                lines.append(r.text[:800])
                lines.append("")

            return ToolResult(success=True, result_for_llm="\n".join(lines))

        except Exception as exc:
            logger.exception("KnowledgeQueryTool failed")
            return ToolResult(success=False, error=f"知识检索失败: {exc}")

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
