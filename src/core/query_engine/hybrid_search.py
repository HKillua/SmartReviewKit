"""Hybrid Search Engine orchestrating Dense + Sparse retrieval with RRF Fusion.

This module implements the HybridSearch class that combines:
1. QueryProcessor: Preprocess queries and extract keywords/filters
2. DenseRetriever: Semantic search using embeddings
3. SparseRetriever: Keyword search using BM25
4. RRFFusion: Combine results using Reciprocal Rank Fusion

Design Principles:
- Graceful Degradation: If one retrieval path fails, fall back to the other
- Pluggable: All components injected via constructor for testability
- Observable: TraceContext integration for debugging and monitoring
- Config-Driven: Top-k and other parameters read from settings
"""

from __future__ import annotations

import logging
import inspect
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.core.response.citation_generator import resolve_source_display
from src.core.types import ProcessedQuery, RetrievalResult

if TYPE_CHECKING:
    from src.core.query_engine.dense_retriever import DenseRetriever
    from src.core.query_engine.fusion import RRFFusion
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.sparse_retriever import SparseRetriever
    from src.libs.reranker.base_reranker import BaseReranker
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


def _snapshot_results(
    results: Optional[List[RetrievalResult]],
) -> List[Dict[str, Any]]:
    """Create a serialisable snapshot of retrieval results for trace storage.

    Args:
        results: List of RetrievalResult objects.

    Returns:
        List of dicts with chunk_id, score, full text, source.
    """
    if not results:
        return []
    return [
        {
            "chunk_id": r.chunk_id,
            "score": round(r.score, 4),
            "text": r.text or "",
            "source": resolve_source_display(r.metadata),
        }
        for r in results
    ]


@dataclass
class HybridSearchConfig:
    """Configuration for HybridSearch."""
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    enable_dense: bool = True
    enable_sparse: bool = True
    parallel_retrieval: bool = True
    metadata_filter_post: bool = True
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    rerank_enabled: bool = False
    rerank_top_k: int = 5
    mmr_enabled: bool = False
    mmr_lambda: float = 0.7
    min_score: float = 0.0
    post_rerank_min_score: float = 0.0
    empty_result_fallback_enabled: bool = True
    post_dedup_enabled: bool = True


@dataclass
class HybridSearchResult:
    """Result of a hybrid search operation.
    
    Attributes:
        results: Final ranked list of RetrievalResults
        dense_results: Results from dense retrieval (for debugging)
        sparse_results: Results from sparse retrieval (for debugging)
        dense_error: Error message if dense retrieval failed
        sparse_error: Error message if sparse retrieval failed
        used_fallback: Whether fallback mode was used
        processed_query: The processed query (for debugging)
    """
    results: List[RetrievalResult] = field(default_factory=list)
    dense_results: Optional[List[RetrievalResult]] = None
    sparse_results: Optional[List[RetrievalResult]] = None
    dense_error: Optional[str] = None
    sparse_error: Optional[str] = None
    used_fallback: bool = False
    processed_query: Optional[ProcessedQuery] = None


class HybridSearch:
    """Hybrid Search Engine combining Dense and Sparse retrieval.
    
    This class orchestrates the complete hybrid search flow:
    1. Query Processing: Extract keywords and filters from raw query
    2. Parallel Retrieval: Run Dense and Sparse retrievers concurrently
    3. Fusion: Combine results using RRF algorithm
    4. Post-Filtering: Apply metadata filters if specified
    
    Design Principles Applied:
    - Graceful Degradation: If one path fails, use results from the other
    - Pluggable: All components via dependency injection
    - Observable: TraceContext support for debugging
    - Config-Driven: All parameters from settings
    
    Example:
        >>> # Initialize components
        >>> query_processor = QueryProcessor()
        >>> dense_retriever = DenseRetriever(settings, embedding_client, vector_store)
        >>> sparse_retriever = SparseRetriever(settings, bm25_indexer, vector_store)
        >>> fusion = RRFFusion(k=60)
        >>> 
        >>> # Create HybridSearch
        >>> hybrid = HybridSearch(
        ...     settings=settings,
        ...     query_processor=query_processor,
        ...     dense_retriever=dense_retriever,
        ...     sparse_retriever=sparse_retriever,
        ...     fusion=fusion
        ... )
        >>> 
        >>> # Search
        >>> results = hybrid.search("如何配置 Azure OpenAI？", top_k=10)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[RRFFusion] = None,
        config: Optional[HybridSearchConfig] = None,
        reranker: Optional[BaseReranker] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize HybridSearch with components.
        
        Args:
            settings: Application settings for extracting configuration.
            query_processor: QueryProcessor for preprocessing queries.
            dense_retriever: DenseRetriever for semantic search.
            sparse_retriever: SparseRetriever for keyword search.
            fusion: RRFFusion for combining results.
            config: Optional HybridSearchConfig. If not provided, extracted from settings.
        
        Note:
            At least one of dense_retriever or sparse_retriever must be provided
            for search to function. The search will gracefully degrade if one
            is unavailable or fails.
        """
        self.query_processor = query_processor
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion = fusion
        self.reranker = reranker
        self.embedding_client = kwargs.get("embedding_client")
        
        # Extract config from settings or use provided/default
        self.config = config or self._extract_config(settings)
        
        logger.info(
            f"HybridSearch initialized: dense={self.dense_retriever is not None}, "
            f"sparse={self.sparse_retriever is not None}, "
            f"config={self.config}"
        )
    
    def _extract_config(self, settings: Optional[Settings]) -> HybridSearchConfig:
        """Extract HybridSearchConfig from Settings.
        
        Args:
            settings: Application settings object.
            
        Returns:
            HybridSearchConfig with values from settings or defaults.
        """
        if settings is None:
            return HybridSearchConfig()
        
        retrieval_config = getattr(settings, 'retrieval', None)
        if retrieval_config is None:
            return HybridSearchConfig()
        
        return HybridSearchConfig(
            dense_top_k=getattr(retrieval_config, 'dense_top_k', 20),
            sparse_top_k=getattr(retrieval_config, 'sparse_top_k', 20),
            fusion_top_k=getattr(retrieval_config, 'fusion_top_k', 10),
            enable_dense=True,
            enable_sparse=True,
            parallel_retrieval=True,
            metadata_filter_post=True,
            dense_weight=getattr(retrieval_config, 'dense_weight', 1.0),
            sparse_weight=getattr(retrieval_config, 'sparse_weight', 1.0),
            rerank_enabled=getattr(retrieval_config, 'rerank_enabled', False),
            rerank_top_k=getattr(retrieval_config, 'rerank_top_k', 5),
            mmr_enabled=getattr(retrieval_config, 'mmr_enabled', False),
            mmr_lambda=getattr(retrieval_config, 'mmr_lambda', 0.7),
            min_score=getattr(retrieval_config, 'min_score', 0.0),
            post_rerank_min_score=getattr(
                retrieval_config,
                'post_rerank_min_score',
                getattr(retrieval_config, 'min_score', 0.0),
            ),
            empty_result_fallback_enabled=getattr(
                retrieval_config,
                'empty_result_fallback_enabled',
                True,
            ),
            post_dedup_enabled=getattr(retrieval_config, 'post_dedup_enabled', True),
        )
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        return_details: bool = False,
        query_vector: Optional[List[float]] = None,
        fast_mode: bool = False,
    ) -> List[RetrievalResult] | HybridSearchResult:
        """Perform hybrid search combining Dense and Sparse retrieval.
        
        Args:
            query: The search query string.
            top_k: Maximum number of results to return. If None, uses config.fusion_top_k.
            filters: Optional metadata filters (e.g., {"collection": "docs"}).
            trace: Optional TraceContext for observability.
            return_details: If True, return HybridSearchResult with debug info.
        
        Returns:
            If return_details=False: List of RetrievalResult sorted by relevance.
            If return_details=True: HybridSearchResult with full details.
        
        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If both retrievers fail or are unavailable.
        
        Example:
            >>> results = hybrid.search("Azure configuration", top_k=5)
            >>> for r in results:
            ...     print(f"[{r.score:.4f}] {r.chunk_id}: {r.text[:50]}...")
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        
        effective_top_k = top_k if top_k is not None else self.config.fusion_top_k
        # Fast mode still keeps cross-encoder rerank for answer quality and
        # trace stability; the main latency win comes from skipping MMR and
        # later extra LLM steps rather than removing rerank entirely.
        rerank_enabled = self.config.rerank_enabled
        mmr_enabled = self.config.mmr_enabled and not fast_mode
        
        logger.debug(f"HybridSearch: query='{query[:50]}...', top_k={effective_top_k}")
        
        # Step 1: Process query
        _t0 = time.monotonic()
        processed_query = self._process_query(query)
        _elapsed = (time.monotonic() - _t0) * 1000.0
        if trace is not None:
            trace.record_stage("query_processing", {
                "method": "query_processor",
                "original_query": query,
                "keywords": processed_query.keywords,
            }, elapsed_ms=_elapsed)
        
        # Merge explicit filters with query-extracted filters
        merged_filters = self._merge_filters(processed_query.filters, filters)
        
        # Step 2: Run retrievals (pass query_vector for HyDE support)
        dense_results, sparse_results, dense_error, sparse_error = self._run_retrievals(
            processed_query=processed_query,
            filters=merged_filters,
            trace=trace,
            query_vector=query_vector,
        )
        
        # Step 3: Handle fallback scenarios
        used_fallback = False
        if dense_error and sparse_error:
            # Both failed - raise error
            raise RuntimeError(
                f"Both retrieval paths failed. "
                f"Dense error: {dense_error}. Sparse error: {sparse_error}"
            )
        elif dense_error:
            # Dense failed, use sparse only
            logger.warning(f"Dense retrieval failed, using sparse only: {dense_error}")
            used_fallback = True
            fused_results = sparse_results or []
        elif sparse_error:
            # Sparse failed, use dense only
            logger.warning(f"Sparse retrieval failed, using dense only: {sparse_error}")
            used_fallback = True
            fused_results = dense_results or []
        elif not dense_results and not sparse_results:
            # Both succeeded but returned empty
            fused_results = []
        else:
            # Step 4: Fuse results
            fused_results = self._fuse_results(
                dense_results=dense_results or [],
                sparse_results=sparse_results or [],
                top_k=effective_top_k,
                trace=trace,
            )
        
        # Step 5: Apply post-fusion metadata filters (if any)
        if merged_filters and self.config.metadata_filter_post:
            before_filter_count = len(fused_results)
            fused_results = self._apply_metadata_filters(fused_results, merged_filters)
            if trace is not None:
                trace.record_stage(
                    "metadata_filter_post",
                    {
                        "filters": merged_filters,
                        "before_count": before_filter_count,
                        "after_count": len(fused_results),
                        "cleared_results": before_filter_count > 0 and len(fused_results) == 0,
                    },
                )
        
        pre_postprocess_results = list(fused_results)

        # Step 6: Reranker (if enabled and available)
        rerank_applied = rerank_enabled and self.reranker is not None
        if rerank_applied:
            fused_results = self._apply_reranker(query, fused_results, trace)

        # Step 7: MMR diversity (if enabled)
        if mmr_enabled and self.embedding_client is not None and fused_results:
            fused_results = self._apply_mmr(query, fused_results, effective_top_k, trace)

        pre_filter_results = list(fused_results)

        # Step 8: Min score filtering
        if rerank_applied and self.config.post_rerank_min_score > 0:
            fused_results = [
                r for r in fused_results
                if r.score >= self.config.post_rerank_min_score
            ]

        # Step 9: Post-retrieval semantic dedup
        if self.config.post_dedup_enabled and len(fused_results) > 1:
            fused_results = self._apply_post_dedup(fused_results)

        if (
            not fused_results
            and self.config.empty_result_fallback_enabled
            and pre_filter_results
        ):
            fused_results = pre_filter_results
            if trace is not None:
                trace.record_stage(
                    "empty_result_fallback",
                    {
                        "reason": "post_filtering_cleared_results",
                        "fallback_count": len(fused_results),
                        "rerank_applied": rerank_applied,
                        "post_rerank_min_score": self.config.post_rerank_min_score,
                        "post_dedup_enabled": self.config.post_dedup_enabled,
                    },
                )
        elif (
            not fused_results
            and self.config.empty_result_fallback_enabled
            and pre_postprocess_results
        ):
            fused_results = pre_postprocess_results
            if trace is not None:
                trace.record_stage(
                    "empty_result_fallback",
                    {
                        "reason": "post_rerank_pipeline_empty",
                        "fallback_count": len(fused_results),
                        "rerank_applied": rerank_applied,
                        "post_rerank_min_score": self.config.post_rerank_min_score,
                        "post_dedup_enabled": self.config.post_dedup_enabled,
                    },
                )

        # Step 10: Limit to top_k
        final_results = fused_results[:effective_top_k]

        # Record retrieval quality summary for observability
        if trace is not None and final_results:
            scores = [r.score for r in final_results]
            _t0_summary = time.monotonic()
            trace.record_stage("retrieval_summary", {
                "method": "summary",
                "total_results": len(final_results),
                "score_max": round(max(scores), 4),
                "score_min": round(min(scores), 4),
                "score_mean": round(sum(scores) / len(scores), 4),
                "reranker_used": rerank_applied,
                "mmr_used": mmr_enabled,
                "dedup_used": self.config.post_dedup_enabled,
                "fast_mode": fast_mode,
                "pipeline": "dense+sparse+rrf",
                "post_rerank_min_score": self.config.post_rerank_min_score,
            }, elapsed_ms=(time.monotonic() - _t0_summary) * 1000.0)
        
        logger.debug(f"HybridSearch: returning {len(final_results)} results")
        
        if return_details:
            return HybridSearchResult(
                results=final_results,
                dense_results=dense_results,
                sparse_results=sparse_results,
                dense_error=dense_error,
                sparse_error=sparse_error,
                used_fallback=used_fallback,
                processed_query=processed_query,
            )
        
        return final_results
    
    def _process_query(self, query: str) -> ProcessedQuery:
        """Process raw query using QueryProcessor.
        
        Args:
            query: Raw query string.
            
        Returns:
            ProcessedQuery with keywords and filters.
        """
        if self.query_processor is None:
            # Fallback: create basic ProcessedQuery
            logger.warning("No QueryProcessor configured, using basic tokenization")
            keywords = query.split()
            return ProcessedQuery(
                original_query=query,
                keywords=keywords,
                filters={},
            )
        
        return self.query_processor.process(query)
    
    def _merge_filters(
        self,
        query_filters: Dict[str, Any],
        explicit_filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge query-extracted filters with explicit filters.
        
        Explicit filters take precedence over query-extracted filters.
        
        Args:
            query_filters: Filters extracted from query by QueryProcessor.
            explicit_filters: Filters passed explicitly to search().
            
        Returns:
            Merged filter dictionary.
        """
        merged = query_filters.copy() if query_filters else {}
        if explicit_filters:
            merged.update(explicit_filters)
        return merged
    
    def _run_retrievals(
        self,
        processed_query: ProcessedQuery,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
        query_vector: Optional[List[float]] = None,
    ) -> Tuple[
        Optional[List[RetrievalResult]],
        Optional[List[RetrievalResult]],
        Optional[str],
        Optional[str],
    ]:
        """Run Dense and Sparse retrievals.
        
        Runs in parallel if configured, otherwise sequentially.
        
        Args:
            processed_query: The processed query with keywords.
            filters: Merged filters to apply.
            trace: Optional TraceContext.
            
        Returns:
            Tuple of (dense_results, sparse_results, dense_error, sparse_error).
        """
        dense_results: Optional[List[RetrievalResult]] = None
        sparse_results: Optional[List[RetrievalResult]] = None
        dense_error: Optional[str] = None
        sparse_error: Optional[str] = None
        
        # Determine what to run
        run_dense = (
            self.config.enable_dense 
            and self.dense_retriever is not None
        )
        run_sparse = (
            self.config.enable_sparse 
            and self.sparse_retriever is not None
            and processed_query.keywords  # Need keywords for sparse
        )
        
        if not run_dense and not run_sparse:
            # Nothing to run
            if self.dense_retriever is None and self.sparse_retriever is None:
                dense_error = "No retriever configured"
                sparse_error = "No retriever configured"
            return dense_results, sparse_results, dense_error, sparse_error
        
        if self.config.parallel_retrieval and run_dense and run_sparse:
            dense_results, sparse_results, dense_error, sparse_error = (
                self._run_parallel_retrievals(processed_query, filters, trace, query_vector)
            )
        else:
            if run_dense:
                dense_results, dense_error = self._run_dense_retrieval(
                    processed_query.original_query, filters, trace, query_vector
                )
            
            if run_sparse:
                sparse_results, sparse_error = self._run_sparse_retrieval(
                    processed_query.keywords, filters, trace
                )
        
        return dense_results, sparse_results, dense_error, sparse_error
    
    def _run_parallel_retrievals(
        self,
        processed_query: ProcessedQuery,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
        query_vector: Optional[List[float]] = None,
    ) -> Tuple[
        Optional[List[RetrievalResult]],
        Optional[List[RetrievalResult]],
        Optional[str],
        Optional[str],
    ]:
        """Run Dense and Sparse retrievals in parallel using ThreadPoolExecutor.
        
        Args:
            processed_query: The processed query.
            filters: Filters to apply.
            trace: Optional TraceContext.
            
        Returns:
            Tuple of (dense_results, sparse_results, dense_error, sparse_error).
        """
        dense_results: Optional[List[RetrievalResult]] = None
        sparse_results: Optional[List[RetrievalResult]] = None
        dense_error: Optional[str] = None
        sparse_error: Optional[str] = None
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            futures['dense'] = executor.submit(
                self._run_dense_retrieval,
                processed_query.original_query,
                filters,
                trace,
                query_vector,
            )
            
            # Submit sparse retrieval
            futures['sparse'] = executor.submit(
                self._run_sparse_retrieval,
                processed_query.keywords,
                filters,
                trace,
            )
            
            # Collect results
            for name, future in futures.items():
                try:
                    results, error = future.result(timeout=30)
                    if name == 'dense':
                        dense_results = results
                        dense_error = error
                    else:
                        sparse_results = results
                        sparse_error = error
                except Exception as e:
                    error_msg = f"{name} retrieval failed with exception: {e}"
                    logger.error(error_msg)
                    if name == 'dense':
                        dense_error = error_msg
                    else:
                        sparse_error = error_msg
        
        return dense_results, sparse_results, dense_error, sparse_error
    
    def _run_dense_retrieval(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
        query_vector: Optional[List[float]] = None,
    ) -> Tuple[Optional[List[RetrievalResult]], Optional[str]]:
        """Run dense retrieval with error handling.
        
        Args:
            query: Original query string.
            filters: Filters to apply.
            trace: Optional TraceContext.
            
        Returns:
            Tuple of (results, error). If successful, error is None.
        """
        if self.dense_retriever is None:
            return None, "Dense retriever not configured"
        
        try:
            _t0 = time.monotonic()
            retrieve_kwargs = {
                "query": query,
                "top_k": self.config.dense_top_k,
                "filters": filters,
                "trace": trace,
            }
            try:
                retrieve_signature = inspect.signature(self.dense_retriever.retrieve)
            except (TypeError, ValueError):
                retrieve_signature = None
            if (
                query_vector is not None
                and retrieve_signature is not None
                and "query_vector" in retrieve_signature.parameters
            ):
                retrieve_kwargs["query_vector"] = query_vector
            results = self.dense_retriever.retrieve(**retrieve_kwargs)
            _elapsed = (time.monotonic() - _t0) * 1000.0
            if trace is not None:
                trace.record_stage("dense_retrieval", {
                    "method": "dense",
                    "provider": getattr(self.dense_retriever, 'provider_name', 'unknown'),
                    "top_k": self.config.dense_top_k,
                    "result_count": len(results) if results else 0,
                    "chunks": _snapshot_results(results),
                }, elapsed_ms=_elapsed)
            return results, None
        except Exception as e:
            error_msg = f"Dense retrieval error: {e}"
            logger.error(error_msg)
            if trace is not None:
                trace.record_stage("dense_retrieval", {
                    "method": "dense",
                    "error": error_msg,
                    "result_count": 0,
                })
            return None, error_msg
    
    def _run_sparse_retrieval(
        self,
        keywords: List[str],
        filters: Optional[Dict[str, Any]],
        trace: Optional[Any],
    ) -> Tuple[Optional[List[RetrievalResult]], Optional[str]]:
        """Run sparse retrieval with error handling.
        
        Args:
            keywords: List of keywords from QueryProcessor.
            filters: Filters to apply.
            trace: Optional TraceContext.
            
        Returns:
            Tuple of (results, error). If successful, error is None.
        """
        if self.sparse_retriever is None:
            return None, "Sparse retriever not configured"
        
        if not keywords:
            return [], None  # No keywords, return empty (not an error)
        
        try:
            # Extract collection from filters if present
            collection = filters.get('collection') if filters else None
            
            _t0 = time.monotonic()
            results = self.sparse_retriever.retrieve(
                keywords=keywords,
                top_k=self.config.sparse_top_k,
                collection=collection,
                trace=trace,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            if trace is not None:
                trace.record_stage("sparse_retrieval", {
                    "method": "bm25",
                    "keyword_count": len(keywords),
                    "top_k": self.config.sparse_top_k,
                    "result_count": len(results) if results else 0,
                    "chunks": _snapshot_results(results),
                }, elapsed_ms=_elapsed)
            return results, None
        except Exception as e:
            error_msg = f"Sparse retrieval error: {e}"
            logger.error(error_msg)
            return None, error_msg
    
    def _fuse_results(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
        trace: Optional[Any],
    ) -> List[RetrievalResult]:
        """Fuse Dense and Sparse results using RRF.
        
        Args:
            dense_results: Results from dense retrieval.
            sparse_results: Results from sparse retrieval.
            top_k: Number of results to return after fusion.
            trace: Optional TraceContext.
            
        Returns:
            Fused and ranked list of RetrievalResults.
        """
        if self.fusion is None:
            # Fallback: interleave results (simple round-robin)
            logger.warning("No fusion configured, using simple interleave")
            return self._interleave_results(dense_results, sparse_results, top_k)
        
        # Build ranking lists for RRF
        ranking_lists = []
        if dense_results:
            ranking_lists.append(dense_results)
        if sparse_results:
            ranking_lists.append(sparse_results)
        
        if not ranking_lists:
            return []
        
        if len(ranking_lists) == 1:
            return ranking_lists[0][:top_k]
        
        weights = [self.config.dense_weight, self.config.sparse_weight]
        weights = weights[:len(ranking_lists)]
        
        _t0 = time.monotonic()
        if hasattr(self.fusion, "fuse_with_weights"):
            fused = self.fusion.fuse_with_weights(
                ranking_lists=ranking_lists,
                weights=weights,
                top_k=top_k,
                trace=trace,
            )
        else:
            fused = self.fusion.fuse(
                ranking_lists=ranking_lists,
                top_k=top_k,
                trace=trace,
            )
        _elapsed = (time.monotonic() - _t0) * 1000.0
        if trace is not None:
            trace.record_stage("fusion", {
                "method": "rrf",
                "input_lists": len(ranking_lists),
                "top_k": top_k,
                "result_count": len(fused),
                "chunks": _snapshot_results(fused),
            }, elapsed_ms=_elapsed)
        return fused
    
    def _interleave_results(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Simple interleave fallback when no fusion is configured.
        
        Args:
            dense_results: Results from dense retrieval.
            sparse_results: Results from sparse retrieval.
            top_k: Maximum results to return.
            
        Returns:
            Interleaved results, deduped by chunk_id.
        """
        seen_ids = set()
        interleaved = []
        
        d_idx, s_idx = 0, 0
        while len(interleaved) < top_k and (d_idx < len(dense_results) or s_idx < len(sparse_results)):
            # Alternate between dense and sparse
            if d_idx < len(dense_results):
                r = dense_results[d_idx]
                d_idx += 1
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    interleaved.append(r)
            
            if len(interleaved) >= top_k:
                break
            
            if s_idx < len(sparse_results):
                r = sparse_results[s_idx]
                s_idx += 1
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    interleaved.append(r)
        
        return interleaved
    
    def _apply_metadata_filters(
        self,
        results: List[RetrievalResult],
        filters: Dict[str, Any],
    ) -> List[RetrievalResult]:
        """Apply metadata filters to results (post-fusion fallback).
        
        This is a backup filter mechanism for cases where the underlying
        storage doesn't fully support the filter syntax.
        
        Args:
            results: Results to filter.
            filters: Filter conditions to apply.
            
        Returns:
            Filtered results.
        """
        if not filters:
            return results
        
        filtered = []
        for result in results:
            if self._matches_filters(result.metadata, filters):
                filtered.append(result)
        
        return filtered
    
    def _apply_mmr(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
        trace: Optional[Any],
    ) -> List[RetrievalResult]:
        """Apply MMR diversity reranking, reusing cached embeddings when available."""
        if not results or self.embedding_client is None:
            return results
        try:
            from src.core.query_engine.mmr import mmr_rerank
            _t0 = time.monotonic()
            q_vecs = self.embedding_client.embed([query])

            cached_count = 0
            c_vecs: List[List[float]] = []
            texts_to_embed: List[tuple[int, str]] = []
            for i, r in enumerate(results):
                if r.embedding is not None:
                    c_vecs.append(r.embedding)
                    cached_count += 1
                else:
                    c_vecs.append([])
                    texts_to_embed.append((i, r.text))

            if texts_to_embed:
                new_vecs = self.embedding_client.embed([t for _, t in texts_to_embed])
                for (idx, _), vec in zip(texts_to_embed, new_vecs):
                    c_vecs[idx] = vec

            mmr_results = mmr_rerank(
                query_embedding=q_vecs[0],
                candidates=results,
                candidate_embeddings=c_vecs,
                top_k=top_k,
                lambda_param=self.config.mmr_lambda,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            if trace is not None:
                trace.record_stage("mmr", {
                    "lambda": self.config.mmr_lambda,
                    "input_count": len(results),
                    "output_count": len(mmr_results),
                    "cached_embeddings": cached_count,
                }, elapsed_ms=_elapsed)
            logger.debug("MMR: %d -> %d results (reused %d cached embeddings)",
                         len(results), len(mmr_results), cached_count)
            return mmr_results
        except Exception as exc:
            logger.warning("MMR failed, returning original order: %s", exc)
            return results

    def _apply_post_dedup(
        self,
        results: List[RetrievalResult],
        threshold: int = 8,
    ) -> List[RetrievalResult]:
        """Remove near-duplicate results using SimHash after retrieval."""
        if len(results) <= 1:
            return results
        try:
            from src.ingestion.transform.chunk_dedup import simhash, hamming_distance
            kept: List[RetrievalResult] = []
            seen_hashes: List[int] = []
            for r in results:
                fp = simhash(r.text)
                is_dup = any(hamming_distance(fp, h) <= threshold for h in seen_hashes)
                if not is_dup:
                    seen_hashes.append(fp)
                    kept.append(r)
            if len(kept) < len(results):
                logger.debug("Post-dedup: %d -> %d results", len(results), len(kept))
            return kept
        except Exception as exc:
            logger.warning("Post-dedup failed: %s", exc)
            return results

    def _apply_reranker(
        self,
        query: str,
        results: List[RetrievalResult],
        trace: Optional[Any],
    ) -> List[RetrievalResult]:
        """Rerank results using the injected reranker."""
        if not results:
            return results
        try:
            _t0 = time.monotonic()
            candidates = [
                {"chunk_id": r.chunk_id, "text": r.text, "metadata": r.metadata}
                for r in results
            ]
            reranked = self.reranker.rerank(  # type: ignore[union-attr]
                query=query,
                candidates=candidates,
                top_k=self.config.rerank_top_k,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0

            reranked_results = [
                RetrievalResult(
                    chunk_id=c["chunk_id"],
                    score=c.get("rerank_score", 0.0),
                    text=c.get("text", ""),
                    metadata=c.get("metadata", {}),
                )
                for c in reranked
            ]
            if trace is not None:
                trace.record_stage(
                    "rerank",
                    {
                        "method": type(self.reranker).__name__,
                        "provider": type(self.reranker).__name__,
                        "input_count": len(results),
                        "output_count": len(reranked_results),
                        "used_fallback": False,
                    },
                    elapsed_ms=_elapsed,
                )
            logger.debug("Reranker: %d -> %d results", len(results), len(reranked_results))
            return reranked_results
        except Exception as exc:
            if trace is not None:
                trace.record_stage(
                    "rerank",
                    {
                        "method": type(self.reranker).__name__ if self.reranker is not None else "none",
                        "provider": type(self.reranker).__name__ if self.reranker is not None else "none",
                        "input_count": len(results),
                        "output_count": len(results),
                        "used_fallback": True,
                        "error": str(exc)[:300],
                    },
                )
            logger.warning("Reranker failed, returning original order: %s", exc)
            return results

    def _matches_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> bool:
        """Check if metadata matches all filter conditions.
        
        Args:
            metadata: Result metadata.
            filters: Filter conditions.
            
        Returns:
            True if all filters match, False otherwise.
        """
        for key, value in filters.items():
            if isinstance(value, dict):
                if "$in" in value:
                    candidate_values = value.get("$in") or []
                    if not isinstance(candidate_values, list):
                        candidate_values = [candidate_values]
                    meta_value = metadata.get(key)
                    if isinstance(meta_value, list):
                        if not set(meta_value) & set(candidate_values):
                            return False
                    elif meta_value not in candidate_values:
                        return False
                    continue
                if "$eq" in value:
                    if metadata.get(key) != value.get("$eq"):
                        return False
                    continue
                logger.debug("Unsupported metadata filter operator for key '%s': %s", key, value)
                return False
            if key == "collection":
                # Collection might be in different metadata keys
                meta_collection = (
                    metadata.get("collection") 
                    or metadata.get("source_collection")
                )
                if meta_collection != value:
                    return False
            elif key == "doc_type":
                if metadata.get("doc_type") != value:
                    return False
            elif key == "tags":
                # Tags is a list - check intersection
                meta_tags = metadata.get("tags", [])
                if not isinstance(value, list):
                    value = [value]
                if not set(meta_tags) & set(value):
                    return False
            elif key == "source_path":
                # Partial match for path
                source = metadata.get("source_path", "")
                if value not in source:
                    return False
            else:
                # Generic exact match
                if metadata.get(key) != value:
                    return False
        
        return True


def create_hybrid_search(
    settings: Optional[Settings] = None,
    query_processor: Optional[QueryProcessor] = None,
    dense_retriever: Optional[DenseRetriever] = None,
    sparse_retriever: Optional[SparseRetriever] = None,
    fusion: Optional[RRFFusion] = None,
) -> HybridSearch:
    """Factory function to create HybridSearch with default components.
    
    This is a convenience function that creates a HybridSearch with
    default RRFFusion if not provided.
    
    Args:
        settings: Application settings.
        query_processor: QueryProcessor instance.
        dense_retriever: DenseRetriever instance.
        sparse_retriever: SparseRetriever instance.
        fusion: RRFFusion instance. If None, creates default with k=60.
        
    Returns:
        Configured HybridSearch instance.
    
    Example:
        >>> hybrid = create_hybrid_search(
        ...     settings=settings,
        ...     query_processor=QueryProcessor(),
        ...     dense_retriever=dense_retriever,
        ...     sparse_retriever=sparse_retriever,
        ... )
    """
    # Create default fusion if not provided
    if fusion is None:
        from src.core.query_engine.fusion import RRFFusion
        rrf_k = 60
        if settings is not None:
            retrieval_config = getattr(settings, 'retrieval', None)
            if retrieval_config is not None:
                rrf_k = getattr(retrieval_config, 'rrf_k', 60)
        fusion = RRFFusion(k=rrf_k)
    
    return HybridSearch(
        settings=settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion,
    )
