#!/usr/bin/env python3
"""Offline retrieval quality evaluation script.

Computes Hit Rate, MRR, and NDCG on a labelled evaluation set
(query → expected_chunk_ids) and prints a summary report.

Usage:
    python scripts/eval_retrieval.py --eval-file data/eval/retrieval_eval.jsonl
    python scripts/eval_retrieval.py --eval-file data/eval/retrieval_eval.jsonl --top-k 10

Evaluation file format (JSONL, one object per line):
    {"query": "TCP三次握手", "expected_ids": ["chunk_001", "chunk_002"]}
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import replace
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


def hit_rate(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    return 1.0 if set(retrieved_ids) & set(expected_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    expected = set(expected_ids)
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in expected:
            return 1.0 / rank
    return 0.0


def ndcg(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    expected = set(expected_ids)
    dcg = 0.0
    seen_hits: set[str] = set()
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in expected and rid not in seen_hits:
            dcg += 1.0 / math.log2(i + 2)
            seen_hits.add(rid)
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def load_eval_set(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _normalize_source_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "/" in text or "\\" in text:
        text = Path(text).name
    return text.casefold()


def _extract_result_sources(results: list[Any]) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for result in results:
        metadata = getattr(result, "metadata", {}) or {}
        source = (
            metadata.get("source_label")
            or metadata.get("original_filename")
            or metadata.get("source")
            or metadata.get("source_path")
        )
        normalized = _normalize_source_label(source)
        if normalized and normalized not in seen:
            seen.add(normalized)
            sources.append(normalized)
    return sources


def _prepare_settings(settings):
    retrieval_cfg = replace(
        settings.retrieval,
        query_rewrite_enabled=False,
        hyde_enabled=False,
        multi_query_enabled=False,
        min_score=0.0,
    )
    return replace(settings, retrieval=retrieval_cfg)


class _EvalSearchAdapter:
    def __init__(self, source_aware_search, task_intent: str) -> None:
        self._source_aware_search = source_aware_search
        self._task_intent = task_intent

    def search(self, *, query: str, top_k: int, filters=None):
        return self._source_aware_search.search(
            query=query,
            task_intent=self._task_intent,
            top_k=top_k,
            filters=filters,
        )


def _resolve_underlying_hybrid_search(search):
    source_aware = getattr(search, "_source_aware_search", None)
    if source_aware is None:
        return None
    return getattr(source_aware, "_hybrid_search", None) or getattr(source_aware, "hybrid_search", None)


class _FallbackSearchAdapter:
    """Try hybrid search first and fall back to sparse-only eval on runtime errors."""

    def __init__(self, primary_search, fallback_search) -> None:
        self._primary_search = primary_search
        self._fallback_search = fallback_search

    def search(self, *, query: str, top_k: int, filters=None):
        try:
            results = self._primary_search.search(query=query, top_k=top_k, filters=filters)
        except Exception as exc:
            logger.warning("Hybrid search failed at runtime, falling back to sparse eval: %s", exc)
            return self._fallback_search.search(query=query, top_k=top_k, filters=filters)
        if self._is_empty(results):
            logger.warning("Hybrid search returned no results, falling back to sparse eval")
            return self._fallback_search.search(query=query, top_k=top_k, filters=filters)
        return results

    @staticmethod
    def _is_empty(results) -> bool:
        if isinstance(results, list):
            return len(results) == 0
        payload = getattr(results, "results", None)
        return payload is None or len(payload) == 0


def _build_hybrid_eval_search(settings, collection: str, task_intent: str, timeout_seconds: float):
    def _init():
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.hybrid_search import HybridSearch
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.fusion import RRFFusion
        from src.core.query_engine.query_router import QueryRouter
        from src.core.query_engine.source_aware_search import SourceAwareSearch
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.embedding.cached_embedding import CachedEmbedding
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        from src.storage.runtime import create_sparse_index

        raw_embedding = EmbeddingFactory.create(settings)
        embedding = CachedEmbedding(raw_embedding, max_size=settings.retrieval.embedding_cache_size)
        vector_store = VectorStoreFactory.create(settings, collection_name=collection)
        dense = create_dense_retriever(settings=settings, embedding_client=embedding, vector_store=vector_store)
        bm25 = create_sparse_index(settings, collection=collection)
        sparse = create_sparse_retriever(settings=settings, bm25_indexer=bm25, vector_store=vector_store)
        sparse.default_collection = collection
        fusion = RRFFusion(k=settings.retrieval.rrf_k)
        hybrid = HybridSearch(
            settings=settings,
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=fusion,
        )
        hybrid.embedding_client = embedding
        query_router = QueryRouter(fallback_to_llm=True)
        source_aware_search = SourceAwareSearch(hybrid_search=hybrid, query_router=query_router)
        return _EvalSearchAdapter(source_aware_search, task_intent)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_init)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError(
            f"Hybrid retrieval initialization timed out after {timeout_seconds:.1f}s"
        ) from exc
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if future.done():
            executor.shutdown(wait=False, cancel_futures=True)


def _attach_reranker_with_timeout(search, settings, timeout_seconds: float) -> None:
    def _init():
        from src.libs.reranker.reranker_factory import RerankerFactory

        return RerankerFactory.create(settings)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_init)
    try:
        target = _resolve_underlying_hybrid_search(search)
        if target is None:
            raise AttributeError("unable to resolve underlying HybridSearch from eval adapter")
        target.reranker = future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        logger.warning(
            "Reranker unavailable for retrieval eval: initialization timed out after %.1fs",
            timeout_seconds,
        )
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:
        logger.warning("Reranker unavailable for retrieval eval: %s", exc)
        executor.shutdown(wait=False, cancel_futures=True)
    finally:
        if future.done():
            executor.shutdown(wait=False, cancel_futures=True)


def run_evaluation(
    eval_file: str,
    top_k: int = 5,
    collection: str = "computer_network",
    settings_path: str = "config/settings.storage_stack.yaml",
    task_intent: str = "knowledge_query",
    mode: str = "sparse",
    enable_rerank: bool = False,
    hybrid_init_timeout: float = 10.0,
):
    from src.core.settings import load_settings

    settings = _prepare_settings(load_settings(settings_path))
    if mode == "sparse":
        from src.observability.evaluation.source_eval_search import SparseSourceEvalSearch

        search = SparseSourceEvalSearch(settings=settings, collection=collection)
    else:
        from src.observability.evaluation.source_eval_search import SparseSourceEvalSearch

        sparse_fallback = SparseSourceEvalSearch(settings=settings, collection=collection)
        try:
            primary_search = _build_hybrid_eval_search(
                settings=settings,
                collection=collection,
                task_intent=task_intent,
                timeout_seconds=hybrid_init_timeout,
            )
            if enable_rerank:
                _attach_reranker_with_timeout(primary_search, settings, hybrid_init_timeout)
            search = _FallbackSearchAdapter(primary_search, sparse_fallback)
        except Exception as exc:
            logger.warning("Hybrid retrieval unavailable, falling back to sparse eval: %s", exc)
            search = sparse_fallback

    eval_items = load_eval_set(eval_file)
    print(f"\n{'='*60}")
    print(f"  Retrieval Quality Evaluation")
    print(f"  Dataset: {eval_file} ({len(eval_items)} queries)")
    print(f"  Collection: {collection}  |  top_k: {top_k}")
    print(f"  Settings: {settings_path}  |  mode: {mode}  |  task_intent: {task_intent}")
    print(f"{'='*60}\n")

    total_hr, total_mrr, total_ndcg = 0.0, 0.0, 0.0
    basis = "chunk_id"

    for i, item in enumerate(eval_items, 1):
        query = item["query"]
        expected_ids = item.get("expected_ids", [])
        expected_sources = [_normalize_source_label(src) for src in item.get("expected_sources", [])]
        results_payload = search.search(query=query, top_k=top_k, filters={"collection": collection})
        results = results_payload if isinstance(results_payload, list) else results_payload.results
        if expected_ids:
            basis = "chunk_id"
            expected = expected_ids
            retrieved = [r.chunk_id for r in results]
        else:
            basis = "source"
            expected = expected_sources
            retrieved = _extract_result_sources(results)

        hr = hit_rate(retrieved, expected)
        mrr = reciprocal_rank(retrieved, expected)
        n = ndcg(retrieved, expected, top_k)
        total_hr += hr
        total_mrr += mrr
        total_ndcg += n

        status = "HIT" if hr > 0 else "MISS"
        print(f"  [{i:3d}] {status}  MRR={mrr:.3f}  NDCG@{top_k}={n:.3f}  query=\"{query[:50]}\"")

    n_queries = len(eval_items)
    print(f"\n{'='*60}")
    print(f"  SUMMARY  (n={n_queries})")
    print(f"    Basis      : {basis}")
    print(f"    Hit Rate   : {total_hr / n_queries:.4f}")
    print(f"    MRR        : {total_mrr / n_queries:.4f}")
    print(f"    NDCG@{top_k:<2d}    : {total_ndcg / n_queries:.4f}")
    print(f"{'='*60}\n")

    return {
        "hit_rate": total_hr / n_queries,
        "mrr": total_mrr / n_queries,
        f"ndcg@{top_k}": total_ndcg / n_queries,
        "n_queries": n_queries,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--eval-file", required=True, help="Path to evaluation JSONL file")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for retrieval")
    parser.add_argument("--collection", default="computer_network", help="Collection name")
    parser.add_argument("--settings", default="config/settings.storage_stack.yaml", help="Path to settings file")
    parser.add_argument("--task-intent", default="knowledge_query", help="Task intent for source-aware retrieval")
    parser.add_argument("--mode", choices=["sparse", "hybrid"], default="sparse", help="Retrieval mode for evaluation")
    parser.add_argument("--enable-rerank", action="store_true", help="Enable cross-encoder reranking during retrieval evaluation")
    parser.add_argument("--hybrid-init-timeout", type=float, default=10.0, help="Timeout in seconds for hybrid retrieval initialization before sparse fallback")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    run_evaluation(args.eval_file, args.top_k, args.collection, args.settings, args.task_intent, args.mode, args.enable_rerank, args.hybrid_init_timeout)


if __name__ == "__main__":
    main()
