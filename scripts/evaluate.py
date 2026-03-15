#!/usr/bin/env python
"""Evaluation script for Modular RAG MCP Server.

Runs batch evaluation against a golden test set and outputs a metrics report.

Usage:
    # Run with default settings (custom evaluator)
    python scripts/evaluate.py

    # Specify a custom golden test set
    python scripts/evaluate.py --test-set path/to/golden.json

    # Use a specific collection
    python scripts/evaluate.py --collection technical_docs

    # JSON output
    python scripts/evaluate.py --json

Exit codes:
    0 - Success
    1 - Evaluation failure
    2 - Configuration error
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class _EvalSearchAdapter:
    """Expose a HybridSearch-like surface over SourceAwareSearch."""

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation against a golden test set."
    )
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set.json",
        help="Path to golden test set JSON file (default: tests/fixtures/golden_test_set.json)",
    )
    parser.add_argument(
        "--settings",
        default="config/settings.storage_stack.yaml",
        help="Path to settings file (default: config/settings.storage_stack.yaml)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name to search within.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of formatted text.",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Skip retrieval (evaluate with mock chunks for testing).",
    )
    parser.add_argument(
        "--task-intent",
        default="knowledge_query",
        help="Task intent used by source-aware retrieval (default: knowledge_query).",
    )
    parser.add_argument(
        "--mode",
        choices=["sparse", "hybrid"],
        default="sparse",
        help="Retrieval mode used for evaluation (default: sparse).",
    )
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Enable cross-encoder reranking during evaluation.",
    )
    return parser.parse_args()


def _prepare_eval_settings(settings, test_cases):
    """Enable evaluation for the CLI without mutating runtime app defaults."""
    metrics: list[str] = []
    if any(tc.expected_chunk_ids for tc in test_cases):
        metrics.extend(["hit_rate", "mrr"])
    if any(tc.expected_sources for tc in test_cases):
        metrics.extend(["source_hit_rate", "source_mrr"])
    if not metrics:
        metrics = ["source_hit_rate", "source_mrr"]

    evaluation_cfg = replace(
        settings.evaluation,
        enabled=True,
        provider="custom",
        metrics=metrics,
    )
    retrieval_cfg = replace(
        settings.retrieval,
        query_rewrite_enabled=False,
        hyde_enabled=False,
        multi_query_enabled=False,
    )
    return replace(settings, evaluation=evaluation_cfg, retrieval=retrieval_cfg)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        from src.core.settings import load_settings
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory
        from src.observability.evaluation.eval_runner import EvalRunner, load_test_set

        test_cases = load_test_set(args.test_set)
        settings = _prepare_eval_settings(load_settings(args.settings), test_cases)
    except Exception as exc:
        print(f"❌ Configuration error: {exc}", file=sys.stderr)
        return 2

    # Create evaluator from config
    try:
        evaluator = EvaluatorFactory.create(settings)
        evaluator_name = type(evaluator).__name__
    except Exception as exc:
        print(f"❌ Failed to create evaluator: {exc}", file=sys.stderr)
        return 2

    # Create HybridSearch (unless --no-search)
    hybrid_search = None
    if not args.no_search:
        try:
            collection = args.collection or "default"
            if args.mode == "sparse":
                from src.observability.evaluation.source_eval_search import SparseSourceEvalSearch

                hybrid_search = SparseSourceEvalSearch(settings=settings, collection=collection)
                print(f"✅ SparseSourceEvalSearch initialized for collection: {collection}")
            else:
                from src.core.query_engine.query_processor import QueryProcessor
                from src.core.query_engine.hybrid_search import create_hybrid_search
                from src.core.query_engine.dense_retriever import create_dense_retriever
                from src.core.query_engine.sparse_retriever import create_sparse_retriever
                from src.core.query_engine.query_router import QueryRouter
                from src.core.query_engine.source_aware_search import SourceAwareSearch
                from src.libs.embedding.embedding_factory import EmbeddingFactory
                from src.libs.embedding.cached_embedding import CachedEmbedding
                from src.libs.vector_store.vector_store_factory import VectorStoreFactory
                from src.storage.runtime import create_sparse_index

                vector_store = VectorStoreFactory.create(
                    settings, collection_name=collection,
                )
                raw_embedding_client = EmbeddingFactory.create(settings)
                embedding_client = CachedEmbedding(
                    raw_embedding_client,
                    max_size=settings.retrieval.embedding_cache_size,
                )
                dense_retriever = create_dense_retriever(
                    settings=settings,
                    embedding_client=embedding_client,
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
                base_hybrid_search = create_hybrid_search(
                    settings=settings,
                    query_processor=query_processor,
                    dense_retriever=dense_retriever,
                    sparse_retriever=sparse_retriever,
                )
                if args.enable_rerank:
                    from src.libs.reranker.reranker_factory import RerankerFactory

                    try:
                        base_hybrid_search.reranker = RerankerFactory.create(settings)
                    except Exception as exc:
                        print(f"⚠️  Reranker unavailable, continuing without rerank: {exc}")

                query_router = QueryRouter(fallback_to_llm=True)
                source_aware_search = SourceAwareSearch(
                    hybrid_search=base_hybrid_search,
                    query_router=query_router,
                )
                hybrid_search = _EvalSearchAdapter(source_aware_search, args.task_intent)
                print(f"✅ SourceAwareSearch initialized for collection: {collection} (task_intent={args.task_intent})")
        except Exception as exc:
            print(f"⚠️  Failed to initialize search (running without retrieval): {exc}")

    # Create and run EvalRunner
    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
    )

    try:
        print(f"\n🔍 Running evaluation with {evaluator_name}...")
        print(f"📄 Test set: {args.test_set}")
        print(f"🔢 Top-K: {args.top_k}\n")

        report = runner.run(
            test_set_path=args.test_set,
            top_k=args.top_k,
            collection=args.collection,
        )
    except Exception as exc:
        print(f"❌ Evaluation failed: {exc}", file=sys.stderr)
        return 1

    # Output results
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_report(report)

    return 0


def _print_report(report) -> None:
    """Print formatted evaluation report."""
    print("=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(f"  Evaluator: {report.evaluator_name}")
    print(f"  Test Set:  {report.test_set_path}")
    print(f"  Queries:   {len(report.query_results)}")
    print(f"  Time:      {report.total_elapsed_ms:.0f} ms")
    print()

    # Aggregate metrics
    print("─" * 60)
    print("  AGGREGATE METRICS")
    print("─" * 60)
    if report.aggregate_metrics:
        for metric, value in sorted(report.aggregate_metrics.items()):
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            print(f"  {metric:<25s} {bar} {value:.4f}")
    else:
        print("  (no metrics computed)")
    print()

    # Per-query details
    print("─" * 60)
    print("  PER-QUERY RESULTS")
    print("─" * 60)
    for i, qr in enumerate(report.query_results, 1):
        print(f"\n  [{i}] {qr.query}")
        print(f"      Retrieved: {len(qr.retrieved_chunk_ids)} chunks")
        if qr.metrics:
            for metric, value in sorted(qr.metrics.items()):
                print(f"      {metric}: {value:.4f}")
        else:
            print("      (no metrics)")
        print(f"      Time: {qr.elapsed_ms:.0f} ms")

    print()
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
