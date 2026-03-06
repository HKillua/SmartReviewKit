#!/usr/bin/env python3
"""Automated RAG evaluation script.

Usage:
    python scripts/run_eval.py [--test-set PATH] [--top-k 10] [--collection NAME]

Runs HybridSearch on every query in the golden test set, computes
IR metrics (Hit Rate, MRR, NDCG@k, Precision@k, Recall@k), and
prints + saves the evaluation report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import Settings
from src.libs.evaluator.retrieval_metrics import RetrievalMetricsEvaluator
from src.observability.evaluation.eval_runner import EvalRunner, load_test_set


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Evaluation Runner")
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set.json",
        help="Path to golden test set JSON",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--output", default="data/eval_report.json")
    args = parser.parse_args()

    print(f"Loading settings ...")
    settings = Settings.load("config/settings.yaml")

    print(f"Initialising HybridSearch ...")
    from src.core.query_engine.hybrid_search import create_hybrid_search
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.dense_retriever import DenseRetriever
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    embedding_client = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings)
    dense_retriever = DenseRetriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
    sparse_retriever = create_sparse_retriever(settings)
    query_processor = QueryProcessor()
    hybrid_search = create_hybrid_search(
        settings=settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )

    evaluator = RetrievalMetricsEvaluator(k=args.top_k)

    runner = EvalRunner(
        settings=settings,
        hybrid_search=hybrid_search,
        evaluator=evaluator,
    )

    print(f"Running evaluation on {args.test_set} ...")
    report = runner.run(
        test_set_path=args.test_set,
        top_k=args.top_k,
        collection=args.collection,
    )

    report_dict = report.to_dict()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation Report  ({report_dict['query_count']} queries)")
    print(f"{'='*60}")
    for k, v in report_dict["aggregate_metrics"].items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"  {'elapsed_ms':20s}: {report_dict['total_elapsed_ms']:.1f}")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
