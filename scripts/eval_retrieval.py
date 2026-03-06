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
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in expected:
            dcg += 1.0 / math.log2(i + 2)
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


def run_evaluation(eval_file: str, top_k: int = 5, collection: str = "computer_network"):
    from src.core.settings import load_settings, resolve_path
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.core.query_engine.fusion import RRFFusion
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    settings = load_settings()
    embedding = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    dense = create_dense_retriever(settings=settings, embedding_client=embedding, vector_store=vector_store)
    bm25 = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
    sparse = create_sparse_retriever(settings=settings, bm25_indexer=bm25, vector_store=vector_store)
    sparse.default_collection = collection
    fusion = RRFFusion(k=60)
    hybrid = HybridSearch(
        settings=settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=fusion,
    )
    hybrid.embedding_client = embedding

    eval_items = load_eval_set(eval_file)
    print(f"\n{'='*60}")
    print(f"  Retrieval Quality Evaluation")
    print(f"  Dataset: {eval_file} ({len(eval_items)} queries)")
    print(f"  Collection: {collection}  |  top_k: {top_k}")
    print(f"{'='*60}\n")

    total_hr, total_mrr, total_ndcg = 0.0, 0.0, 0.0

    for i, item in enumerate(eval_items, 1):
        query = item["query"]
        expected = item["expected_ids"]
        results = hybrid.search(query=query, top_k=top_k)
        retrieved = [r.chunk_id for r in results]

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
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    run_evaluation(args.eval_file, args.top_k, args.collection)


if __name__ == "__main__":
    main()
