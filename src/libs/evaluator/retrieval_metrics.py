"""Information-Retrieval evaluation metrics: Hit Rate, MRR, NDCG.

These metrics operate on **ranked lists** of retrieved chunk IDs
compared against a ground-truth set of relevant IDs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator


def hit_rate(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Binary: 1.0 if any relevant doc appears in retrieved list, else 0.0."""
    return 1.0 if set(retrieved_ids) & set(relevant_ids) else 0.0


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant hit (0 if none)."""
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg(retrieved_ids: List[str], relevant_ids: List[str], k: int = 0) -> float:
    """Normalised Discounted Cumulative Gain @ k.

    Binary relevance (1 if in ground-truth, else 0).
    """
    if not relevant_ids:
        return 0.0
    if k <= 0:
        k = len(retrieved_ids)

    relevant_set = set(relevant_ids)
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k]):
        if cid in relevant_set:
            dcg += 1.0 / math.log2(i + 2)

    ideal = sorted([1] * min(len(relevant_set), k) + [0] * max(k - len(relevant_set), 0), reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal) if rel > 0)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 0) -> float:
    """Precision@k — fraction of top-k that are relevant."""
    if k <= 0:
        k = len(retrieved_ids)
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & set(relevant_ids)) / len(top_k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 0) -> float:
    """Recall@k — fraction of relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    if k <= 0:
        k = len(retrieved_ids)
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(relevant_ids)) / len(relevant_ids)


class RetrievalMetricsEvaluator(BaseEvaluator):
    """Evaluator that computes IR metrics given retrieved IDs vs ground-truth.

    Metrics produced: hit_rate, mrr, ndcg@k, precision@k, recall@k.
    """

    def __init__(self, k: int = 10, **kwargs: Any) -> None:
        self.k = k

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.validate_query(query)

        if ground_truth is None or not isinstance(ground_truth, dict):
            return {}
        relevant_ids: List[str] = ground_truth.get("ids", [])
        if not relevant_ids:
            return {}

        retrieved_ids: List[str] = []
        for c in retrieved_chunks:
            if isinstance(c, str):
                retrieved_ids.append(c)
            elif isinstance(c, dict):
                retrieved_ids.append(c.get("chunk_id", c.get("id", str(c))))
            elif hasattr(c, "chunk_id"):
                retrieved_ids.append(str(c.chunk_id))
            else:
                retrieved_ids.append(str(c))

        return {
            "hit_rate": hit_rate(retrieved_ids, relevant_ids),
            "mrr": mrr(retrieved_ids, relevant_ids),
            f"ndcg@{self.k}": ndcg(retrieved_ids, relevant_ids, self.k),
            f"precision@{self.k}": precision_at_k(retrieved_ids, relevant_ids, self.k),
            f"recall@{self.k}": recall_at_k(retrieved_ids, relevant_ids, self.k),
        }
