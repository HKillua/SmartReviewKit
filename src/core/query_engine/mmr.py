"""Maximal Marginal Relevance (MMR) for diversity-aware reranking.

MMR balances relevance and diversity by iteratively selecting the
candidate that is most relevant to the query *and* most different
from already-selected results.

    MMR = argmax_{d_i} [ λ · sim(d_i, q) - (1-λ) · max_{d_j ∈ S} sim(d_i, d_j) ]

Reference:
    Carbonell & Goldstein, 1998.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

import numpy as np

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)


def mmr_rerank(
    query_embedding: List[float],
    candidates: List[RetrievalResult],
    candidate_embeddings: List[List[float]],
    top_k: int = 10,
    lambda_param: float = 0.7,
) -> List[RetrievalResult]:
    """Apply MMR diversity reranking on *candidates*.

    Args:
        query_embedding: Embedding vector of the query.
        candidates: Candidate retrieval results.
        candidate_embeddings: One embedding per candidate (same order).
        top_k: Number of results to return.
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).

    Returns:
        Re-ordered list of top_k RetrievalResult objects.
    """
    if not candidates or not candidate_embeddings:
        return candidates[:top_k]

    n = len(candidates)
    q_vec = np.array(query_embedding, dtype=np.float32)
    c_mat = np.array(candidate_embeddings, dtype=np.float32)

    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm
    c_norms = np.linalg.norm(c_mat, axis=1, keepdims=True)
    c_norms[c_norms == 0] = 1.0
    c_mat = c_mat / c_norms

    rel_scores = c_mat @ q_vec  # (n,)

    selected_indices: List[int] = []
    remaining = set(range(n))

    for _ in range(min(top_k, n)):
        best_idx = -1
        best_score = -float("inf")

        for idx in remaining:
            relevance = float(rel_scores[idx])
            if selected_indices:
                sel_vecs = c_mat[selected_indices]
                max_sim = float(np.max(sel_vecs @ c_mat[idx]))
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx < 0:
            break
        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return [candidates[i] for i in selected_indices]
