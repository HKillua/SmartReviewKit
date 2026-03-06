"""Embedding-based conflict detection.

When two chunks have high semantic similarity but divergent factual claims,
flag them as potential conflicts. Uses the embeddings already attached to
RetrievalResult (no extra API calls when embeddings are present).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional

from src.core.conflict.strategies.base import ConflictStrategy
from src.core.conflict.types import Conflict, ConflictType
from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.libs.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger(__name__)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _jaccard_tokens(text_a: str, text_b: str) -> float:
    sa = set(text_a.split())
    sb = set(text_b.split())
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


class EmbeddingSimStrategy(ConflictStrategy):
    """Flag chunk pairs where semantic similarity is high but token overlap is low.

    This heuristic catches paraphrased contradictions that rule-based detection misses.
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedding] = None,
        sim_threshold: float = 0.90,
        jaccard_ceiling: float = 0.35,
    ) -> None:
        self._embedder = embedder
        self._sim_threshold = sim_threshold
        self._jaccard_ceiling = jaccard_ceiling

    async def detect(self, query: str, results: List[RetrievalResult]) -> List[Conflict]:
        n = len(results)
        if n < 2:
            return []

        embeddings = await self._get_embeddings(results)
        if embeddings is None:
            return []

        conflicts: list[Conflict] = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = _cosine_sim(embeddings[i], embeddings[j])
                if sim < self._sim_threshold:
                    continue
                jac = _jaccard_tokens(results[i].text, results[j].text)
                if jac > self._jaccard_ceiling:
                    continue
                conflicts.append(Conflict(
                    type=ConflictType.FACTUAL,
                    chunk_a_id=results[i].chunk_id,
                    chunk_b_id=results[j].chunk_id,
                    claim_a=results[i].text[:100],
                    claim_b=results[j].text[:100],
                    confidence=round(sim * (1 - jac), 2),
                    description=(
                        f"高语义相似 (cos={sim:.2f}) 但低词汇重叠 (jaccard={jac:.2f})，"
                        "可能存在事实分歧"
                    ),
                ))
        return conflicts

    async def _get_embeddings(self, results: List[RetrievalResult]) -> Optional[List[List[float]]]:
        pre = [r.embedding for r in results]
        if all(e is not None for e in pre):
            return pre  # type: ignore[return-value]

        if self._embedder is None:
            logger.debug("No embedder and missing pre-computed embeddings; skipping")
            return None

        import asyncio
        texts = [r.text for r in results]
        vecs = await asyncio.to_thread(self._embedder.embed, texts)
        return vecs
