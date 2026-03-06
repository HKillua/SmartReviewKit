"""ConflictDetector — orchestrates multiple strategies and merges results."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from src.core.conflict.resolver import ConflictResolver
from src.core.conflict.strategies.base import ConflictStrategy
from src.core.conflict.types import Conflict, ConflictReport
from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.libs.embedding.base_embedding import BaseEmbedding
    from src.libs.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class ConflictDetector:
    """Runs all enabled strategies, deduplicates conflicts, and resolves.

    Usage::

        detector = ConflictDetector.from_config(settings, embedder, llm)
        report = await detector.detect(query, retrieval_results)
        if report.has_conflicts:
            ...
    """

    def __init__(
        self,
        strategies: List[ConflictStrategy] | None = None,
        resolver: ConflictResolver | None = None,
    ) -> None:
        self._strategies: list[ConflictStrategy] = strategies or []
        self._resolver = resolver or ConflictResolver()

    async def detect(self, query: str, results: List[RetrievalResult]) -> ConflictReport:
        if len(results) < 2:
            return ConflictReport()

        all_conflicts: list[Conflict] = []
        for strategy in self._strategies:
            try:
                found = await strategy.detect(query, results)
                all_conflicts.extend(found)
            except Exception:
                logger.warning("Strategy %s failed", type(strategy).__name__, exc_info=True)

        deduped = self._deduplicate(all_conflicts)
        report = self._resolver.resolve(deduped, results)
        if deduped:
            logger.info("Conflict detection: %d conflicts found", len(deduped))
        return report

    @staticmethod
    def _deduplicate(conflicts: list[Conflict]) -> list[Conflict]:
        """Remove duplicate conflicts between the same pair of chunks."""
        seen: set[tuple[str, str]] = set()
        out: list[Conflict] = []
        for c in conflicts:
            pair = tuple(sorted([c.chunk_a_id, c.chunk_b_id]))
            key = (pair[0], pair[1])
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    @classmethod
    def from_config(
        cls,
        embedder: Optional[BaseEmbedding] = None,
        llm: Optional[BaseLLM] = None,
        *,
        enable_rule: bool = True,
        enable_embedding: bool = True,
        enable_nli: bool = False,
        nli_max_pairs: int = 5,
        sim_threshold: float = 0.90,
    ) -> ConflictDetector:
        strategies: list[ConflictStrategy] = []

        if enable_rule:
            from src.core.conflict.strategies.rule_based import RuleBasedStrategy
            strategies.append(RuleBasedStrategy())

        if enable_embedding:
            from src.core.conflict.strategies.embedding_sim import EmbeddingSimStrategy
            strategies.append(EmbeddingSimStrategy(
                embedder=embedder,
                sim_threshold=sim_threshold,
            ))

        if enable_nli and llm is not None:
            from src.core.conflict.strategies.llm_nli import LLMNliStrategy
            strategies.append(LLMNliStrategy(llm=llm, max_pairs=nli_max_pairs))

        return cls(strategies=strategies)
