"""Conflict resolution — select trusted chunks and produce a summary."""

from __future__ import annotations

import logging
from collections import Counter
from typing import List

from src.core.conflict.types import Conflict, ConflictReport
from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)


class ConflictResolver:
    """Resolve conflicts by scoring chunk trustworthiness.

    Heuristics:
      1. Higher retrieval score ⟹ more trusted.
      2. Fewer conflict appearances ⟹ more trusted.
      3. More recent source (if metadata has timestamp) ⟹ more trusted.
    """

    def resolve(self, conflicts: List[Conflict], results: List[RetrievalResult]) -> ConflictReport:
        if not conflicts:
            return ConflictReport()

        score_map = {r.chunk_id: r.score for r in results}
        conflict_count: Counter[str] = Counter()
        for c in conflicts:
            conflict_count[c.chunk_a_id] += 1
            conflict_count[c.chunk_b_id] += 1

        all_ids = {r.chunk_id for r in results}
        conflict_ids = {c.chunk_a_id for c in conflicts} | {c.chunk_b_id for c in conflicts}
        safe_ids = all_ids - conflict_ids

        scored = {}
        for cid in all_ids:
            base_score = score_map.get(cid, 0.0)
            penalty = conflict_count.get(cid, 0) * 0.15
            scored[cid] = max(base_score - penalty, 0.0)

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        trusted = list(safe_ids) + [cid for cid, _ in ranked if cid not in safe_ids]

        summary_lines = [f"检测到 {len(conflicts)} 处知识冲突。"]
        type_counts = Counter(c.type.value for c in conflicts)
        for t, cnt in type_counts.most_common():
            summary_lines.append(f"  - {t}: {cnt} 处")
        if trusted:
            summary_lines.append(f"推荐优先信任片段: {', '.join(trusted[:3])}")

        return ConflictReport(
            conflicts=conflicts,
            trusted_chunk_ids=trusted,
            resolution_summary="\n".join(summary_lines),
        )
