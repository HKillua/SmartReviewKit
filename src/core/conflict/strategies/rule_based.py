"""Rule-based conflict detection — zero LLM calls, fastest strategy.

Detects:
  - Numerical inconsistencies (same concept, different numbers)
  - Definitional conflicts (same term, different definitions)
  - Source divergence (same topic from conflicting sources)
"""

from __future__ import annotations

import re
import logging
from typing import List

from src.core.conflict.strategies.base import ConflictStrategy
from src.core.conflict.types import Conflict, ConflictType
from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)

_NUM_PATTERN = re.compile(
    r"(?:为|是|=|等于|约|大约|最[大小]|默认|端口|超时|窗口|大小|长度|速率|带宽|延迟|TTL)\s*"
    r"[:：]?\s*(\d+(?:\.\d+)?)\s*(ms|s|秒|毫秒|字节|bytes?|KB|MB|GB|Mbps|Gbps|位|bit|个|次|%)?",
    re.I,
)

_DEF_PATTERN = re.compile(
    r"([\u4e00-\u9fff\w]{2,15})\s*(?:是指|定义为|即|是一种|称为|指的是|全称)\s*(.{5,80})",
)


class RuleBasedStrategy(ConflictStrategy):
    """Regex-based detection for numerical and definitional conflicts."""

    async def detect(self, query: str, results: List[RetrievalResult]) -> List[Conflict]:
        conflicts: list[Conflict] = []
        n = len(results)

        num_facts = [self._extract_numbers(r.text) for r in results]
        def_facts = [self._extract_definitions(r.text) for r in results]

        for i in range(n):
            for j in range(i + 1, n):
                conflicts.extend(self._compare_numbers(results[i], results[j], num_facts[i], num_facts[j]))
                conflicts.extend(self._compare_definitions(results[i], results[j], def_facts[i], def_facts[j]))

        return conflicts

    @staticmethod
    def _extract_numbers(text: str) -> dict[str, list[tuple[str, str]]]:
        """Extract (context_key, (value, unit)) pairs."""
        facts: dict[str, list[tuple[str, str]]] = {}
        for m in _NUM_PATTERN.finditer(text):
            start = max(0, m.start() - 20)
            ctx = text[start:m.start()].strip()
            ctx_key = re.sub(r"\s+", "", ctx)[-15:]
            val = m.group(1)
            unit = m.group(2) or ""
            facts.setdefault(ctx_key, []).append((val, unit))
        return facts

    @staticmethod
    def _extract_definitions(text: str) -> dict[str, str]:
        defs: dict[str, str] = {}
        for m in _DEF_PATTERN.finditer(text):
            term = m.group(1).strip()
            definition = m.group(2).strip()
            defs[term] = definition
        return defs

    @staticmethod
    def _compare_numbers(
        ra: RetrievalResult, rb: RetrievalResult,
        facts_a: dict, facts_b: dict,
    ) -> list[Conflict]:
        conflicts: list[Conflict] = []
        common_keys = set(facts_a.keys()) & set(facts_b.keys())
        for key in common_keys:
            vals_a = {v for v, _ in facts_a[key]}
            vals_b = {v for v, _ in facts_b[key]}
            if vals_a and vals_b and not vals_a & vals_b:
                conflicts.append(Conflict(
                    type=ConflictType.NUMERICAL,
                    chunk_a_id=ra.chunk_id,
                    chunk_b_id=rb.chunk_id,
                    claim_a=f"{key}: {', '.join(vals_a)}",
                    claim_b=f"{key}: {', '.join(vals_b)}",
                    confidence=0.7,
                    description=f"数值不一致: '{key}' 在两个片段中分别为 {vals_a} 和 {vals_b}",
                ))
        return conflicts

    @staticmethod
    def _compare_definitions(
        ra: RetrievalResult, rb: RetrievalResult,
        defs_a: dict, defs_b: dict,
    ) -> list[Conflict]:
        conflicts: list[Conflict] = []
        common_terms = set(defs_a.keys()) & set(defs_b.keys())
        for term in common_terms:
            da, db = defs_a[term], defs_b[term]
            if da != db and len(set(da.split()) & set(db.split())) < max(len(da.split()), len(db.split())) * 0.5:
                conflicts.append(Conflict(
                    type=ConflictType.DEFINITIONAL,
                    chunk_a_id=ra.chunk_id,
                    chunk_b_id=rb.chunk_id,
                    claim_a=f"{term}: {da[:80]}",
                    claim_b=f"{term}: {db[:80]}",
                    confidence=0.6,
                    description=f"定义冲突: '{term}' 在两个片段中有不同定义",
                ))
        return conflicts
