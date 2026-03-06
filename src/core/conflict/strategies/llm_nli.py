"""LLM-based Natural Language Inference conflict detection.

Sends pairs of chunks to an LLM to determine if they contain contradictory
claims. Highest accuracy but most expensive strategy — used sparingly after
cheaper strategies have pre-filtered candidates.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, List, Optional

from src.core.conflict.strategies.base import ConflictStrategy
from src.core.conflict.types import Conflict, ConflictType
from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.libs.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)

_NLI_PROMPT = """\
你是一个事实一致性检查器。给你两段知识片段和一个查询，请判断两段内容是否存在矛盾。

查询: {query}

片段 A:
{text_a}

片段 B:
{text_b}

请用以下 JSON 格式回答 (不要额外文字):
{{"verdict": "contradiction" | "neutral" | "entailment", "confidence": 0.0~1.0, "reason": "简述矛盾点"}}
"""


class LLMNliStrategy(ConflictStrategy):
    """Use LLM to judge contradiction between chunk pairs.

    Only compares top-k pairs to control cost.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        max_pairs: int = 5,
    ) -> None:
        self._llm = llm
        self._max_pairs = max_pairs

    async def detect(self, query: str, results: List[RetrievalResult]) -> List[Conflict]:
        if self._llm is None:
            return []

        pairs = self._select_pairs(results)
        conflicts: list[Conflict] = []

        for ra, rb in pairs:
            conflict = await self._judge_pair(query, ra, rb)
            if conflict is not None:
                conflicts.append(conflict)

        return conflicts

    def _select_pairs(self, results: List[RetrievalResult]) -> list[tuple[RetrievalResult, RetrievalResult]]:
        """Select up to max_pairs pairs prioritising adjacent-ranked chunks."""
        n = len(results)
        pairs: list[tuple[RetrievalResult, RetrievalResult]] = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((results[i], results[j]))
                if len(pairs) >= self._max_pairs:
                    return pairs
        return pairs

    async def _judge_pair(self, query: str, ra: RetrievalResult, rb: RetrievalResult) -> Optional[Conflict]:
        prompt = _NLI_PROMPT.format(
            query=query,
            text_a=ra.text[:500],
            text_b=rb.text[:500],
        )

        try:
            raw = await asyncio.to_thread(
                self._llm.generate, prompt, temperature=0.0, max_tokens=300,
            )
        except Exception:
            logger.warning("LLM NLI call failed for pair (%s, %s)", ra.chunk_id, rb.chunk_id, exc_info=True)
            return None

        from src.agent.utils.json_helpers import safe_parse_json
        parsed = safe_parse_json(raw, fallback={})
        verdict = parsed.get("verdict", "neutral")
        confidence = float(parsed.get("confidence", 0.5))
        reason = parsed.get("reason", "")

        if verdict == "contradiction" and confidence >= 0.5:
            return Conflict(
                type=ConflictType.FACTUAL,
                chunk_a_id=ra.chunk_id,
                chunk_b_id=rb.chunk_id,
                claim_a=ra.text[:100],
                claim_b=rb.text[:100],
                confidence=min(confidence, 1.0),
                description=reason or "LLM 判定存在矛盾",
            )
        return None
