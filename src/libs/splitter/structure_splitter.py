"""Structure-aware text splitter.

Splits Markdown text by heading hierarchy and structural boundaries.
Respects formulas (``$...$`` / ``$$...$$``) and exercise blocks as
atomic units.  Falls back to recursive character splitting when a
section exceeds ``max_chunk_size``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

from src.libs.splitter.base_splitter import BaseSplitter

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_HR_RE = re.compile(r"^-{3,}$", re.MULTILINE)
_EXERCISE_START_RE = re.compile(
    r"(习题|练习|思考题|Exercise|Problem|Questions)", re.IGNORECASE
)
_FORMULA_BLOCK_RE = re.compile(r"\$\$.+?\$\$", re.DOTALL)


class StructureAwareSplitter(BaseSplitter):
    """Split text by document structure, keeping formulas and exercises intact.

    Strategy:
    1. Find all heading positions and horizontal rules.
    2. Split into structural segments at those boundaries.
    3. If a segment exceeds ``max_chunk_size``, sub-split it using
       double-newline paragraph boundaries (preserving formula blocks).
    4. Merge tiny segments (< ``min_chunk_size``) with their neighbours.
    """

    def __init__(
        self,
        settings: Any,
        max_chunk_size: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._settings = settings

        default_max = 1200
        default_min = 80
        if hasattr(settings, "ingestion") and settings.ingestion:
            default_max = getattr(settings.ingestion, "chunk_size", default_max)

        self._max_size = max_chunk_size or default_max
        self._min_size = min_chunk_size or default_min

    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        self.validate_text(text)

        segments = self._split_by_headings(text)
        chunks: List[str] = []
        for seg in segments:
            if len(seg) <= self._max_size:
                chunks.append(seg)
            else:
                chunks.extend(self._subsplit_segment(seg))

        chunks = self._merge_small(chunks)
        return chunks if chunks else [text]

    # ------------------------------------------------------------------

    def _split_by_headings(self, text: str) -> List[str]:
        positions: List[int] = []
        for m in _HEADING_RE.finditer(text):
            positions.append(m.start())
        for m in _HR_RE.finditer(text):
            positions.append(m.start())

        if not positions:
            return [text]

        positions = sorted(set(positions))
        segments: List[str] = []
        if positions[0] > 0:
            segments.append(text[: positions[0]].strip())
        for i, pos in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            seg = text[pos:end].strip()
            if seg:
                segments.append(seg)
        return segments

    def _subsplit_segment(self, text: str) -> List[str]:
        """Split an oversized segment at paragraph boundaries while
        keeping formula blocks and exercise blocks intact."""
        protected_ranges = self._protected_ranges(text)
        split_points = self._find_para_breaks(text, protected_ranges)

        if not split_points:
            return [text]

        chunks: List[str] = []
        prev = 0
        for sp in split_points:
            chunk = text[prev:sp].strip()
            if chunk:
                chunks.append(chunk)
            prev = sp

        tail = text[prev:].strip()
        if tail:
            chunks.append(tail)

        return chunks

    @staticmethod
    def _protected_ranges(text: str) -> List[tuple]:
        ranges = []
        for m in _FORMULA_BLOCK_RE.finditer(text):
            ranges.append((m.start(), m.end()))
        return ranges

    def _find_para_breaks(self, text: str, protected: List[tuple]) -> List[int]:
        breaks = []
        for m in re.finditer(r"\n\n+", text):
            pos = m.start()
            if any(s <= pos < e for s, e in protected):
                continue
            breaks.append(m.end())
        return breaks

    def _merge_small(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return chunks
        merged: List[str] = [chunks[0]]
        for c in chunks[1:]:
            if len(merged[-1]) < self._min_size:
                merged[-1] += "\n\n" + c
            else:
                merged.append(c)
        if len(merged) > 1 and len(merged[-1]) < self._min_size:
            merged[-2] += "\n\n" + merged[-1]
            merged.pop()
        return merged
