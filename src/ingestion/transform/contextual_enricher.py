"""Contextual Retrieval — inject document-level context into each chunk.

Inspired by Anthropic's "Contextual Retrieval" technique:
for every chunk we prepend a short context string that situates the
chunk within the full document, drastically improving retrieval quality
for ambiguous fragments.

Modes:
1. ``rule``  — first heading / filename / section title (fast, no LLM).
2. ``llm``   — use LLM to generate a 1-2 sentence situating summary.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

from src.core.types import Chunk

logger = logging.getLogger(__name__)


_CONTEXTUAL_PROMPT = (
    "请用1-2句话简要说明以下片段在整篇文档中的位置和上下文。\n"
    "文档标题：{doc_title}\n"
    "片段内容：\n{chunk_text}\n\n"
    "上下文说明："
)


class ContextualEnricher:
    """Prepend contextual prefix to each chunk for better retrieval."""

    def __init__(
        self,
        mode: str = "rule",
        llm_service: Any = None,
    ) -> None:
        self._mode = mode
        self._llm = llm_service

    def enrich(self, chunks: List[Chunk], doc_title: str = "") -> List[Chunk]:
        """Add contextual prefix to every chunk (sync / rule-based)."""
        if self._mode == "llm" and self._llm is not None:
            logger.info("Contextual enrichment via LLM for %d chunks", len(chunks))
            return self._enrich_llm_sync(chunks, doc_title)
        return self._enrich_rule(chunks, doc_title)

    def _enrich_rule(self, chunks: List[Chunk], doc_title: str) -> List[Chunk]:
        for chunk in chunks:
            section = chunk.metadata.get("section_title", "")
            source = chunk.metadata.get("source_path", "")
            parts = [p for p in [doc_title, section, source] if p]
            if parts:
                prefix = f"[上下文：{'> '.join(parts)}]\n"
                chunk.text = prefix + chunk.text
                chunk.metadata["contextual_prefix"] = prefix.strip()
        return chunks

    def _enrich_llm_sync(self, chunks: List[Chunk], doc_title: str) -> List[Chunk]:
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._enrich_llm(chunks, doc_title))
        finally:
            loop.close()

    async def _enrich_llm(self, chunks: List[Chunk], doc_title: str) -> List[Chunk]:
        for chunk in chunks:
            try:
                prompt = _CONTEXTUAL_PROMPT.format(
                    doc_title=doc_title or "未知",
                    chunk_text=chunk.text[:500],
                )
                messages = [{"role": "user", "content": prompt}]
                response = await self._llm.chat(messages)
                prefix = response.content.strip()
                if prefix:
                    chunk.text = f"[上下文：{prefix}]\n{chunk.text}"
                    chunk.metadata["contextual_prefix"] = prefix
            except Exception as exc:
                logger.warning("Contextual LLM enrichment failed: %s", exc)
        return chunks
