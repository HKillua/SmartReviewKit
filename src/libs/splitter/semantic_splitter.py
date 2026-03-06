"""Embedding-based semantic text splitter.

Splits text by detecting semantic boundaries via cosine similarity of
consecutive paragraph embeddings.  Falls back to RecursiveSplitter when
the text has too few separable units.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

import numpy as np

from src.libs.splitter.base_splitter import BaseSplitter

logger = logging.getLogger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


_PARA_SEP = re.compile(r"\n{2,}")


class SemanticSplitter(BaseSplitter):
    """Split text at semantic boundaries detected by embedding similarity.

    Consecutive paragraphs are merged into a chunk as long as their
    similarity exceeds *similarity_threshold*.  When it drops, a new
    chunk is started.
    """

    def __init__(
        self,
        settings: Any,
        similarity_threshold: Optional[float] = None,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._settings = settings

        semantic_cfg = {}
        if hasattr(settings, "ingestion") and settings.ingestion:
            raw = getattr(settings.ingestion, "chunk_refiner", None)
            if isinstance(raw, dict):
                semantic_cfg = raw.get("semantic_splitter", {}) or {}

        self._threshold = similarity_threshold or semantic_cfg.get("similarity_threshold", 0.5)
        self._min_size = min_chunk_size or semantic_cfg.get("min_chunk_size", 100)
        self._max_size = max_chunk_size or semantic_cfg.get("max_chunk_size", 1500)
        self._embed_fn = kwargs.get("embed_fn")

    def set_embed_fn(self, fn):
        """Inject embedding function after construction (dependency injection)."""
        self._embed_fn = fn

    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        self.validate_text(text)

        embed_fn = kwargs.get("embed_fn", self._embed_fn)
        if embed_fn is None:
            embed_fn = self._get_default_embed_fn()

        paragraphs = [p.strip() for p in _PARA_SEP.split(text) if p.strip()]
        if len(paragraphs) <= 1:
            return [text]

        try:
            embeddings = embed_fn(paragraphs)
        except Exception as e:
            logger.warning(f"Embedding failed, falling back to single chunk: {e}")
            return [text]

        chunks: List[str] = []
        current_parts: List[str] = [paragraphs[0]]
        current_len = len(paragraphs[0])

        for i in range(1, len(paragraphs)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            para_len = len(paragraphs[i])

            if sim >= self._threshold and current_len + para_len <= self._max_size:
                current_parts.append(paragraphs[i])
                current_len += para_len
            else:
                chunk = "\n\n".join(current_parts)
                if len(chunk) >= self._min_size:
                    chunks.append(chunk)
                else:
                    current_parts.append(paragraphs[i])
                    current_len += para_len
                    continue
                current_parts = [paragraphs[i]]
                current_len = para_len

        if current_parts:
            last = "\n\n".join(current_parts)
            if chunks and len(last) < self._min_size:
                chunks[-1] += "\n\n" + last
            else:
                chunks.append(last)

        return chunks if chunks else [text]

    def _get_default_embed_fn(self):
        """Lazily build an embedding function from settings."""
        try:
            from src.libs.embedding.embedding_factory import EmbeddingFactory
            embedder = EmbeddingFactory.create(self._settings)

            def _embed_batch(texts: List[str]) -> List[List[float]]:
                return embedder.embed(texts)

            self._embed_fn = _embed_batch
            return _embed_batch
        except Exception as e:
            raise RuntimeError(
                f"No embed_fn provided and default embedding init failed: {e}"
            ) from e
