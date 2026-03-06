"""LRU-cached wrapper around any BaseEmbedding implementation.

Wraps an existing embedder and caches results keyed by text hash.
Identical texts (e.g. during re-ingestion or repeated queries) hit
the cache, avoiding redundant API calls.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger(__name__)


class CachedEmbedding(BaseEmbedding):
    """Transparent LRU cache for an underlying BaseEmbedding.

    Args:
        delegate: The actual embedding provider.
        max_size: Maximum number of cached vectors (default 4096).
    """

    def __init__(self, delegate: BaseEmbedding, max_size: int = 4096) -> None:
        self._delegate = delegate
        self._max_size = max_size
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._total_api_calls = 0
        self._total_texts_embedded = 0

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        self.validate_texts(texts)

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        with self._lock:
            for i, text in enumerate(texts):
                key = self._hash(text)
                if key in self._cache:
                    self._cache.move_to_end(key)
                    results[i] = self._cache[key]
                    self._hits += 1
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
                    self._misses += 1

        if uncached_texts:
            self._total_api_calls += 1
            self._total_texts_embedded += len(uncached_texts)
            new_vectors = self._delegate.embed(uncached_texts, trace=trace, **kwargs)
            with self._lock:
                for idx, vec in zip(uncached_indices, new_vectors):
                    key = self._hash(texts[idx])
                    self._cache[key] = vec
                    results[idx] = vec
                    if len(self._cache) > self._max_size:
                        self._cache.popitem(last=False)

        if (self._hits + self._misses) % 200 == 0 and self._hits > 0:
            total = self._hits + self._misses
            logger.debug(
                "Embedding cache: %d hits / %d total (%.1f%%), size=%d",
                self._hits, total, 100.0 * self._hits / total, len(self._cache),
            )

        return results  # type: ignore[return-value]

    def get_dimension(self) -> int:
        return self._delegate.get_dimension()

    @property
    def cache_stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            "size": len(self._cache),
            "max_size": self._max_size,
            "api_calls": self._total_api_calls,
            "texts_embedded": self._total_texts_embedded,
        }
