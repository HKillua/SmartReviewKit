"""Semantic cache — caches retrieval results keyed by query embedding similarity.

When a new query arrives, its embedding is compared against cached query
embeddings via cosine similarity.  If the best match exceeds a configurable
threshold **and** has not expired (TTL), the cached result is returned
immediately, avoiding a full HybridSearch + LLM call.

Cache invalidation is supported per-collection so that newly ingested
documents immediately take effect.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cached result."""

    query: str
    result: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    collection: str = ""


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


class SemanticCache:
    """LRU semantic cache with TTL and per-collection invalidation.

    Parameters:
        embedding_fn:  ``async (text) -> list[float]`` or sync callable.
        similarity_threshold: Minimum cosine similarity for a cache hit (default 0.92).
        ttl_seconds: Time-to-live for cache entries in seconds (default 3600).
        max_size: Maximum number of entries (LRU eviction, default 500).
    """

    def __init__(
        self,
        embedding_fn: Callable,
        *,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,
        max_size: int = 500,
    ) -> None:
        self._embed_fn = embedding_fn
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0}

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join((query or "").split()).casefold()

    @classmethod
    def _entry_key(cls, query: str, collection: str = "") -> str:
        return f"{collection}::{cls._normalize_query(query)}"

    async def get(self, query: str, collection: str = "") -> Optional[CacheEntry]:
        """Look up a semantically similar cached result."""
        now = time.time()
        entry_key = self._entry_key(query, collection)

        with self._lock:
            exact_entry = self._entries.get(entry_key)
            if exact_entry is not None:
                if now - exact_entry.created_at <= self._ttl:
                    self._entries.move_to_end(entry_key)
                    self._stats["hits"] += 1
                    logger.debug("Semantic cache EXACT HIT: %s", query[:60])
                    return exact_entry
                self._entries.pop(entry_key, None)

        embedding = await self._get_embedding(query)
        if not embedding:
            return None

        best_entry: Optional[CacheEntry] = None
        best_sim = 0.0
        best_cache_key = ""

        with self._lock:
            expired_keys: list[str] = []
            for key, entry in self._entries.items():
                if now - entry.created_at > self._ttl:
                    expired_keys.append(key)
                    continue
                if collection and entry.collection != collection:
                    continue
                sim = _cosine_sim(embedding, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry
                    best_cache_key = key

            for k in expired_keys:
                self._entries.pop(k, None)

            if best_entry is not None and best_sim >= self._threshold:
                self._entries.move_to_end(best_cache_key)
                self._stats["hits"] += 1
                logger.debug("Semantic cache HIT (sim=%.3f): %s", best_sim, query[:60])
                return best_entry

        self._stats["misses"] += 1
        return None

    async def put(
        self,
        query: str,
        result: str,
        metadata: Optional[Dict] = None,
        collection: str = "",
    ) -> None:
        """Store a query-result pair in the cache."""
        embedding = await self._get_embedding(query)
        if not embedding:
            return

        effective_collection = collection or (metadata.get("collection", "") if metadata else "")

        entry = CacheEntry(
            query=query,
            result=result,
            embedding=embedding,
            metadata=metadata or {},
            collection=effective_collection,
        )
        with self._lock:
            entry_key = self._entry_key(query, effective_collection)
            self._entries[entry_key] = entry
            self._entries.move_to_end(entry_key)
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)

    def invalidate_by_collection(self, collection: str) -> int:
        """Remove all cache entries belonging to *collection*."""
        with self._lock:
            to_remove = [
                k for k, v in self._entries.items() if v.collection == collection
            ]
            for k in to_remove:
                self._entries.pop(k, None)
        if to_remove:
            logger.info("Semantic cache: invalidated %d entries for collection '%s'", len(to_remove), collection)
        return len(to_remove)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @property
    def stats(self) -> Dict[str, int]:
        return {**self._stats, "size": len(self._entries)}

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            import asyncio
            if not text or not text.strip():
                return None
            used_list_wrapper = False
            try:
                if asyncio.iscoroutinefunction(self._embed_fn):
                    result = await self._embed_fn([text])
                else:
                    result = await asyncio.to_thread(self._embed_fn, [text])
                used_list_wrapper = True
            except TypeError:
                if asyncio.iscoroutinefunction(self._embed_fn):
                    result = await self._embed_fn(text)
                else:
                    result = await asyncio.to_thread(self._embed_fn, text)
            if (
                used_list_wrapper
                and isinstance(result, list)
                and result
                and not isinstance(result[0], list)
            ):
                if asyncio.iscoroutinefunction(self._embed_fn):
                    result = await self._embed_fn(text)
                else:
                    result = await asyncio.to_thread(self._embed_fn, text)
            if isinstance(result, list) and result and isinstance(result[0], list):
                return result[0]
            return result
        except Exception:
            logger.warning("Semantic cache embedding failed", exc_info=True)
            return None
