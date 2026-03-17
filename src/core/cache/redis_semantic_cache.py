"""Redis-backed semantic cache for multi-instance deployments."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional in dev
    redis = None
    REDIS_AVAILABLE = False


@dataclass
class RedisCacheEntry:
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


class RedisSemanticCache:
    """Redis-backed semantic cache with the same interface as SemanticCache."""

    def __init__(
        self,
        *,
        redis_url: str,
        embedding_fn: Callable,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,
        max_size: int = 500,
        namespace: str = "semantic_cache",
    ) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError("redis is required for RedisSemanticCache. Install with: pip install redis")
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._embed_fn = embedding_fn
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._namespace = namespace
        self._entries_key = f"{namespace}:entries"
        self._stats_key = f"{namespace}:stats"

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join((query or "").split()).casefold()

    @classmethod
    def _entry_key(cls, query: str, collection: str = "") -> str:
        return f"{collection}::{cls._normalize_query(query)}"

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        try:
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
            logger.warning("Redis semantic cache embedding failed", exc_info=True)
            return None

    async def get(self, query: str, collection: str = "") -> Optional[RedisCacheEntry]:
        def _get_exact() -> Optional[RedisCacheEntry]:
            now = time.time()
            exact_key = self._entry_key(query, collection)
            exact_raw = self._redis.hget(self._entries_key, exact_key)
            if exact_raw:
                payload = json.loads(exact_raw)
                if now - float(payload["created_at"]) <= self._ttl:
                    self._redis.hincrby(self._stats_key, "hits", 1)
                    return RedisCacheEntry(**payload)
                self._redis.hdel(self._entries_key, exact_key)
            return None

        exact_entry = await asyncio.to_thread(_get_exact)
        if exact_entry is not None:
            return exact_entry

        embedding = await self._get_embedding(query)
        if not embedding:
            return None

        def _get_semantic() -> Optional[RedisCacheEntry]:
            now = time.time()
            data = self._redis.hgetall(self._entries_key)
            best_entry: Optional[RedisCacheEntry] = None
            best_sim = 0.0
            expired: list[str] = []
            for key, raw in data.items():
                payload = json.loads(raw)
                if now - float(payload["created_at"]) > self._ttl:
                    expired.append(key)
                    continue
                if collection and payload.get("collection", "") != collection:
                    continue
                sim = _cosine_sim(embedding, payload["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_entry = RedisCacheEntry(**payload)
            if expired:
                self._redis.hdel(self._entries_key, *expired)
            if best_entry and best_sim >= self._threshold:
                self._redis.hincrby(self._stats_key, "hits", 1)
                return best_entry
            self._redis.hincrby(self._stats_key, "misses", 1)
            return None

        return await asyncio.to_thread(_get_semantic)

    async def put(
        self,
        query: str,
        result: str,
        metadata: Optional[Dict] = None,
        collection: str = "",
    ) -> None:
        embedding = await self._get_embedding(query)
        if not embedding:
            return

        effective_collection = collection or (metadata or {}).get("collection", "")

        entry = RedisCacheEntry(
            query=query,
            result=result,
            embedding=embedding,
            metadata=metadata or {},
            collection=effective_collection,
        )

        def _put() -> None:
            cache_key = self._entry_key(query, effective_collection)
            self._redis.hset(self._entries_key, cache_key, json.dumps(entry.__dict__, ensure_ascii=False))
            size = self._redis.hlen(self._entries_key)
            if size > self._max_size:
                items = self._redis.hkeys(self._entries_key)
                overflow = max(0, size - self._max_size)
                if overflow:
                    self._redis.hdel(self._entries_key, *items[:overflow])

        await asyncio.to_thread(_put)

    def invalidate_by_collection(self, collection: str) -> int:
        data = self._redis.hgetall(self._entries_key)
        to_remove: list[str] = []
        for key, raw in data.items():
            payload = json.loads(raw)
            if payload.get("collection") == collection:
                to_remove.append(key)
        if to_remove:
            self._redis.hdel(self._entries_key, *to_remove)
        return len(to_remove)

    def clear(self) -> None:
        self._redis.delete(self._entries_key)

    @property
    def stats(self) -> Dict[str, int]:
        raw = self._redis.hgetall(self._stats_key)
        return {
            "hits": int(raw.get("hits", 0)),
            "misses": int(raw.get("misses", 0)),
            "size": int(self._redis.hlen(self._entries_key)),
        }
