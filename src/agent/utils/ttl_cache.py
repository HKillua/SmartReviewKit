"""TTL-bounded dictionary for caches that should not grow unbounded."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import TypeVar

V = TypeVar("V")


class TTLCache(OrderedDict[str, tuple[float, V]]):
    """OrderedDict-based cache with max_size and TTL eviction.

    Each entry stores ``(timestamp, value)``. Expired or oldest entries are
    evicted on ``get`` / ``put`` to keep memory bounded.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 3600.0) -> None:
        super().__init__()
        self._max_size = max_size
        self._ttl = ttl_seconds

    def get_value(self, key: str) -> V | None:
        entry = super().get(key)
        if entry is None:
            return None
        ts, val = entry
        if time.monotonic() - ts > self._ttl:
            del self[key]
            return None
        self.move_to_end(key)
        return val

    def put(self, key: str, value: V) -> None:
        if key in self:
            del self[key]
        self[key] = (time.monotonic(), value)
        self._evict()

    def _evict(self) -> None:
        now = time.monotonic()
        expired = [k for k, (ts, _) in self.items() if now - ts > self._ttl]
        for k in expired:
            del self[k]
        while len(self) > self._max_size:
            self.popitem(last=False)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        entry = super().get(key)
        if entry is None:
            return False
        ts, _ = entry
        if time.monotonic() - ts > self._ttl:
            del self[key]
            return False
        return True
