"""Rate limiting and circuit breaker for agent stability."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Optional

from src.agent.hooks.lifecycle import LifecycleHook
from src.agent.hooks.middleware import LlmMiddleware
from src.agent.types import LlmRequest, LlmResponse, ToolContext

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when a user exceeds the request rate limit."""


class TokenBucket:
    """Simple token-bucket rate limiter."""

    def __init__(self, rate: float, capacity: int) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()

    def acquire(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class RateLimitHook(LifecycleHook):
    """Per-user rate limiting using token-bucket algorithm."""

    _MAX_BUCKETS = 1024

    def __init__(self, requests_per_minute: int = 20) -> None:
        self._rpm = requests_per_minute
        self._buckets: dict[str, TokenBucket] = {}

    def _get_bucket(self, user_id: str) -> TokenBucket:
        if user_id not in self._buckets:
            if len(self._buckets) >= self._MAX_BUCKETS:
                oldest_key = next(iter(self._buckets))
                del self._buckets[oldest_key]
            self._buckets[user_id] = TokenBucket(
                rate=self._rpm / 60.0,
                capacity=self._rpm,
            )
        return self._buckets[user_id]

    async def before_message(self, user_id: str, message: str) -> Optional[str]:
        bucket = self._get_bucket(user_id)
        if not bucket.acquire():
            raise RateLimitExceeded(f"请求过于频繁，请稍后再试（限制: {self._rpm} 次/分钟）")
        return None


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Three-state circuit breaker for LLM calls."""

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0) -> None:
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self._cooldown:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def record_success(self) -> None:
        self._failures = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning("Circuit breaker OPEN after %d failures", self._failures)

    def allow_request(self) -> bool:
        s = self.state
        if s == CircuitState.CLOSED:
            return True
        if s == CircuitState.HALF_OPEN:
            return True
        return False
