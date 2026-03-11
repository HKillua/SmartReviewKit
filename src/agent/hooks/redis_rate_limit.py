"""Redis-backed rate limiting for multi-instance agent deployments."""

from __future__ import annotations

import time
from typing import Optional

from src.agent.hooks.lifecycle import LifecycleHook
from src.agent.hooks.rate_limit import RateLimitExceeded

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional in dev
    redis = None
    REDIS_AVAILABLE = False


_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(data[1])
local last_refill = tonumber(data[2])

if tokens == nil then
  tokens = capacity
  last_refill = now
end

local elapsed = math.max(0, now - last_refill)
tokens = math.min(capacity, tokens + elapsed * rate)

if tokens < 1.0 then
  redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
  redis.call('EXPIRE', key, ttl)
  return 0
end

tokens = tokens - 1.0
redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
redis.call('EXPIRE', key, ttl)
return 1
"""


class RedisRateLimitHook(LifecycleHook):
    """Distributed per-user rate limiting backed by Redis token buckets."""

    def __init__(self, redis_url: str, requests_per_minute: int = 20) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError("redis is required for RedisRateLimitHook. Install with: pip install redis")
        self._rpm = requests_per_minute
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._script = self._redis.register_script(_TOKEN_BUCKET_LUA)

    async def before_message(self, user_id: str, message: str) -> Optional[str]:
        allowed = self._script(
            keys=[f"rate_limit:{user_id}"],
            args=[self._rpm / 60.0, self._rpm, time.time(), 120],
        )
        if int(allowed) != 1:
            raise RateLimitExceeded(f"请求过于频繁，请稍后再试（限制: {self._rpm} 次/分钟）")
        return None
