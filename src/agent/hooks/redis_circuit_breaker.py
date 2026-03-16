"""Redis-backed circuit breaker for multi-instance LLM retries."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from src.agent.hooks.rate_limit import CircuitBreakerBackend, CircuitState

logger = logging.getLogger(__name__)

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional in dev
    redis = None
    REDIS_AVAILABLE = False


_ALLOW_REQUEST_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local cooldown = tonumber(ARGV[2])
local owner = ARGV[3]
local probe_ttl = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'state', 'opened_at', 'probe_owner', 'probe_until')
local state = data[1]
local opened_at = tonumber(data[2])
local probe_owner = data[3]
local probe_until = tonumber(data[4])

if not state or state == '' then
  return {'closed', '1'}
end

if state == 'open' then
  if opened_at and (now - opened_at) < cooldown then
    return {'open', '0'}
  end
  state = 'half_open'
  redis.call('HSET', key, 'state', state)
  redis.call('HDEL', key, 'opened_at')
  if probe_until and probe_until <= now then
    redis.call('HDEL', key, 'probe_owner', 'probe_until')
    probe_owner = false
    probe_until = nil
  end
end

if state == 'half_open' then
  if probe_owner == owner then
    return {'half_open', '1'}
  end
  if (not probe_until) or probe_until <= now then
    redis.call('HSET', key, 'state', 'half_open', 'probe_owner', owner, 'probe_until', now + probe_ttl)
    redis.call('HDEL', key, 'opened_at')
    return {'half_open', '1'}
  end
  return {'half_open', '0'}
end

return {'closed', '1'}
"""


_RECORD_FAILURE_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local threshold = tonumber(ARGV[2])

local data = redis.call('HMGET', key, 'state', 'failures')
local state = data[1]
local failures = tonumber(data[2]) or 0

if not state or state == '' then
  state = 'closed'
end

if state == 'half_open' then
  redis.call('HSET', key, 'state', 'open', 'failures', threshold, 'opened_at', now)
  redis.call('HDEL', key, 'probe_owner', 'probe_until')
  return 'open'
end

if state == 'open' then
  return 'open'
end

failures = failures + 1
if failures >= threshold then
  redis.call('HSET', key, 'state', 'open', 'failures', failures, 'opened_at', now)
  redis.call('HDEL', key, 'probe_owner', 'probe_until')
  return 'open'
end

redis.call('HSET', key, 'state', 'closed', 'failures', failures)
redis.call('HDEL', key, 'opened_at', 'probe_owner', 'probe_until')
return 'closed'
"""


_RECORD_SUCCESS_LUA = """
local key = KEYS[1]
local owner = ARGV[1]

local data = redis.call('HMGET', key, 'state', 'probe_owner')
local state = data[1]
local probe_owner = data[2]

if not state or state == '' then
  redis.call('DEL', key)
  return 'closed'
end

if state == 'open' then
  return 'open'
end

if state == 'half_open' and probe_owner and probe_owner ~= owner then
  return 'half_open'
end

redis.call('DEL', key)
return 'closed'
"""


_STATE_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local cooldown = tonumber(ARGV[2])

local data = redis.call('HMGET', key, 'state', 'opened_at', 'probe_until')
local state = data[1]
local opened_at = tonumber(data[2])
local probe_until = tonumber(data[3])

if not state or state == '' then
  return 'closed'
end

if state == 'open' then
  if opened_at and (now - opened_at) >= cooldown then
    redis.call('HSET', key, 'state', 'half_open')
    redis.call('HDEL', key, 'opened_at')
    if probe_until and probe_until <= now then
      redis.call('HDEL', key, 'probe_owner', 'probe_until')
    end
    return 'half_open'
  end
  return 'open'
end

if state == 'half_open' and probe_until and probe_until <= now then
  redis.call('HDEL', key, 'probe_owner', 'probe_until')
end

return state
"""


class RedisCircuitBreaker(CircuitBreakerBackend):
    """Cluster-wide circuit breaker backed by a shared Redis key."""

    _fallback_locks: dict[int, threading.Lock] = {}
    _fallback_locks_guard = threading.Lock()

    def __init__(
        self,
        redis_url: str,
        *,
        scope_key: str,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        probe_ttl_seconds: float | None = None,
        redis_client: Any | None = None,
        instance_id: str | None = None,
        ping_on_init: bool = True,
    ) -> None:
        if not REDIS_AVAILABLE and redis_client is None:
            raise ImportError("redis is required for RedisCircuitBreaker. Install with: pip install redis")
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._probe_ttl = probe_ttl_seconds or cooldown_seconds
        self._scope_key = scope_key
        self._instance_id = instance_id or uuid.uuid4().hex
        self._redis = redis_client or redis.Redis.from_url(redis_url, decode_responses=True)
        if ping_on_init:
            try:
                self._redis.ping()
            except Exception as exc:  # pragma: no cover - exercised via unit tests
                raise RuntimeError(
                    f"Redis circuit breaker requires reachable Redis at {redis_url}"
                ) from exc

    @property
    def scope_key(self) -> str:
        return self._scope_key

    @classmethod
    def _fallback_lock_for(cls, redis_client: Any) -> threading.Lock:
        client_id = id(redis_client)
        with cls._fallback_locks_guard:
            lock = cls._fallback_locks.get(client_id)
            if lock is None:
                lock = threading.Lock()
                cls._fallback_locks[client_id] = lock
            return lock

    def _execute_script(self, script: str, *, keys: list[str], args: list[Any]) -> Any:
        try:
            return self._redis.eval(script, len(keys), *(keys + args))
        except Exception as exc:
            if "unknown command 'eval'" not in str(exc).lower():
                raise
            return self._execute_fallback(script, keys=keys, args=args)

    def _execute_fallback(self, script: str, *, keys: list[str], args: list[Any]) -> Any:
        key = keys[0]
        with self._fallback_lock_for(self._redis):
            if script == _STATE_LUA:
                now = float(args[0])
                cooldown = float(args[1])
                state, opened_at, probe_until = self._redis.hmget(
                    key, "state", "opened_at", "probe_until"
                )
                if not state:
                    return CircuitState.CLOSED.value
                if state == CircuitState.OPEN.value:
                    if opened_at and (now - float(opened_at)) >= cooldown:
                        self._redis.hset(key, mapping={"state": CircuitState.HALF_OPEN.value})
                        self._redis.hdel(key, "opened_at")
                        if probe_until and float(probe_until) <= now:
                            self._redis.hdel(key, "probe_owner", "probe_until")
                        return CircuitState.HALF_OPEN.value
                    return CircuitState.OPEN.value
                if state == CircuitState.HALF_OPEN.value and probe_until and float(probe_until) <= now:
                    self._redis.hdel(key, "probe_owner", "probe_until")
                return state

            if script == _ALLOW_REQUEST_LUA:
                now = float(args[0])
                cooldown = float(args[1])
                owner = str(args[2])
                probe_ttl = float(args[3])
                state, opened_at, probe_owner, probe_until = self._redis.hmget(
                    key, "state", "opened_at", "probe_owner", "probe_until"
                )
                if not state:
                    return [CircuitState.CLOSED.value, "1"]
                if state == CircuitState.OPEN.value:
                    if opened_at and (now - float(opened_at)) < cooldown:
                        return [CircuitState.OPEN.value, "0"]
                    state = CircuitState.HALF_OPEN.value
                    self._redis.hset(key, mapping={"state": state})
                    self._redis.hdel(key, "opened_at")
                    if probe_until and float(probe_until) <= now:
                        self._redis.hdel(key, "probe_owner", "probe_until")
                        probe_owner = None
                        probe_until = None
                if state == CircuitState.HALF_OPEN.value:
                    if probe_owner == owner:
                        return [CircuitState.HALF_OPEN.value, "1"]
                    if (not probe_until) or float(probe_until) <= now:
                        self._redis.hset(
                            key,
                            mapping={
                                "state": CircuitState.HALF_OPEN.value,
                                "probe_owner": owner,
                                "probe_until": now + probe_ttl,
                            },
                        )
                        self._redis.hdel(key, "opened_at")
                        return [CircuitState.HALF_OPEN.value, "1"]
                    return [CircuitState.HALF_OPEN.value, "0"]
                return [CircuitState.CLOSED.value, "1"]

            if script == _RECORD_FAILURE_LUA:
                now = float(args[0])
                threshold = int(args[1])
                state, failures = self._redis.hmget(key, "state", "failures")
                current_state = state or CircuitState.CLOSED.value
                failure_count = int(failures or 0)
                if current_state == CircuitState.HALF_OPEN.value:
                    self._redis.hset(
                        key,
                        mapping={
                            "state": CircuitState.OPEN.value,
                            "failures": threshold,
                            "opened_at": now,
                        },
                    )
                    self._redis.hdel(key, "probe_owner", "probe_until")
                    return CircuitState.OPEN.value
                if current_state == CircuitState.OPEN.value:
                    return CircuitState.OPEN.value
                failure_count += 1
                if failure_count >= threshold:
                    self._redis.hset(
                        key,
                        mapping={
                            "state": CircuitState.OPEN.value,
                            "failures": failure_count,
                            "opened_at": now,
                        },
                    )
                    self._redis.hdel(key, "probe_owner", "probe_until")
                    return CircuitState.OPEN.value
                self._redis.hset(
                    key,
                    mapping={
                        "state": CircuitState.CLOSED.value,
                        "failures": failure_count,
                    },
                )
                self._redis.hdel(key, "opened_at", "probe_owner", "probe_until")
                return CircuitState.CLOSED.value

            if script == _RECORD_SUCCESS_LUA:
                owner = str(args[0])
                state, probe_owner = self._redis.hmget(key, "state", "probe_owner")
                if not state:
                    self._redis.delete(key)
                    return CircuitState.CLOSED.value
                if state == CircuitState.OPEN.value:
                    return CircuitState.OPEN.value
                if state == CircuitState.HALF_OPEN.value and probe_owner and probe_owner != owner:
                    return CircuitState.HALF_OPEN.value
                self._redis.delete(key)
                return CircuitState.CLOSED.value

        raise RuntimeError("Unsupported fallback script execution path")

    @property
    def state(self) -> CircuitState:
        raw_state = self._execute_script(
            _STATE_LUA,
            keys=[self._scope_key],
            args=[time.time(), self._cooldown],
        )
        return CircuitState(str(raw_state))

    def record_success(self) -> None:
        raw_state = self._execute_script(
            _RECORD_SUCCESS_LUA,
            keys=[self._scope_key],
            args=[self._instance_id],
        )
        if str(raw_state) == CircuitState.CLOSED.value:
            logger.debug("Redis circuit breaker CLOSED for %s", self._scope_key)

    def record_failure(self) -> None:
        raw_state = self._execute_script(
            _RECORD_FAILURE_LUA,
            keys=[self._scope_key],
            args=[time.time(), self._threshold],
        )
        if str(raw_state) == CircuitState.OPEN.value:
            logger.warning("Redis circuit breaker OPEN for %s", self._scope_key)

    def allow_request(self) -> bool:
        state, allowed = self._execute_script(
            _ALLOW_REQUEST_LUA,
            keys=[self._scope_key],
            args=[time.time(), self._cooldown, self._instance_id, self._probe_ttl],
        )
        return str(allowed) == "1" and str(state) in {
            CircuitState.CLOSED.value,
            CircuitState.HALF_OPEN.value,
        }
