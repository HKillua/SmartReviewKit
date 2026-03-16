"""Tests for the distributed Redis circuit breaker."""

from __future__ import annotations

import textwrap
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.agent.hooks.rate_limit import CircuitBreaker, CircuitState
from src.agent.hooks.redis_circuit_breaker import RedisCircuitBreaker
from src.agent.hooks.retry_middleware import RetryWithBackoffMiddleware
from src.agent.types import LlmMessage, LlmRequest, LlmResponse
from src.core.settings import load_settings
from src.server.app import _build_retry_middleware


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def _fake_redis():
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeRedis(decode_responses=True)


def _breaker(fake_client, *, scope_key: str = "circuit_breaker:test:model:123", instance_id: str) -> RedisCircuitBreaker:
    return RedisCircuitBreaker(
        "redis://unused/0",
        scope_key=scope_key,
        failure_threshold=3,
        cooldown_seconds=0.05,
        redis_client=fake_client,
        ping_on_init=False,
        instance_id=instance_id,
    )


def _open_breaker(cb: RedisCircuitBreaker) -> None:
    for _ in range(3):
        cb.record_failure()


class TestRedisCircuitBreaker:
    def test_opens_after_threshold_across_instances(self) -> None:
        fake_client = _fake_redis()
        breaker_a = _breaker(fake_client, instance_id="a")
        breaker_b = _breaker(fake_client, instance_id="b")

        _open_breaker(breaker_a)

        assert breaker_a.state == CircuitState.OPEN
        assert breaker_b.state == CircuitState.OPEN

    def test_opened_instance_blocks_other_instances(self) -> None:
        fake_client = _fake_redis()
        breaker_a = _breaker(fake_client, instance_id="a")
        breaker_b = _breaker(fake_client, instance_id="b")

        _open_breaker(breaker_a)

        assert not breaker_b.allow_request()

    def test_cooldown_transitions_to_half_open(self) -> None:
        fake_client = _fake_redis()
        breaker_a = _breaker(fake_client, instance_id="a")
        breaker_b = _breaker(fake_client, instance_id="b")

        _open_breaker(breaker_a)
        time.sleep(0.06)

        assert breaker_a.state == CircuitState.HALF_OPEN
        assert breaker_b.state == CircuitState.HALF_OPEN

    def test_only_one_instance_acquires_half_open_probe(self) -> None:
        fake_client = _fake_redis()
        breaker_a = _breaker(fake_client, instance_id="a")
        breaker_b = _breaker(fake_client, instance_id="b")

        _open_breaker(breaker_a)
        time.sleep(0.06)

        assert breaker_a.allow_request() is True
        assert breaker_b.allow_request() is False

    def test_half_open_success_closes_cluster_wide(self) -> None:
        fake_client = _fake_redis()
        breaker_a = _breaker(fake_client, instance_id="a")
        breaker_b = _breaker(fake_client, instance_id="b")

        _open_breaker(breaker_a)
        time.sleep(0.06)
        assert breaker_a.allow_request() is True

        breaker_a.record_success()

        assert breaker_b.state == CircuitState.CLOSED
        assert breaker_b.allow_request() is True

    def test_half_open_failure_reopens_cluster_wide(self) -> None:
        fake_client = _fake_redis()
        breaker_a = _breaker(fake_client, instance_id="a")
        breaker_b = _breaker(fake_client, instance_id="b")

        _open_breaker(breaker_a)
        time.sleep(0.06)
        assert breaker_a.allow_request() is True

        breaker_a.record_failure()

        assert breaker_b.state == CircuitState.OPEN
        assert breaker_b.allow_request() is False


class TestRetryMiddlewareDistributedBreaker:
    @pytest.mark.asyncio
    async def test_two_middlewares_share_redis_breaker_state(self) -> None:
        fake_client = _fake_redis()
        breaker_a = RedisCircuitBreaker(
            "redis://unused/0",
            scope_key="circuit_breaker:test:model:retry",
            failure_threshold=1,
            cooldown_seconds=0.2,
            redis_client=fake_client,
            ping_on_init=False,
            instance_id="retry-a",
        )
        breaker_b = RedisCircuitBreaker(
            "redis://unused/0",
            scope_key="circuit_breaker:test:model:retry",
            failure_threshold=1,
            cooldown_seconds=0.2,
            redis_client=fake_client,
            ping_on_init=False,
            instance_id="retry-b",
        )

        request = LlmRequest(messages=[LlmMessage(role="user", content="test")])
        initial_response = LlmResponse(error="timeout error")

        llm_a = AsyncMock()
        llm_a.send_request = AsyncMock(return_value=LlmResponse(error="timeout error"))
        mw_a = RetryWithBackoffMiddleware(
            llm_service=llm_a,
            max_retries=1,
            base_delay=0.01,
            circuit_breaker=breaker_a,
        )

        llm_b = AsyncMock()
        llm_b.send_request = AsyncMock(return_value=LlmResponse(content="should not happen"))
        mw_b = RetryWithBackoffMiddleware(
            llm_service=llm_b,
            max_retries=1,
            base_delay=0.01,
            circuit_breaker=breaker_b,
        )

        result_a = await mw_a.after_llm_response(request, initial_response)
        result_b = await mw_b.after_llm_response(request, initial_response)

        assert result_a.error == "timeout error"
        assert result_b.error == "服务暂时不可用（熔断器已打开），请稍后重试。"
        llm_b.send_request.assert_not_called()


def test_build_retry_middleware_reads_llm_resilience_config(tmp_path: Path) -> None:
    config = """
    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
      max_tokens: 1024
    embedding:
      provider: openai
      model: text-embedding-3-small
      dimensions: 1536
    vector_store:
      provider: chroma
      persist_directory: ./data/db/chroma
      collection_name: knowledge_hub
    retrieval:
      dense_top_k: 20
      sparse_top_k: 20
      fusion_top_k: 10
      rrf_k: 60
    rerank:
      enabled: false
      provider: none
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k: 5
    evaluation:
      enabled: false
      provider: custom
      metrics: [hit_rate]
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    llm_resilience:
      retry:
        max_retries: 6
        base_delay_seconds: 0.25
        max_delay_seconds: 5.0
      circuit_breaker:
        failure_threshold: 4
        cooldown_seconds: 12.0
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)
    settings = load_settings(settings_path)

    middleware = _build_retry_middleware(
        settings,
        llm_service=object(),
        circuit_breaker=CircuitBreaker(),
    )

    assert middleware._max_retries == 6
    assert middleware._base_delay == 0.25
    assert middleware._max_delay == 5.0
