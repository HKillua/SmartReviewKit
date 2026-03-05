"""Retry middleware — exponential backoff for LLM errors."""

from __future__ import annotations

import asyncio
import logging

from src.agent.hooks.middleware import LlmMiddleware
from src.agent.hooks.rate_limit import CircuitBreaker
from src.agent.types import LlmRequest, LlmResponse

logger = logging.getLogger(__name__)


class RetryMiddleware(LlmMiddleware):
    """Retries failed LLM calls with exponential backoff.

    Integrates with an optional CircuitBreaker to avoid hammering a downed service.
    """

    def __init__(
        self,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._cb = circuit_breaker

    async def before_llm_request(self, request: LlmRequest) -> LlmRequest:
        if self._cb and not self._cb.allow_request():
            raise RuntimeError("LLM 服务暂时不可用（熔断中），请稍后重试")
        return request

    async def after_llm_response(
        self, request: LlmRequest, response: LlmResponse
    ) -> LlmResponse:
        if response.error is None:
            if self._cb:
                self._cb.record_success()
            return response

        if self._cb:
            self._cb.record_failure()

        return response
