"""Retry middleware with exponential backoff and circuit breaker integration."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from src.agent.hooks.middleware import LlmMiddleware
from src.agent.hooks.rate_limit import CircuitBreaker
from src.agent.types import LlmRequest, LlmResponse

logger = logging.getLogger(__name__)

_RETRYABLE_PATTERNS = re.compile(
    r"(timeout|rate.?limit|429|502|503|504|overloaded|capacity|try again)",
    re.IGNORECASE,
)


class RetryWithBackoffMiddleware(LlmMiddleware):
    """Retries failed LLM calls with exponential backoff.

    Integrates a CircuitBreaker to avoid hammering a failing service.
    """

    def __init__(
        self,
        *,
        llm_service: object,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 16.0,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self._llm = llm_service
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._cb = circuit_breaker or CircuitBreaker()

    @staticmethod
    def _is_retryable(error: str) -> bool:
        return bool(_RETRYABLE_PATTERNS.search(error))

    async def after_llm_response(
        self, request: LlmRequest, response: LlmResponse
    ) -> LlmResponse:
        if not response.error:
            self._cb.record_success()
            return response

        if not self._is_retryable(response.error):
            self._cb.record_failure()
            return response

        for attempt in range(1, self._max_retries + 1):
            if not self._cb.allow_request():
                logger.warning("Circuit breaker OPEN — skipping retry #%d", attempt)
                return LlmResponse(
                    error="服务暂时不可用（熔断器已打开），请稍后重试。"
                )

            delay = min(self._base_delay * (2 ** (attempt - 1)), self._max_delay)
            logger.info(
                "Retrying LLM call (attempt %d/%d) after %.1fs — error: %s",
                attempt, self._max_retries, delay, response.error[:100],
            )
            await asyncio.sleep(delay)

            try:
                response = await self._llm.send_request(request)
            except Exception as exc:
                response = LlmResponse(error=str(exc))

            if not response.error:
                self._cb.record_success()
                logger.info("Retry #%d succeeded", attempt)
                return response

            self._cb.record_failure()

            if not self._is_retryable(response.error):
                break

        return response
