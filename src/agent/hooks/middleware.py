"""LLM middleware base class — intercept LLM requests and responses."""

from __future__ import annotations

from src.agent.types import LlmRequest, LlmResponse


class LlmMiddleware:
    """Base class with default pass-through implementations.

    Multiple middlewares are chained: ``before_llm_request`` runs first-to-last,
    ``after_llm_response`` runs last-to-first.
    """

    async def before_llm_request(self, request: LlmRequest) -> LlmRequest:
        """Modify or log the request before it reaches the LLM."""
        return request

    async def after_llm_response(
        self, request: LlmRequest, response: LlmResponse
    ) -> LlmResponse:
        """Modify or log the response after the LLM returns."""
        return response
