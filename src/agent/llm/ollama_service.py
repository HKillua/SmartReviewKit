"""Ollama LLM service — uses OpenAI-compatible endpoint at localhost."""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from src.agent.llm.openai_service import OpenAILlmService
from src.agent.types import LlmRequest, LlmResponse, LlmStreamChunk

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaLlmService(OpenAILlmService):
    """Ollama uses an OpenAI-compatible API at a local endpoint.

    Tool calling support varies by model.  When the model does not return
    structured ``tool_calls``, the raw text is still returned so the Agent
    can decide how to proceed.
    """

    def __init__(
        self,
        *,
        model: str = "llama3",
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
    ) -> None:
        super().__init__(api_key="ollama", model=model, base_url=base_url)

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        resp = await super().send_request(request)
        if resp.error and request.tools:
            logger.warning("Ollama tool-calling may not be supported; retrying without tools")
            fallback_req = request.model_copy(update={"tools": None})
            resp = await super().send_request(fallback_req)
        return resp

    async def stream_request(self, request: LlmRequest) -> AsyncGenerator[LlmStreamChunk, None]:
        try:
            async for chunk in super().stream_request(request):
                yield chunk
        except Exception:
            if request.tools:
                logger.warning("Ollama streaming with tools failed; retrying without tools")
                fallback_req = request.model_copy(update={"tools": None})
                async for chunk in super().stream_request(fallback_req):
                    yield chunk
            else:
                raise
