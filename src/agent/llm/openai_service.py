"""OpenAI / Azure OpenAI LLM service implementation."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

from src.agent.llm.base import LlmService
from src.agent.types import (
    LlmMessage,
    LlmRequest,
    LlmResponse,
    LlmStreamChunk,
    ToolCallData,
)

logger = logging.getLogger(__name__)


def _build_messages(messages: list[LlmMessage]) -> list[dict[str, Any]]:
    """Convert internal LlmMessage list to OpenAI API format."""
    out: list[dict[str, Any]] = []
    for m in messages:
        msg: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            msg["content"] = m.content
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in m.tool_calls
            ]
        if m.tool_call_id is not None:
            msg["tool_call_id"] = m.tool_call_id
        out.append(msg)
    return out


def _parse_tool_calls(raw_calls: list[Any] | None) -> list[ToolCallData] | None:
    """Parse OpenAI tool_calls into ToolCallData list."""
    if not raw_calls:
        return None
    result: list[ToolCallData] = []
    for tc in raw_calls:
        fn = tc.function
        try:
            args = json.loads(fn.arguments) if fn.arguments else {}
        except json.JSONDecodeError:
            args = {"_raw": fn.arguments}
        result.append(ToolCallData(id=tc.id, name=fn.name, arguments=args))
    return result or None


class OpenAILlmService(LlmService):
    """LlmService backed by the ``openai`` SDK (works for both OpenAI and Azure)."""

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "gpt-4o",
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        deployment_name: str | None = None,
        is_azure: bool = False,
        timeout_seconds: float = 120.0,
        connect_timeout_seconds: float = 10.0,
    ) -> None:
        try:
            import httpx
            import openai
        except ImportError as exc:
            raise ImportError("openai package is required: pip install openai>=1.0") from exc

        http_timeout = httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds)

        if is_azure or azure_endpoint:
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key or None,
                azure_endpoint=azure_endpoint or "",
                api_version=api_version or "2024-02-15-preview",
                timeout=http_timeout,
            )
            self._model = deployment_name or model
        else:
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            kwargs["timeout"] = http_timeout
            self._client = openai.AsyncOpenAI(**kwargs)
            self._model = model

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _build_messages(request.messages),
            "temperature": request.temperature,
            "stream": False,
        }
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = "auto"

        try:
            resp = await self._client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            return LlmResponse(
                content=choice.message.content,
                tool_calls=_parse_tool_calls(choice.message.tool_calls),
                usage=resp.usage.model_dump() if resp.usage else None,
            )
        except Exception as exc:
            logger.exception("LLM request failed")
            return LlmResponse(error=str(exc))

    async def stream_request(self, request: LlmRequest) -> AsyncGenerator[LlmStreamChunk, None]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _build_messages(request.messages),
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = "auto"

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                tool_call_deltas = None
                if delta.tool_calls:
                    tool_call_deltas = [
                        {
                            "index": tc.index,
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": getattr(tc.function, "name", None),
                                "arguments": getattr(tc.function, "arguments", None),
                            } if tc.function else None,
                        }
                        for tc in delta.tool_calls
                    ]
                yield LlmStreamChunk(
                    delta_content=delta.content,
                    delta_tool_calls=tool_call_deltas,
                    finish_reason=chunk.choices[0].finish_reason,
                )
        except Exception as exc:
            logger.exception("LLM stream failed")
            yield LlmStreamChunk(finish_reason="error")

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            await close()
