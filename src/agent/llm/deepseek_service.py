"""DeepSeek LLM service — thin wrapper over OpenAILlmService with custom base URL."""

from __future__ import annotations

from src.agent.llm.openai_service import OpenAILlmService

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekLlmService(OpenAILlmService):
    """DeepSeek uses an OpenAI-compatible API."""

    def __init__(self, *, api_key: str = "", model: str = "deepseek-chat") -> None:
        super().__init__(api_key=api_key, model=model, base_url=DEEPSEEK_BASE_URL)
