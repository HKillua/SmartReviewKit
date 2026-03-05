"""LLM service factory — creates the correct LlmService from settings."""

from __future__ import annotations

import os
from typing import Any

from src.agent.llm.base import LlmService


def _resolve_api_key(llm_cfg: dict[str, Any], env_var: str) -> str:
    """Return api_key from config, falling back to an environment variable."""
    key = llm_cfg.get("api_key", "") or ""
    if not key:
        key = os.environ.get(env_var, "")
    return key


def create_llm_service(settings: dict[str, Any]) -> LlmService:
    """Instantiate the appropriate LlmService based on ``settings["llm"]["provider"]``.

    Supported providers: ``openai``, ``azure``, ``deepseek``, ``ollama``, ``zhipu``.
    """
    llm_cfg: dict[str, Any] = settings.get("llm", {})
    provider = llm_cfg.get("provider", "").lower()

    if provider in ("openai", "azure"):
        from src.agent.llm.openai_service import OpenAILlmService

        return OpenAILlmService(
            api_key=llm_cfg.get("api_key", ""),
            model=llm_cfg.get("model", "gpt-4o"),
            azure_endpoint=llm_cfg.get("azure_endpoint") if provider == "azure" else None,
            api_version=llm_cfg.get("api_version"),
            deployment_name=llm_cfg.get("deployment_name"),
            is_azure=(provider == "azure"),
        )

    if provider == "deepseek":
        from src.agent.llm.deepseek_service import DeepSeekLlmService

        return DeepSeekLlmService(
            api_key=llm_cfg.get("api_key", ""),
            model=llm_cfg.get("model", "deepseek-chat"),
        )

    if provider == "ollama":
        from src.agent.llm.ollama_service import OllamaLlmService

        return OllamaLlmService(
            model=llm_cfg.get("model", "llama3"),
            base_url=llm_cfg.get("base_url", "http://localhost:11434/v1"),
        )

    if provider == "zhipu":
        from src.agent.llm.openai_service import OpenAILlmService

        return OpenAILlmService(
            api_key=_resolve_api_key(llm_cfg, "ZHIPU_API_KEY"),
            model=llm_cfg.get("model", "glm-4-flash"),
            base_url=llm_cfg.get("base_url", "https://open.bigmodel.cn/api/paas/v4/"),
        )

    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        "Supported: openai, azure, deepseek, ollama, zhipu"
    )
