"""Unit tests for LlmService base and factory."""

import pytest

from src.agent.llm.base import LlmService
from src.agent.llm.factory import create_llm_service


class TestLlmServiceABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            LlmService()

    def test_validate_tools_valid(self):
        class Dummy(LlmService):
            async def send_request(self, request):
                pass
            async def stream_request(self, request):
                yield
        d = Dummy()
        errors = d.validate_tools([
            {"type": "function", "function": {"name": "t", "parameters": {}}}
        ])
        assert errors == []

    def test_validate_tools_invalid(self):
        class Dummy(LlmService):
            async def send_request(self, request):
                pass
            async def stream_request(self, request):
                yield
        d = Dummy()
        errors = d.validate_tools([{"type": "invalid"}])
        assert len(errors) > 0


class TestLlmServiceFactory:
    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_service({"llm": {"provider": "invalid"}})

    def test_create_ollama(self):
        svc = create_llm_service({"llm": {"provider": "ollama", "model": "llama3"}})
        assert svc is not None

    def test_create_deepseek(self):
        svc = create_llm_service({"llm": {"provider": "deepseek", "api_key": "test"}})
        assert svc is not None
