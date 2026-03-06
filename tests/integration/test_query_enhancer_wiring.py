"""Integration tests for QueryEnhancer wiring.

Validates that the llm_service property setter works and that
rewrite / hyde_embed / decompose correctly invoke the LLM when set.

Usage::

    pytest tests/integration/test_query_enhancer_wiring.py -v
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List
from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.integration]


@dataclass
class FakeLLMResponse:
    content: str


class FakeLLMService:
    """Minimal mock that records calls and returns configurable responses."""

    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self._call_count = 0
        self.last_messages: Any = None

    async def chat(self, messages, **kwargs):
        self.last_messages = messages
        self._call_count += 1
        idx = min(self._call_count - 1, len(self._responses) - 1)
        return FakeLLMResponse(content=self._responses[idx])

    @property
    def call_count(self):
        return self._call_count


def _fake_embed_fn(texts):
    return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def enhancer():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.core.query_engine.query_enhancer import QueryEnhancer
    return QueryEnhancer(embedding_fn=_fake_embed_fn)


class TestLlmServiceProperty:
    def test_default_is_none(self, enhancer):
        assert enhancer.llm_service is None

    def test_setter_updates_internal(self, enhancer):
        fake = FakeLLMService(["hello"])
        enhancer.llm_service = fake
        assert enhancer._llm is fake
        assert enhancer.llm_service is fake


class TestRewrite:
    def test_rewrite_without_llm_returns_original(self, enhancer):
        result = asyncio.get_event_loop().run_until_complete(
            enhancer.rewrite("TCP三次握手")
        )
        assert result == "TCP三次握手"

    def test_rewrite_with_llm_returns_rewritten(self, enhancer):
        fake = FakeLLMService(["TCP协议三次握手建立连接过程"])
        enhancer.llm_service = fake
        result = asyncio.get_event_loop().run_until_complete(
            enhancer.rewrite("TCP三次握手")
        )
        assert result == "TCP协议三次握手建立连接过程"
        assert fake.call_count == 1


class TestHydeEmbed:
    def test_hyde_without_llm_returns_none(self, enhancer):
        result = asyncio.get_event_loop().run_until_complete(
            enhancer.hyde_embed("什么是OSPF")
        )
        assert result is None

    def test_hyde_with_llm_returns_vector(self, enhancer):
        fake = FakeLLMService(["OSPF是一种链路状态路由协议，工作在AS内部..."])
        enhancer.llm_service = fake
        result = asyncio.get_event_loop().run_until_complete(
            enhancer.hyde_embed("什么是OSPF")
        )
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 3  # _fake_embed_fn returns 3-dim vectors
        assert fake.call_count == 1


class TestDecompose:
    def test_decompose_without_llm_returns_original(self, enhancer):
        result = asyncio.get_event_loop().run_until_complete(
            enhancer.decompose("比较TCP和UDP")
        )
        assert result == ["比较TCP和UDP"]

    def test_decompose_with_llm_returns_subqueries(self, enhancer):
        fake = FakeLLMService(['["TCP的特点是什么", "UDP的特点是什么", "TCP和UDP的区别"]'])
        enhancer.llm_service = fake
        result = asyncio.get_event_loop().run_until_complete(
            enhancer.decompose("比较TCP和UDP")
        )
        assert len(result) == 3
        assert "TCP" in result[0]
        assert fake.call_count == 1

    def test_decompose_with_invalid_json_returns_original(self, enhancer):
        fake = FakeLLMService(["not valid json"])
        enhancer.llm_service = fake
        result = asyncio.get_event_loop().run_until_complete(
            enhancer.decompose("比较TCP和UDP")
        )
        assert result == ["比较TCP和UDP"]


class TestAppWiring:
    """Verify the app.py wiring pattern works end-to-end."""

    def test_app_wiring_pattern(self, enhancer):
        fake = FakeLLMService(["rewritten query"])
        enhancer.llm_service = fake

        assert enhancer._llm is fake

        result = asyncio.get_event_loop().run_until_complete(
            enhancer.rewrite("原始查询")
        )
        assert result == "rewritten query"
        assert fake.call_count == 1
