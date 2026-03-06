"""Unit tests for cost tracking in CachedEmbedding and CrossEncoderReranker.

Validates that API call counters and cache statistics work correctly.

Usage::

    pytest tests/unit/test_cost_tracking.py -v -m unit
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

import pytest

pytestmark = [pytest.mark.unit]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.cached_embedding import CachedEmbedding


class StubEmbedding(BaseEmbedding):
    """Minimal embedding that returns deterministic vectors."""

    def __init__(self, dim: int = 4):
        self._dim = dim
        self.raw_call_count = 0

    def embed(self, texts: List[str], trace=None, **kwargs) -> List[List[float]]:
        self.raw_call_count += 1
        return [[float(i + 1)] * self._dim for i in range(len(texts))]

    def get_dimension(self) -> int:
        return self._dim


class TestCachedEmbeddingStats:
    def test_initial_stats(self):
        cached = CachedEmbedding(StubEmbedding(), max_size=10)
        stats = cached.cache_stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["api_calls"] == 0
        assert stats["texts_embedded"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["size"] == 0

    def test_first_call_all_misses(self):
        cached = CachedEmbedding(StubEmbedding(), max_size=10)
        cached.embed(["hello", "world"])
        stats = cached.cache_stats
        assert stats["misses"] == 2
        assert stats["hits"] == 0
        assert stats["api_calls"] == 1
        assert stats["texts_embedded"] == 2
        assert stats["size"] == 2

    def test_cache_hit(self):
        cached = CachedEmbedding(StubEmbedding(), max_size=10)
        cached.embed(["hello", "world"])
        cached.embed(["hello"])
        stats = cached.cache_stats
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["api_calls"] == 1
        assert stats["hit_rate"] == round(1 / 3, 3)

    def test_partial_cache_hit(self):
        cached = CachedEmbedding(StubEmbedding(), max_size=10)
        cached.embed(["a", "b"])
        cached.embed(["b", "c"])
        stats = cached.cache_stats
        assert stats["hits"] == 1
        assert stats["misses"] == 3
        assert stats["api_calls"] == 2
        assert stats["texts_embedded"] == 3

    def test_full_cache_hit_no_api_call(self):
        stub = StubEmbedding()
        cached = CachedEmbedding(stub, max_size=10)
        cached.embed(["x", "y"])
        assert stub.raw_call_count == 1
        cached.embed(["x", "y"])
        assert stub.raw_call_count == 1
        stats = cached.cache_stats
        assert stats["api_calls"] == 1
        assert stats["hits"] == 2

    def test_eviction(self):
        cached = CachedEmbedding(StubEmbedding(), max_size=2)
        cached.embed(["a"])
        cached.embed(["b"])
        cached.embed(["c"])
        stats = cached.cache_stats
        assert stats["size"] == 2

    def test_vectors_returned_correctly(self):
        cached = CachedEmbedding(StubEmbedding(dim=3), max_size=10)
        v1 = cached.embed(["test"])
        v2 = cached.embed(["test"])
        assert v1 == v2


class TestRerankerStats:
    @pytest.fixture
    def reranker(self):
        from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker

        class FakeModel:
            def predict(self, pairs):
                return [0.9] * len(pairs)

        class FakeSettings:
            class rerank:
                model = "test-model"
                top_k = 5

        return CrossEncoderReranker(
            settings=FakeSettings(),
            model=FakeModel(),
            timeout=5.0,
        )

    def test_initial_stats(self, reranker):
        stats = reranker.rerank_stats
        assert stats["rerank_calls"] == 0
        assert stats["rerank_pairs_scored"] == 0

    def test_rerank_increments_counters(self, reranker):
        candidates = [
            {"text": "TCP is a transport protocol", "chunk_id": "c1", "score": 0.8},
            {"text": "UDP is connectionless", "chunk_id": "c2", "score": 0.7},
            {"text": "HTTP uses TCP", "chunk_id": "c3", "score": 0.6},
        ]
        reranker.rerank("TCP protocol", candidates)
        stats = reranker.rerank_stats
        assert stats["rerank_calls"] == 1
        assert stats["rerank_pairs_scored"] == 3

    def test_multiple_reranks_accumulate(self, reranker):
        c1 = [{"text": "a", "chunk_id": "1", "score": 0.5}]
        c2 = [{"text": "b", "chunk_id": "2", "score": 0.5}, {"text": "c", "chunk_id": "3", "score": 0.4}]
        reranker.rerank("q1", c1)
        reranker.rerank("q2", c2)
        stats = reranker.rerank_stats
        assert stats["rerank_calls"] == 2
        assert stats["rerank_pairs_scored"] == 3
