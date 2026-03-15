"""Unit tests for offline evaluation script helpers."""

from __future__ import annotations

from types import SimpleNamespace

from scripts.evaluate import _FallbackSearchAdapter as EvaluateFallbackSearchAdapter
from scripts.eval_retrieval import _FallbackSearchAdapter as RetrievalFallbackSearchAdapter
from src.observability.evaluation.source_eval_search import SparseSourceEvalSearch


class _StubSearch:
    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.calls: list[tuple[str, int, object]] = []

    def search(self, *, query: str, top_k: int, filters=None):
        self.calls.append((query, top_k, filters))
        if self._error is not None:
            raise self._error
        return self._result


def test_evaluate_fallback_uses_sparse_when_primary_raises():
    primary = _StubSearch(error=RuntimeError("boom"))
    fallback = _StubSearch(result=["fallback"])

    adapter = EvaluateFallbackSearchAdapter(primary, fallback)

    assert adapter.search(query="q", top_k=3, filters={"collection": "c"}) == ["fallback"]
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_retrieval_fallback_uses_sparse_when_primary_is_empty():
    primary = _StubSearch(result=[])
    fallback = _StubSearch(result=["fallback"])

    adapter = RetrievalFallbackSearchAdapter(primary, fallback)

    assert adapter.search(query="q", top_k=3) == ["fallback"]
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_retrieval_fallback_accepts_non_empty_result_payload():
    primary = _StubSearch(result=SimpleNamespace(results=["hit"]))
    fallback = _StubSearch(result=["fallback"])

    adapter = RetrievalFallbackSearchAdapter(primary, fallback)

    result = adapter.search(query="q", top_k=3)

    assert result.results == ["hit"]
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 0


def test_sparse_source_eval_search_skips_hits_without_source_mapping():
    search = SparseSourceEvalSearch.__new__(SparseSourceEvalSearch)
    search._default_collection = "computer_network"
    search._query_processor = SimpleNamespace(process=lambda query: SimpleNamespace(keywords=["tcp"]))
    search._sparse_index = SimpleNamespace(
        query=lambda **kwargs: [
            {"chunk_id": "chunk-a", "doc_hash": "known", "score": 0.9},
            {"chunk_id": "chunk-b", "doc_hash": "unknown", "score": 0.8},
        ]
    )
    search._source_lookup = {"known": "/tmp/chapter1.pdf"}

    results = search.search(query="tcp", top_k=5)

    assert [item.chunk_id for item in results] == ["chunk-a"]
    assert results[0].metadata["source_label"] == "chapter1.pdf"
