from __future__ import annotations

from types import SimpleNamespace

from src.core.query_engine.source_aware_search import SourceAwareSearch
from src.core.types import RetrievalResult


def _result(
    chunk_id: str,
    text: str,
    score: float = 0.9,
    **metadata,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata=metadata,
    )


def _hybrid_with_search(search_fn):
    return SimpleNamespace(
        search=search_fn,
        config=SimpleNamespace(
            rerank_enabled=False,
            mmr_enabled=False,
            post_dedup_enabled=False,
        ),
        dense_retriever=SimpleNamespace(
            vector_store=SimpleNamespace(get_by_ids=lambda ids: []),
        ),
    )


def test_question_bank_results_stay_flat_units() -> None:
    def _search(*, query, top_k, filters=None, query_vector=None):
        assert filters == {"source_type": "question_bank"}
        return [
            _result(
                "q1",
                "TCP 是什么？",
                source_type="question_bank",
                answer="传输控制协议",
            )
        ]

    svc = SourceAwareSearch(_hybrid_with_search(_search))
    result = svc.search(
        query="TCP",
        task_intent="quiz_generator",
        top_k=3,
        allowed_sources=["question_bank"],
    )

    assert len(result.answer_units) == 1
    assert result.answer_units[0].unit_kind == "question_item"
    assert result.answer_units[0].backing_chunk_ids == ["q1"]
    assert result.routing_metadata["unit_kind_distribution"] == {"question_item": 1}


def test_textbook_child_hits_collapse_to_parent_unit() -> None:
    parent_record = {
        "id": "parent_1",
        "text": "TCP 教材完整解释。",
        "metadata": {"source_type": "textbook", "title": "TCP 教材"},
    }

    def _search(*, query, top_k, filters=None, query_vector=None):
        assert filters == {"source_type": "textbook"}
        return [
            _result(
                "child_1",
                "TCP 子块 A",
                source_type="textbook",
                parent_chunk_id="parent_1",
                is_parent=False,
            ),
            _result(
                "child_2",
                "TCP 子块 B",
                score=0.82,
                source_type="textbook",
                parent_chunk_id="parent_1",
                is_parent=False,
            ),
        ]

    hybrid = _hybrid_with_search(_search)
    hybrid.dense_retriever.vector_store = SimpleNamespace(get_by_ids=lambda ids: [parent_record])
    svc = SourceAwareSearch(hybrid)

    result = svc.search(
        query="解释 TCP",
        task_intent="knowledge_query",
        top_k=3,
        allowed_sources=["textbook"],
    )

    assert len(result.answer_units) == 1
    unit = result.answer_units[0]
    assert unit.unit_kind == "textbook_parent"
    assert unit.unit_id == "parent_1"
    assert unit.support_count == 2
    assert sorted(unit.backing_chunk_ids) == ["child_1", "child_2"]
    assert result.routing_metadata["collapsed_parent_count"] == 1
    assert result.routing_metadata["parent_promotions"] == 1


def test_global_fallback_runs_when_source_specific_searches_all_empty() -> None:
    def _search(*, query, top_k, filters=None, query_vector=None):
        if filters:
            return []
        return [
            _result(
                "doc_1",
                "TCP 通过三次握手建立连接。",
                source_path="network.md",
                title="TCP",
            )
        ]

    svc = SourceAwareSearch(_hybrid_with_search(_search))
    result = svc.search(query="TCP", task_intent="quiz_generator", top_k=2)

    assert len(result.answer_units) == 1
    assert result.answer_units[0].source_type == "textbook"
    assert result.routing_metadata["fallback_global_used"] is True


def test_duplicate_units_are_coalesced_across_source_queries() -> None:
    def _search(*, query, top_k, filters=None, query_vector=None):
        return [
            _result(
                "shared_chunk",
                "DNS 解析包含递归查询和迭代查询。",
                source_path="dns.pdf",
                title="DNS",
            )
        ]

    svc = SourceAwareSearch(_hybrid_with_search(_search))
    result = svc.search(query="DNS", task_intent="review_summary", top_k=5)

    assert len(result.answer_units) == 1
    assert len(result.results) == 1


def test_slide_results_remain_slide_units() -> None:
    def _search(*, query, top_k, filters=None, query_vector=None):
        assert filters == {"source_type": "slide"}
        return [
            _result(
                "slide_1",
                "TCP 三次握手流程图。",
                source_type="slide",
                source_path="lecture01.pptx",
            )
        ]

    svc = SourceAwareSearch(_hybrid_with_search(_search))
    result = svc.search(
        query="TCP 三次握手",
        task_intent="review_summary",
        top_k=2,
        allowed_sources=["slide"],
    )

    assert len(result.answer_units) == 1
    assert result.answer_units[0].unit_kind == "slide_chunk"
    assert result.answer_units[0].source_type == "slide"


def test_explanatory_profile_orders_textbook_before_question_bank() -> None:
    def _search(*, query, top_k, filters=None, query_vector=None):
        source = filters.get("source_type")
        if source == "textbook":
            return [
                _result(
                    "textbook_1",
                    "TCP 通过三次握手建立可靠连接。",
                    score=0.7,
                    source_type="textbook",
                    is_parent=True,
                )
            ]
        if source == "question_bank":
            return [
                _result(
                    "qb_1",
                    "题目：TCP 三次握手的第二次报文是什么？",
                    score=0.95,
                    source_type="question_bank",
                )
            ]
        return []

    svc = SourceAwareSearch(_hybrid_with_search(_search))
    result = svc.search(
        query="解释 TCP 三次握手",
        task_intent="knowledge_query",
        top_k=3,
    )

    assert [unit.source_type for unit in result.answer_units[:2]] == ["textbook", "question_bank"]
    assert result.routing_metadata["evidence_profile"] == "explanatory"
