"""Unit tests for HybridSearch metadata filter compatibility."""

from __future__ import annotations

from src.core.query_engine.hybrid_search import HybridSearch


def test_matches_filters_supports_in_operator() -> None:
    hybrid = HybridSearch()

    metadata = {
        "source_type": "slide",
        "collection": "computer_network",
    }
    filters = {
        "source_type": {"$in": ["slide", "textbook"]},
    }

    assert hybrid._matches_filters(metadata, filters) is True


def test_matches_filters_rejects_non_member_for_in_operator() -> None:
    hybrid = HybridSearch()

    metadata = {
        "source_type": "question_bank",
        "collection": "computer_network",
    }
    filters = {
        "source_type": {"$in": ["slide", "textbook"]},
    }

    assert hybrid._matches_filters(metadata, filters) is False
