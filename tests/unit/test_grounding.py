from __future__ import annotations

from src.agent.grounding import (
    GroundingEvaluator,
    build_evidence_bundle,
    build_grounding_context,
)


def test_build_evidence_bundle_preserves_query_trace_ids() -> None:
    bundle = build_evidence_bundle(
        "knowledge_query",
        {
            "grounding_capable": True,
            "citations": [
                {
                    "index": 1,
                    "source": "network.md",
                    "text_snippet": "TCP 使用三次握手建立连接。",
                }
            ],
            "query_trace_id": "q-1",
        },
    )

    assert bundle is not None
    assert bundle.has_evidence is True
    assert bundle.query_trace_ids == ["q-1"]
    assert bundle.source_count == 1


def test_grounding_evaluator_scores_supported_answer_higher() -> None:
    bundle = build_evidence_bundle(
        "knowledge_query",
        {
            "grounding_capable": True,
            "citations": [
                {
                    "index": 1,
                    "source": "network.md",
                    "text_snippet": "TCP 使用 SYN、SYN-ACK、ACK 完成三次握手。",
                }
            ],
            "evidence_summary": "[1] `network.md`: TCP 使用 SYN、SYN-ACK、ACK 完成三次握手。",
            "source_count": 1,
        },
    )
    evaluator = GroundingEvaluator(threshold=0.4)

    assessment = evaluator.assess("TCP 通过 SYN、SYN-ACK、ACK 完成三次握手 [1]。", bundle)

    assert assessment.low_evidence is False
    assert assessment.citation_count == 1
    assert assessment.score >= 0.4


def test_grounding_evaluator_marks_missing_evidence_as_low_confidence() -> None:
    evaluator = GroundingEvaluator(threshold=0.4)

    assessment = evaluator.assess("我来补充一些课件里没有的结论。", None)

    assert assessment.low_evidence is True
    assert assessment.has_evidence is False
    assert assessment.score == 0.0


def test_grounding_evaluator_accepts_short_fact_answer_with_valid_citation() -> None:
    bundle = build_evidence_bundle(
        "knowledge_query",
        {
            "grounding_capable": True,
            "citations": [
                {
                    "index": 1,
                    "source": "blogger_intro.pdf",
                    "text_snippet": "这本笔记售价 199 元，适合新手自学。",
                }
            ],
            "evidence_summary": "[1] `blogger_intro.pdf`: 这本笔记售价 199 元，适合新手自学。",
            "source_count": 1,
        },
    )
    evaluator = GroundingEvaluator(threshold=0.4)

    assessment = evaluator.assess("价格是 199 元。[1]", bundle)

    assert assessment.low_evidence is False
    assert assessment.citation_count == 1
    assert assessment.score >= 0.85


def test_grounding_evaluator_accepts_factful_segment_in_longer_answer() -> None:
    bundle = build_evidence_bundle(
        "knowledge_query",
        {
            "grounding_capable": True,
            "citations": [
                {
                    "index": 1,
                    "source": "blogger_intro.pdf",
                    "text_snippet": "笔记目前在小红书链接售卖，价格199元。",
                }
            ],
            "evidence_summary": "[1] `blogger_intro.pdf`: 笔记目前在小红书链接售卖，价格199元。",
            "source_count": 1,
        },
    )
    evaluator = GroundingEvaluator(threshold=0.4)

    assessment = evaluator.assess(
        "博主的笔记售价为199元。[1] 目前可以通过小红书链接购买，但更详细的售卖细节暂无更多证据。",
        bundle,
    )

    assert assessment.low_evidence is False
    assert assessment.citation_count == 1
    assert assessment.score >= 0.8


def test_grounding_evaluator_uses_full_evidence_text_when_snippet_is_truncated() -> None:
    bundle = build_evidence_bundle(
        "knowledge_query",
        {
            "grounding_capable": True,
            "citations": [
                {
                    "index": 1,
                    "source": "blogger_intro.pdf",
                    "text_snippet": "博主介绍与笔记说明……",
                }
            ],
            "evidence_texts": [
                "博主介绍与笔记说明。笔记目前在小红书链接售卖，价格199元。"
            ],
            "source_count": 1,
        },
    )
    evaluator = GroundingEvaluator(threshold=0.4)

    assessment = evaluator.assess("博主的笔记售价为199元。[1]", bundle)

    assert assessment.low_evidence is False
    assert assessment.score >= 0.8


def test_build_grounding_context_includes_conservative_instruction_when_no_evidence() -> None:
    context = build_grounding_context(None, course_task=True)
    assert "不要编造课程知识" in context
