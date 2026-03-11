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


def test_build_grounding_context_includes_conservative_instruction_when_no_evidence() -> None:
    context = build_grounding_context(None, course_task=True)
    assert "不要编造课程知识" in context
