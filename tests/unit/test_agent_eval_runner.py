"""Tests for the AgentEvalRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.types import StreamEvent, StreamEventType


class _FakeAgent:
    def __init__(self, events):
        self._events = events

    async def chat(self, message, user_id, conversation_id):
        for event in self._events:
            yield event


@pytest.mark.asyncio
async def test_agent_eval_runner_reconstructs_case_outputs(tmp_path: Path) -> None:
    from src.observability.evaluation.agent_eval_runner import AgentEvalRunner

    test_set_path = tmp_path / "agent_test_set.json"
    test_set_path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "id": "case_1",
                        "message": "Explain TCP",
                        "expected_tools": ["knowledge_query"],
                        "forbidden_tools": ["document_ingest"],
                        "expected_answer_substrings": ["TCP"],
                        "forbidden_answer_substrings": ["I do not know"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    trace_service = MagicMock()
    trace_service.get_trace.return_value = {
        "trace_id": "agent-trace-1",
        "stages": [
            {"stage": "llm_iteration"},
            {"stage": "tool_execution"},
            {"stage": "llm_iteration"},
        ],
    }

    fake_agent = _FakeAgent(
        [
            StreamEvent(type=StreamEventType.TOOL_START, tool_name="knowledge_query"),
            StreamEvent(
                type=StreamEventType.TOOL_RESULT,
                tool_name="knowledge_query",
                content="backend unavailable",
                metadata={
                    "success": False,
                    "error_type": "timeout",
                    "retryable": True,
                    "tool_name": "knowledge_query",
                },
            ),
            StreamEvent(type=StreamEventType.TEXT_DELTA, content="TCP"),
            StreamEvent(type=StreamEventType.TEXT_DELTA, content=" answer"),
            StreamEvent(
                type=StreamEventType.DONE,
                metadata={"trace_id": "agent-trace-1"},
            ),
        ]
    )

    runner = AgentEvalRunner(agent=fake_agent, trace_service=trace_service)
    report = await runner.run_async(test_set_path)

    assert len(report.case_results) == 1
    case = report.case_results[0]
    assert case.actual_tool_chain == ["knowledge_query"]
    assert case.final_answer == "TCP answer"
    assert case.trace_id == "agent-trace-1"
    assert case.iterations == 2
    assert case.tool_errors == [
        {
            "tool_name": "knowledge_query",
            "error_type": "timeout",
            "retryable": True,
            "error": "backend unavailable",
        }
    ]
    assert case.metrics["success"] == 1.0
    assert case.metrics["expected_tool_recall"] == 1.0
    assert report.aggregate_metrics["success_rate"] == 1.0
    assert report.aggregate_metrics["avg_iterations"] == 2.0


def test_agent_eval_report_to_dict_contains_agent_fields() -> None:
    from src.observability.evaluation.agent_eval_runner import AgentEvalReport

    report = AgentEvalReport()
    payload = report.to_dict()

    assert payload["evaluator_name"] == "agent_eval"
    assert payload["case_count"] == 0
