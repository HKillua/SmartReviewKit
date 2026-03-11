"""Tests for Agent Traces and Agent Evaluation dashboard helpers."""

from __future__ import annotations

from pathlib import Path


class TestAgentEvaluationPanelHelpers:
    def test_save_and_load_history(self, tmp_path: Path) -> None:
        from src.observability.dashboard.pages import agent_evaluation_panel as aep

        original = aep.AGENT_EVAL_HISTORY_PATH
        aep.AGENT_EVAL_HISTORY_PATH = tmp_path / "agent_eval_history.jsonl"

        try:
            report = {
                "evaluator_name": "agent_eval",
                "case_count": 2,
                "total_elapsed_ms": 111.5,
                "aggregate_metrics": {"success_rate": 1.0},
            }
            aep._save_to_history(report)
            history = aep._load_history()

            assert len(history) == 1
            assert history[0]["evaluator_name"] == "agent_eval"
            assert history[0]["aggregate_metrics"]["success_rate"] == 1.0
            assert "timestamp" in history[0]
        finally:
            aep.AGENT_EVAL_HISTORY_PATH = original

    def test_load_history_empty(self, tmp_path: Path) -> None:
        from src.observability.dashboard.pages import agent_evaluation_panel as aep

        original = aep.AGENT_EVAL_HISTORY_PATH
        aep.AGENT_EVAL_HISTORY_PATH = tmp_path / "missing.jsonl"
        try:
            assert aep._load_history() == []
        finally:
            aep.AGENT_EVAL_HISTORY_PATH = original


class TestAgentTracesHelpers:
    def test_extract_tool_chain_and_query_trace_ids(self) -> None:
        from src.observability.dashboard.pages.agent_traces import (
            _extract_planner_info,
            _extract_linked_query_trace_ids,
            _extract_tool_chain,
        )

        trace = {
            "metadata": {
                "status": "success",
                "planner_task_intent": "review_summary",
                "planner_control_mode": "force_tool",
            },
            "stages": [
                {
                    "stage": "tool_execution",
                    "data": {"tool_name": "knowledge_query", "query_trace_id": "q1"},
                },
                {
                    "stage": "tool_execution",
                    "data": {"tool_name": "review_summary", "query_trace_id": ""},
                },
            ],
        }

        assert _extract_tool_chain(trace) == ["knowledge_query", "review_summary"]
        assert _extract_linked_query_trace_ids(trace) == ["q1"]
        assert _extract_planner_info(trace)["task_intent"] == "review_summary"

    def test_filter_agent_traces_by_keyword_tool_and_status(self) -> None:
        from src.observability.dashboard.pages.agent_traces import _filter_agent_traces

        traces = [
            {
                "metadata": {"status": "success", "message_preview": "TCP handshake"},
                "stages": [
                    {"stage": "tool_execution", "data": {"tool_name": "knowledge_query"}}
                ],
            },
            {
                "metadata": {"status": "error", "message_preview": "Upload PDF"},
                "stages": [
                    {"stage": "tool_execution", "data": {"tool_name": "document_ingest"}}
                ],
            },
        ]

        filtered = _filter_agent_traces(
            traces,
            keyword="tcp",
            tool_name="knowledge_query",
            status="success",
        )
        assert len(filtered) == 1
        assert filtered[0]["metadata"]["message_preview"] == "TCP handshake"

    def test_agent_page_modules_import(self) -> None:
        from src.observability.dashboard.pages import (
            agent_evaluation_panel,
            agent_traces,
        )

        assert hasattr(agent_traces, "render")
        assert hasattr(agent_evaluation_panel, "render")
