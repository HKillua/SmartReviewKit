"""Agent evaluation runner for end-to-end agent runtime regression checks."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.types import StreamEventType
from src.observability.dashboard.services.trace_service import TraceService


@dataclass
class AgentGoldenTestCase:
    """A single agent evaluation case."""

    id: str
    message: str
    user_id: str = "eval_user"
    conversation_id: Optional[str] = None
    expected_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    expected_answer_substrings: List[str] = field(default_factory=list)
    forbidden_answer_substrings: List[str] = field(default_factory=list)
    expected_planner_intent: str = ""
    expected_control_mode: str = ""
    require_citations: bool = False
    expected_grounding_action: str = ""
    expected_generation_mode: str = ""
    expected_evaluation_mode: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentGoldenTestCase":
        return cls(
            id=str(data["id"]),
            message=str(data["message"]),
            user_id=str(data.get("user_id", "eval_user")),
            conversation_id=data.get("conversation_id"),
            expected_tools=[str(v) for v in data.get("expected_tools", [])],
            forbidden_tools=[str(v) for v in data.get("forbidden_tools", [])],
            expected_answer_substrings=[
                str(v) for v in data.get("expected_answer_substrings", [])
            ],
            forbidden_answer_substrings=[
                str(v) for v in data.get("forbidden_answer_substrings", [])
            ],
            expected_planner_intent=str(data.get("expected_planner_intent", "")),
            expected_control_mode=str(data.get("expected_control_mode", "")),
            require_citations=bool(data.get("require_citations", False)),
            expected_grounding_action=str(data.get("expected_grounding_action", "")),
            expected_generation_mode=str(data.get("expected_generation_mode", "")),
            expected_evaluation_mode=str(data.get("expected_evaluation_mode", "")),
            notes=str(data.get("notes", "")),
        )


@dataclass
class AgentEvalCaseResult:
    """Evaluation result for one agent case."""

    id: str
    message: str
    expected_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    expected_answer_substrings: List[str] = field(default_factory=list)
    forbidden_answer_substrings: List[str] = field(default_factory=list)
    expected_planner_intent: str = ""
    expected_control_mode: str = ""
    require_citations: bool = False
    expected_grounding_action: str = ""
    expected_generation_mode: str = ""
    expected_evaluation_mode: str = ""
    notes: str = ""
    actual_tool_chain: List[str] = field(default_factory=list)
    final_answer: str = ""
    trace_id: str = ""
    actual_planner_intent: str = ""
    actual_control_mode: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    grounding_score: float = 0.0
    grounding_policy_action: str = ""
    generation_mode: str = ""
    actual_evaluation_mode: str = ""
    has_evidence: bool = False
    error: str = ""
    tool_errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    iterations: int = 0


@dataclass
class AgentEvalReport:
    """Aggregated report across multiple agent evaluation cases."""

    case_results: List[AgentEvalCaseResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0
    test_set_path: str = ""
    evaluator_name: str = "agent_eval"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "test_set_path": self.test_set_path,
            "query_count": len(self.case_results),
            "case_count": len(self.case_results),
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "aggregate_metrics": {
                key: round(value, 4) for key, value in self.aggregate_metrics.items()
            },
            "case_results": [
                {
                    "id": result.id,
                    "message": result.message,
                    "expected_tools": list(result.expected_tools),
                    "forbidden_tools": list(result.forbidden_tools),
                    "expected_answer_substrings": list(result.expected_answer_substrings),
                    "forbidden_answer_substrings": list(result.forbidden_answer_substrings),
                    "expected_planner_intent": result.expected_planner_intent,
                    "expected_control_mode": result.expected_control_mode,
                    "require_citations": result.require_citations,
                    "expected_grounding_action": result.expected_grounding_action,
                    "expected_generation_mode": result.expected_generation_mode,
                    "expected_evaluation_mode": result.expected_evaluation_mode,
                    "notes": result.notes,
                    "actual_tool_chain": list(result.actual_tool_chain),
                    "final_answer": result.final_answer,
                    "trace_id": result.trace_id,
                    "actual_planner_intent": result.actual_planner_intent,
                    "actual_control_mode": result.actual_control_mode,
                    "citations": list(result.citations),
                    "grounding_score": round(result.grounding_score, 4),
                    "grounding_policy_action": result.grounding_policy_action,
                    "generation_mode": result.generation_mode,
                    "actual_evaluation_mode": result.actual_evaluation_mode,
                    "has_evidence": result.has_evidence,
                    "error": result.error,
                    "tool_errors": list(result.tool_errors),
                    "metrics": {
                        key: round(value, 4) for key, value in result.metrics.items()
                    },
                    "elapsed_ms": round(result.elapsed_ms, 1),
                    "iterations": result.iterations,
                }
                for result in self.case_results
            ],
        }


def load_agent_test_set(path: str | Path) -> List[AgentGoldenTestCase]:
    """Load an agent golden test set from JSON."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Agent golden test set not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if "test_cases" not in data:
        raise ValueError("Invalid agent golden test set format: missing 'test_cases'")

    return [AgentGoldenTestCase.from_dict(case) for case in data["test_cases"]]


class AgentEvalRunner:
    """Runs deterministic end-to-end evaluation against Agent.chat()."""

    def __init__(self, agent: Any, trace_service: Optional[TraceService] = None) -> None:
        self.agent = agent
        self.trace_service = trace_service or TraceService()

    def run(self, test_set_path: str | Path) -> AgentEvalReport:
        return asyncio.run(self.run_async(test_set_path))

    async def run_async(self, test_set_path: str | Path) -> AgentEvalReport:
        cases = load_agent_test_set(test_set_path)
        if not cases:
            raise ValueError("Agent golden test set is empty.")

        report = AgentEvalReport(test_set_path=str(test_set_path))
        started = time.monotonic()

        for case in cases:
            report.case_results.append(await self._run_case(case))

        report.total_elapsed_ms = (time.monotonic() - started) * 1000.0
        report.aggregate_metrics = self._aggregate_metrics(report.case_results)
        return report

    async def _run_case(self, case: AgentGoldenTestCase) -> AgentEvalCaseResult:
        started = time.monotonic()
        text_parts: List[str] = []
        tool_chain: List[str] = []
        trace_id = ""
        done_metadata: Dict[str, Any] = {}
        error = ""
        tool_errors: List[Dict[str, Any]] = []

        async for event in self.agent.chat(
            case.message,
            case.user_id,
            case.conversation_id,
        ):
            if event.type == StreamEventType.TEXT_DELTA and event.content:
                text_parts.append(event.content)
            elif event.type == StreamEventType.TOOL_START and event.tool_name:
                tool_chain.append(event.tool_name)
            elif event.type == StreamEventType.TOOL_RESULT and event.tool_name:
                if not bool(event.metadata.get("success", False)):
                    tool_errors.append(
                        {
                            "tool_name": event.tool_name,
                            "error_type": str(event.metadata.get("error_type", "")),
                            "retryable": bool(event.metadata.get("retryable", False)),
                            "error": event.content or "",
                        }
                    )
            elif event.type == StreamEventType.ERROR and event.content:
                error = event.content
            elif event.type == StreamEventType.DONE:
                done_metadata = dict(event.metadata)
                trace_id = str(done_metadata.get("trace_id", ""))

        final_answer = "".join(text_parts).strip()
        iterations = self._lookup_iterations(trace_id)
        planner_intent, control_mode = self._lookup_planner(trace_id)
        citations = [
            citation
            for citation in done_metadata.get("citations", [])
            if isinstance(citation, dict)
        ]
        grounding_score = float(done_metadata.get("grounding_score", 0.0) or 0.0)
        grounding_policy_action = str(done_metadata.get("grounding_policy_action", "") or "")
        generation_mode = str(done_metadata.get("generation_mode", "") or "")
        evaluation_mode = str(done_metadata.get("evaluation_mode", "") or "")
        has_evidence = bool(done_metadata.get("has_evidence", False))
        elapsed_ms = (time.monotonic() - started) * 1000.0
        metrics = self._score_case(
            case,
            tool_chain,
            final_answer,
            error,
            iterations,
            elapsed_ms,
            planner_intent,
            control_mode,
            citations,
            grounding_score,
            grounding_policy_action,
            generation_mode,
            evaluation_mode,
        )

        return AgentEvalCaseResult(
            id=case.id,
            message=case.message,
            expected_tools=list(case.expected_tools),
            forbidden_tools=list(case.forbidden_tools),
            expected_answer_substrings=list(case.expected_answer_substrings),
            forbidden_answer_substrings=list(case.forbidden_answer_substrings),
            expected_planner_intent=case.expected_planner_intent,
            expected_control_mode=case.expected_control_mode,
            require_citations=case.require_citations,
            expected_grounding_action=case.expected_grounding_action,
            expected_generation_mode=case.expected_generation_mode,
            expected_evaluation_mode=case.expected_evaluation_mode,
            notes=case.notes,
            actual_tool_chain=tool_chain,
            final_answer=final_answer,
            trace_id=trace_id,
            actual_planner_intent=planner_intent,
            actual_control_mode=control_mode,
            citations=citations,
            grounding_score=grounding_score,
            grounding_policy_action=grounding_policy_action,
            generation_mode=generation_mode,
            actual_evaluation_mode=evaluation_mode,
            has_evidence=has_evidence,
            error=error,
            tool_errors=tool_errors,
            metrics=metrics,
            elapsed_ms=elapsed_ms,
            iterations=iterations,
        )

    def _lookup_iterations(self, trace_id: str) -> int:
        if not trace_id:
            return 0
        trace = self.trace_service.get_trace(trace_id)
        if not trace:
            return 0
        stages = trace.get("stages", [])
        return sum(1 for stage in stages if stage.get("stage") == "llm_iteration")

    def _lookup_planner(self, trace_id: str) -> tuple[str, str]:
        if not trace_id:
            return "", ""
        trace = self.trace_service.get_trace(trace_id)
        if not trace:
            return "", ""
        metadata = trace.get("metadata", {})
        task_intent = str(metadata.get("planner_task_intent", ""))
        control_mode = str(
            metadata.get(
                "planner_final_control_mode",
                metadata.get("planner_control_mode", ""),
            )
        )
        if task_intent and control_mode:
            return task_intent, control_mode
        for stage in trace.get("stages", []):
            if stage.get("stage") == "planner_decision":
                data = stage.get("data", {})
                return str(data.get("task_intent", "")), str(data.get("control_mode", ""))
        return "", ""

    def _score_case(
        self,
        case: AgentGoldenTestCase,
        tool_chain: List[str],
        final_answer: str,
        error: str,
        iterations: int,
        elapsed_ms: float,
        planner_intent: str,
        control_mode: str,
        citations: List[Dict[str, Any]],
        grounding_score: float,
        grounding_policy_action: str,
        generation_mode: str,
        evaluation_mode: str,
    ) -> Dict[str, float]:
        expected_hit_count = sum(1 for tool in case.expected_tools if tool in tool_chain)
        forbidden_tool_violations = sum(
            1 for tool in case.forbidden_tools if tool in tool_chain
        )
        expected_answer_hits = sum(
            1 for snippet in case.expected_answer_substrings if snippet in final_answer
        )
        forbidden_answer_hits = sum(
            1 for snippet in case.forbidden_answer_substrings if snippet in final_answer
        )
        planner_intent_hit = (
            1.0
            if not case.expected_planner_intent
            else float(case.expected_planner_intent == planner_intent)
        )
        planner_control_mode_hit = (
            1.0
            if not case.expected_control_mode
            else float(case.expected_control_mode == control_mode)
        )
        citation_presence = (
            1.0
            if not case.require_citations
            else float(bool(citations))
        )
        grounding_action_hit = (
            1.0
            if not case.expected_grounding_action
            else float(case.expected_grounding_action == grounding_policy_action)
        )
        generation_mode_hit = (
            1.0
            if not case.expected_generation_mode
            else float(case.expected_generation_mode == generation_mode)
        )
        evaluation_mode_hit = (
            1.0
            if not case.expected_evaluation_mode
            else float(case.expected_evaluation_mode == evaluation_mode)
        )
        quiz_grounded_success = 1.0
        quiz_insufficient_evidence = 0.0
        quiz_evaluator_evidence = 1.0
        if case.expected_tools == ["quiz_generator"] or case.expected_planner_intent == "quiz_generator":
            quiz_grounded_success = float(generation_mode in {"question_bank", "rag_backed"})
            quiz_insufficient_evidence = float(generation_mode == "insufficient_evidence")
        if case.expected_tools == ["quiz_evaluator"] or case.expected_planner_intent == "quiz_evaluator":
            quiz_evaluator_evidence = float(evaluation_mode == "evidence_enhanced")

        return {
            "success": 0.0 if error else 1.0,
            "expected_tool_recall": (
                expected_hit_count / len(case.expected_tools)
                if case.expected_tools
                else 1.0
            ),
            "forbidden_tool_pass_rate": 0.0 if forbidden_tool_violations else 1.0,
            "answer_keyword_recall": (
                expected_answer_hits / len(case.expected_answer_substrings)
                if case.expected_answer_substrings
                else 1.0
            ),
            "answer_forbidden_pass_rate": 0.0 if forbidden_answer_hits else 1.0,
            "planner_intent_hit_rate": planner_intent_hit,
            "planner_control_mode_hit_rate": planner_control_mode_hit,
            "citation_presence_rate": citation_presence,
            "grounding_score": grounding_score,
            "grounding_action_hit_rate": grounding_action_hit,
            "generation_mode_hit_rate": generation_mode_hit,
            "evaluation_mode_hit_rate": evaluation_mode_hit,
            "quiz_grounded_success_rate": quiz_grounded_success,
            "quiz_insufficient_evidence_rate": quiz_insufficient_evidence,
            "quiz_evaluator_evidence_rate": quiz_evaluator_evidence,
            "tool_calls": float(len(tool_chain)),
            "iterations": float(iterations),
            "latency_ms": elapsed_ms,
        }

    def _aggregate_metrics(self, results: List[AgentEvalCaseResult]) -> Dict[str, float]:
        if not results:
            return {}

        success_rate = sum(result.metrics.get("success", 0.0) for result in results) / len(results)
        expected_tool_recall = sum(
            result.metrics.get("expected_tool_recall", 0.0) for result in results
        ) / len(results)
        forbidden_tool_pass_rate = sum(
            result.metrics.get("forbidden_tool_pass_rate", 0.0) for result in results
        ) / len(results)
        answer_keyword_recall = sum(
            result.metrics.get("answer_keyword_recall", 0.0) for result in results
        ) / len(results)
        answer_forbidden_pass_rate = sum(
            result.metrics.get("answer_forbidden_pass_rate", 0.0) for result in results
        ) / len(results)
        planner_intent_hit_rate = sum(
            result.metrics.get("planner_intent_hit_rate", 0.0) for result in results
        ) / len(results)
        planner_control_mode_hit_rate = sum(
            result.metrics.get("planner_control_mode_hit_rate", 0.0) for result in results
        ) / len(results)
        citation_presence_rate = sum(
            result.metrics.get("citation_presence_rate", 0.0) for result in results
        ) / len(results)
        avg_grounding_score = sum(
            result.metrics.get("grounding_score", 0.0) for result in results
        ) / len(results)
        low_evidence_action_rate = sum(
            1.0
            for result in results
            if result.grounding_policy_action == "low_evidence_warning"
        ) / len(results)
        conservative_rewrite_rate = sum(
            1.0
            for result in results
            if result.grounding_policy_action == "conservative_rewrite"
        ) / len(results)
        quiz_grounded_success_rate = sum(
            result.metrics.get("quiz_grounded_success_rate", 0.0) for result in results
        ) / len(results)
        quiz_insufficient_evidence_rate = sum(
            result.metrics.get("quiz_insufficient_evidence_rate", 0.0) for result in results
        ) / len(results)
        evaluation_mode_hit_rate = sum(
            result.metrics.get("evaluation_mode_hit_rate", 0.0) for result in results
        ) / len(results)
        quiz_evaluator_evidence_rate = sum(
            result.metrics.get("quiz_evaluator_evidence_rate", 0.0) for result in results
        ) / len(results)
        avg_tool_calls = sum(len(result.actual_tool_chain) for result in results) / len(results)
        avg_iterations = sum(result.iterations for result in results) / len(results)
        avg_latency_ms = sum(result.elapsed_ms for result in results) / len(results)

        return {
            "success_rate": success_rate,
            "expected_tool_recall": expected_tool_recall,
            "forbidden_tool_pass_rate": forbidden_tool_pass_rate,
            "answer_keyword_recall": answer_keyword_recall,
            "answer_forbidden_pass_rate": answer_forbidden_pass_rate,
            "planner_intent_hit_rate": planner_intent_hit_rate,
            "planner_control_mode_hit_rate": planner_control_mode_hit_rate,
            "citation_presence_rate": citation_presence_rate,
            "avg_grounding_score": avg_grounding_score,
            "low_evidence_action_rate": low_evidence_action_rate,
            "conservative_rewrite_rate": conservative_rewrite_rate,
            "quiz_grounded_success_rate": quiz_grounded_success_rate,
            "quiz_insufficient_evidence_rate": quiz_insufficient_evidence_rate,
            "evaluation_mode_hit_rate": evaluation_mode_hit_rate,
            "quiz_evaluator_evidence_rate": quiz_evaluator_evidence_rate,
            "avg_tool_calls": avg_tool_calls,
            "avg_iterations": avg_iterations,
            "avg_latency_ms": avg_latency_ms,
        }
