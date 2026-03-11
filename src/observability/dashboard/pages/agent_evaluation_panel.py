"""Agent Evaluation dashboard page."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - optional UI dependency
    class _StreamlitStub:
        def cache_resource(self, *args, **kwargs):
            def _decorator(func):
                return func

            return _decorator

        def __getattr__(self, name: str):
            def _missing(*args, **kwargs):
                raise ModuleNotFoundError(
                    "streamlit is required to render the Agent Evaluation page"
                )

            return _missing

    st = _StreamlitStub()

logger = logging.getLogger(__name__)

DEFAULT_AGENT_GOLDEN_SET = Path("tests/fixtures/agent_golden_test_set.json")
AGENT_EVAL_HISTORY_PATH = Path("logs/agent_eval_history.jsonl")


@st.cache_resource(show_spinner=False)
def _load_agent_runtime() -> Any:
    from src.server.app import create_app

    app = create_app()
    agent = getattr(app.state, "agent", None)
    if agent is None:
        raise RuntimeError("FastAPI app does not expose app.state.agent")
    return agent


def render() -> None:
    """Render the Agent Evaluation page."""
    st.header("🧪 Agent Evaluation")
    st.markdown(
        "Run deterministic golden-set evaluation against the full agent loop, "
        "including tool selection, final answer content, latency, and linked trace IDs."
    )

    st.subheader("⚙️ Configuration")
    golden_path_str = st.text_input(
        "Agent Golden Test Set Path",
        value=str(DEFAULT_AGENT_GOLDEN_SET),
        key="agent_eval_golden_path",
        help="Path to the agent_golden_test_set.json file.",
    )
    golden_path = Path(golden_path_str)

    if not golden_path.exists():
        st.warning(f"⚠️ Agent golden test set not found: `{golden_path}`")

    run_clicked = st.button(
        "▶️ Run Agent Evaluation",
        type="primary",
        key="agent_eval_run_btn",
        disabled=not golden_path.exists(),
    )
    if run_clicked:
        _run_agent_evaluation(golden_path)

    st.divider()
    _render_history()


def _run_agent_evaluation(golden_path: Path) -> None:
    with st.spinner("Loading agent runtime and executing golden set…"):
        try:
            report = _execute_agent_evaluation(golden_path)
        except Exception as exc:
            st.error(f"❌ Agent evaluation failed: {exc}")
            logger.exception("Agent evaluation failed")
            return

    st.success("✅ Agent evaluation complete!")
    _render_aggregate_metrics(report)
    _render_case_details(report)
    _save_to_history(report)


def _execute_agent_evaluation(golden_path: Path) -> Dict[str, Any]:
    from src.observability.evaluation.agent_eval_runner import AgentEvalRunner

    agent = _load_agent_runtime()
    runner = AgentEvalRunner(agent=agent)
    return runner.run(golden_path).to_dict()


def _render_aggregate_metrics(report: Dict[str, Any]) -> None:
    st.subheader("📊 Aggregate Metrics")
    aggregate_metrics = report.get("aggregate_metrics", {})
    if not aggregate_metrics:
        st.info("No aggregate metrics available.")
        return

    cols = st.columns(min(len(aggregate_metrics), 4))
    for index, (name, value) in enumerate(sorted(aggregate_metrics.items())):
        with cols[index % len(cols)]:
            if name.endswith("_ms"):
                display = f"{value:.1f}"
            else:
                display = f"{value:.4f}"
            st.metric(name.replace("_", " ").title(), display)

    st.caption(
        f"Runner: **{report.get('evaluator_name', 'agent_eval')}** · "
        f"Cases: **{report.get('case_count', report.get('query_count', 0))}** · "
        f"Total time: **{report.get('total_elapsed_ms', 0):.0f} ms**"
    )


def _render_case_details(report: Dict[str, Any]) -> None:
    st.subheader("🔍 Per-Case Details")
    case_results = report.get("case_results", [])
    if not case_results:
        st.info("No case-level results available.")
        return

    for index, result in enumerate(case_results, start=1):
        metrics = result.get("metrics", {})
        summary = " · ".join(
            f"{key}: {value:.3f}" for key, value in sorted(metrics.items())
        ) or "no metrics"
        title = (
            f"Case {index}: {result.get('id', 'unknown')} · "
            f"{result.get('elapsed_ms', 0):.0f} ms · {summary}"
        )
        with st.expander(title, expanded=(index == 1)):
            st.markdown(f"**Message:** {result.get('message', '—')}")
            trace_id = result.get("trace_id", "")
            if trace_id:
                st.markdown(f"**Trace ID:** `{trace_id}`")
            if result.get("error"):
                st.error(result["error"])

            compare_cols = st.columns(2)
            with compare_cols[0]:
                st.markdown("**Expected**")
                st.json(
                    {
                        "expected_tools": result.get("expected_tools", []),
                        "forbidden_tools": result.get("forbidden_tools", []),
                        "expected_answer_substrings": result.get(
                            "expected_answer_substrings", []
                        ),
                        "forbidden_answer_substrings": result.get(
                            "forbidden_answer_substrings", []
                        ),
                        "expected_planner_intent": result.get("expected_planner_intent", ""),
                        "expected_control_mode": result.get("expected_control_mode", ""),
                        "require_citations": result.get("require_citations", False),
                        "expected_grounding_action": result.get("expected_grounding_action", ""),
                        "expected_generation_mode": result.get("expected_generation_mode", ""),
                    }
                )
            with compare_cols[1]:
                st.markdown("**Actual**")
                st.json(
                    {
                        "actual_tool_chain": result.get("actual_tool_chain", []),
                        "iterations": result.get("iterations", 0),
                        "actual_planner_intent": result.get("actual_planner_intent", ""),
                        "actual_control_mode": result.get("actual_control_mode", ""),
                        "grounding_score": result.get("grounding_score", 0.0),
                        "grounding_policy_action": result.get("grounding_policy_action", ""),
                        "generation_mode": result.get("generation_mode", ""),
                        "citation_count": len(result.get("citations", [])),
                        "final_answer": result.get("final_answer", ""),
                    }
                )

            if result.get("notes"):
                st.caption(f"Notes: {result['notes']}")

            tool_errors = result.get("tool_errors", [])
            if tool_errors:
                st.markdown("**Tool Errors**")
                st.json(tool_errors)

            citations = result.get("citations", [])
            if citations:
                st.markdown("**Citations**")
                st.json(citations)

            if metrics:
                metric_cols = st.columns(min(len(metrics), 4))
                for metric_index, (name, value) in enumerate(sorted(metrics.items())):
                    with metric_cols[metric_index % len(metric_cols)]:
                        if name.endswith("_ms"):
                            display = f"{value:.1f}"
                        else:
                            display = f"{value:.4f}"
                        st.metric(name, display)


def _render_history() -> None:
    st.subheader("📈 Agent Evaluation History")
    history = _load_history()
    if not history:
        st.info("No agent evaluation history yet. Run an evaluation to start tracking!")
        return

    rows = []
    for entry in history[-10:]:
        rows.append(
            {
                "Timestamp": entry.get("timestamp", "—"),
                "Runner": entry.get("evaluator_name", "agent_eval"),
                "Cases": entry.get("case_count", entry.get("query_count", 0)),
                "Time (ms)": round(entry.get("total_elapsed_ms", 0)),
                **{
                    key: round(value, 4)
                    for key, value in entry.get("aggregate_metrics", {}).items()
                },
            }
        )
    st.dataframe(rows, use_container_width=True)


def _save_to_history(report: Dict[str, Any]) -> None:
    try:
        AGENT_EVAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **report,
        }
        with AGENT_EVAL_HISTORY_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to save agent evaluation history: %s", exc)


def _load_history() -> List[Dict[str, Any]]:
    if not AGENT_EVAL_HISTORY_PATH.exists():
        return []

    entries: List[Dict[str, Any]] = []
    try:
        with AGENT_EVAL_HISTORY_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.warning("Failed to load agent evaluation history: %s", exc)
    return entries
