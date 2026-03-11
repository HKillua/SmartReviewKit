"""Agent Traces page for browsing end-to-end agent trajectories."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - optional UI dependency
    class _StreamlitStub:
        def __getattr__(self, name: str):
            def _missing(*args, **kwargs):
                raise ModuleNotFoundError(
                    "streamlit is required to render the Agent Traces page"
                )

            return _missing

    st = _StreamlitStub()

from src.observability.dashboard.services.trace_service import TraceService


def _get_total_elapsed_ms(trace: Dict[str, Any]) -> float | None:
    value = trace.get("total_elapsed_ms", trace.get("elapsed_ms"))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _get_status(trace: Dict[str, Any]) -> str:
    metadata = trace.get("metadata", {})
    status = metadata.get("status", "unknown")
    return str(status or "unknown")


def _get_stage_entries(trace: Dict[str, Any], stage_name: str) -> List[Dict[str, Any]]:
    return [
        stage
        for stage in trace.get("stages", [])
        if stage.get("stage") == stage_name
    ]


def _extract_tool_chain(trace: Dict[str, Any]) -> List[str]:
    return [
        str(stage.get("data", {}).get("tool_name", ""))
        for stage in _get_stage_entries(trace, "tool_execution")
        if stage.get("data", {}).get("tool_name")
    ]


def _extract_linked_query_trace_ids(trace: Dict[str, Any]) -> List[str]:
    seen: set[str] = set()
    query_trace_ids: List[str] = []
    for stage in _get_stage_entries(trace, "tool_execution"):
        query_trace_id = str(stage.get("data", {}).get("query_trace_id", "") or "")
        if query_trace_id and query_trace_id not in seen:
            seen.add(query_trace_id)
            query_trace_ids.append(query_trace_id)
    return query_trace_ids


def _filter_agent_traces(
    traces: List[Dict[str, Any]],
    keyword: str = "",
    tool_name: str = "",
    status: str = "all",
) -> List[Dict[str, Any]]:
    keyword_norm = keyword.strip().lower()
    tool_norm = tool_name.strip().lower()
    status_norm = status.strip().lower()

    filtered: List[Dict[str, Any]] = []
    for trace in traces:
        if keyword_norm:
            haystack = str(trace).lower()
            if keyword_norm not in haystack:
                continue
        if tool_norm:
            tools = [name.lower() for name in _extract_tool_chain(trace)]
            if tool_norm not in tools:
                continue
        if status_norm and status_norm != "all":
            if _get_status(trace).lower() != status_norm:
                continue
        filtered.append(trace)
    return filtered


def _build_stage_waterfall_rows(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    counters: dict[str, int] = {}
    for stage in trace.get("stages", []):
        elapsed_ms = stage.get("elapsed_ms")
        if not isinstance(elapsed_ms, (int, float)):
            continue
        stage_name = str(stage.get("stage", "unknown"))
        counters[stage_name] = counters.get(stage_name, 0) + 1
        label = f"{stage_name} #{counters[stage_name]}"
        rows.append({"Stage": label, "Elapsed (ms)": round(float(elapsed_ms), 2)})
    return rows


def render() -> None:
    """Render the Agent Traces page."""
    st.header("🤖 Agent Traces")

    svc = TraceService()
    traces = svc.list_traces(trace_type="agent")

    if not traces:
        st.info("No agent traces recorded yet. Run an agent chat first!")
        return

    filter_cols = st.columns([2, 1, 1])
    with filter_cols[0]:
        keyword = st.text_input("Search keyword", value="", key="agent_trace_keyword")
    with filter_cols[1]:
        tool_name = st.text_input("Filter tool", value="", key="agent_trace_tool")
    with filter_cols[2]:
        status = st.selectbox(
            "Status",
            options=["all", "success", "error"],
            index=0,
            key="agent_trace_status",
        )

    traces = _filter_agent_traces(
        traces,
        keyword=keyword,
        tool_name=tool_name,
        status=status,
    )

    st.subheader(f"📋 Agent Trace History ({len(traces)})")

    for index, trace in enumerate(traces):
        metadata = trace.get("metadata", {})
        trace_id = str(trace.get("trace_id", "unknown"))
        status_value = _get_status(trace)
        total_ms = _get_total_elapsed_ms(trace)
        total_label = f"{total_ms:.0f} ms" if total_ms is not None else "—"
        started = str(trace.get("started_at", "—"))
        message_preview = str(metadata.get("message_preview", "")) or "—"
        tool_chain = _extract_tool_chain(trace)
        query_trace_ids = _extract_linked_query_trace_ids(trace)

        expander_title = (
            f"🤖 {message_preview[:50]}{'…' if len(message_preview) > 50 else ''} "
            f"· {status_value} · {total_label} · {started[:19]}"
        )

        with st.expander(expander_title, expanded=(index == 0)):
            st.markdown("#### Request Overview")
            overview_cols = st.columns(4)
            with overview_cols[0]:
                st.metric("Status", status_value)
            with overview_cols[1]:
                st.metric("Iterations", len(_get_stage_entries(trace, "llm_iteration")))
            with overview_cols[2]:
                st.metric("Tool Calls", len(tool_chain))
            with overview_cols[3]:
                st.metric("Linked Query Traces", len(query_trace_ids))

            st.caption(
                " · ".join(
                    [
                        f"trace_id=`{trace_id}`",
                        f"request_id=`{metadata.get('request_id', '—')}`",
                        f"conversation_id=`{metadata.get('conversation_id', '—')}`",
                        f"user_id_hash=`{metadata.get('user_id_hash', '—')}`",
                    ]
                )
            )

            if tool_chain:
                st.markdown("**Tool Chain**")
                st.code(" -> ".join(tool_chain), language=None)

            if query_trace_ids:
                st.markdown("**Linked Query Trace IDs**")
                st.code("\n".join(query_trace_ids), language=None)

            st.divider()

            waterfall_rows = _build_stage_waterfall_rows(trace)
            if waterfall_rows:
                st.markdown("#### ⏱️ Stage Timings")
                st.bar_chart(
                    {row["Stage"]: row["Elapsed (ms)"] for row in waterfall_rows},
                    horizontal=True,
                )
                st.table(waterfall_rows)

            st.divider()

            llm_iterations = _get_stage_entries(trace, "llm_iteration")
            st.markdown("#### 🧠 LLM Iterations")
            if llm_iterations:
                for llm_index, stage in enumerate(llm_iterations, start=1):
                    data = stage.get("data", {})
                    tool_names = data.get("tool_names", [])
                    title = (
                        f"Iteration {llm_index} · tools={len(tool_names)} · "
                        f"text_len={data.get('output_text_length', 0)}"
                    )
                    with st.expander(title, expanded=(llm_index == 1)):
                        st.json(data)
            else:
                st.info("No LLM iteration stages recorded.")

            st.divider()

            tool_steps = _get_stage_entries(trace, "tool_execution")
            st.markdown("#### 🛠️ Tool Executions")
            if tool_steps:
                for step_index, stage in enumerate(tool_steps, start=1):
                    data = stage.get("data", {})
                    title = (
                        f"{step_index}. {data.get('tool_name', 'unknown')} "
                        f"· success={data.get('success', False)}"
                    )
                    with st.expander(title, expanded=(step_index == 1)):
                        st.json(data)
            else:
                st.info("No tool executions recorded.")

            st.divider()

            final_responses = _get_stage_entries(trace, "final_response")
            st.markdown("#### 📝 Final Response")
            if final_responses:
                final_data = final_responses[-1].get("data", {})
                st.caption(f"content_length={final_data.get('content_length', 0)}")
                preview = final_data.get("content_preview", "")
                if preview:
                    st.text(preview)
                else:
                    st.info("Final response preview was not captured.")
            else:
                st.info("No final response recorded.")
