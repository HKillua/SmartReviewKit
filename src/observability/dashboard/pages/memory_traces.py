"""Memory Traces page for browsing memory extraction and write-gating traces."""

from __future__ import annotations

from typing import Any, Dict, List

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - optional UI dependency
    class _StreamlitStub:
        def __getattr__(self, name: str):
            def _missing(*args, **kwargs):
                raise ModuleNotFoundError(
                    "streamlit is required to render the Memory Traces page"
                )

            return _missing

    st = _StreamlitStub()

from src.observability.dashboard.services.trace_service import TraceService


def _get_stage_data(trace: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    for stage in trace.get("stages", []):
        if stage.get("stage") == stage_name:
            data = stage.get("data", {})
            return data if isinstance(data, dict) else {}
    return {}


def _extract_memory_overview(trace: Dict[str, Any]) -> Dict[str, Any]:
    metadata = trace.get("metadata", {})
    extraction = _get_stage_data(trace, "memory_extraction")
    gate = _get_stage_data(trace, "memory_quality_gate")
    session = _get_stage_data(trace, "session_memory_write")
    profile = _get_stage_data(trace, "profile_update")
    return {
        "conversation_id": metadata.get("conversation_id", ""),
        "user_id_hash": metadata.get("user_id_hash", ""),
        "configured_mode": metadata.get("configured_extraction_mode", ""),
        "mode": metadata.get("extraction_mode_used", extraction.get("mode", "")),
        "confidence": float(metadata.get("confidence", extraction.get("confidence", 0.0)) or 0.0),
        "signal_counts": extraction.get("signal_counts", {}),
        "preference_conflicts": metadata.get(
            "preference_conflicts",
            extraction.get("preference_conflicts", []),
        ),
        "write_decisions": metadata.get("write_decisions", gate.get("write_decisions", {})),
        "session_status": session.get("status", ""),
        "session_reason": session.get("reason", ""),
        "profile_status": profile.get("status", ""),
        "profile_reason": profile.get("reason", ""),
    }


def _filter_memory_traces(
    traces: List[Dict[str, Any]],
    *,
    keyword: str = "",
    conflict_only: bool = False,
    session_status: str = "all",
    profile_status: str = "all",
) -> List[Dict[str, Any]]:
    keyword_norm = keyword.strip().lower()
    session_norm = session_status.strip().lower()
    profile_norm = profile_status.strip().lower()
    filtered: List[Dict[str, Any]] = []
    for trace in traces:
        overview = _extract_memory_overview(trace)
        if keyword_norm:
            haystack = f"{trace.get('metadata', {})} {trace.get('stages', [])}".lower()
            if keyword_norm not in haystack:
                continue
        if conflict_only and not overview["preference_conflicts"]:
            continue
        if session_norm != "all" and str(overview["session_status"]).lower() != session_norm:
            continue
        if profile_norm != "all" and str(overview["profile_status"]).lower() != profile_norm:
            continue
        filtered.append(trace)
    return filtered


def _compute_summary(traces: List[Dict[str, Any]]) -> Dict[str, float]:
    if not traces:
        return {
            "avg_confidence": 0.0,
            "conflict_rate": 0.0,
            "session_skip_rate": 0.0,
            "preference_skip_rate": 0.0,
        }
    overviews = [_extract_memory_overview(trace) for trace in traces]
    count = len(overviews)
    return {
        "avg_confidence": sum(item["confidence"] for item in overviews) / count,
        "conflict_rate": sum(1.0 for item in overviews if item["preference_conflicts"]) / count,
        "session_skip_rate": sum(
            1.0 for item in overviews if item["session_status"] == "skipped"
        ) / count,
        "preference_skip_rate": sum(
            1.0
            for item in overviews
            if item["profile_status"] == "skipped"
            and item["profile_reason"] not in {"stats_only", "student_profile_disabled"}
        ) / count,
    }


def _build_stage_rows(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stage in trace.get("stages", []):
        rows.append(
            {
                "Stage": str(stage.get("stage", "unknown")),
                "Elapsed (ms)": round(float(stage.get("elapsed_ms", 0.0) or 0.0), 2),
            }
        )
    return rows


def render() -> None:
    st.header("🧠 Memory Traces")

    service = TraceService()
    traces = service.list_traces(trace_type="memory")
    if not traces:
        st.info("No memory traces recorded yet. Run an agent conversation first!")
        return

    filter_cols = st.columns([2, 1, 1, 1])
    with filter_cols[0]:
        keyword = st.text_input("Search keyword", value="", key="memory_trace_keyword")
    with filter_cols[1]:
        conflict_only = st.checkbox("Preference conflicts only", value=False, key="memory_trace_conflicts")
    with filter_cols[2]:
        session_status = st.selectbox(
            "Session write",
            options=["all", "saved", "skipped", "failed"],
            index=0,
            key="memory_trace_session_status",
        )
    with filter_cols[3]:
        profile_status = st.selectbox(
            "Profile update",
            options=["all", "updated", "skipped", "failed"],
            index=0,
            key="memory_trace_profile_status",
        )

    traces = _filter_memory_traces(
        traces,
        keyword=keyword,
        conflict_only=conflict_only,
        session_status=session_status,
        profile_status=profile_status,
    )

    summary = _compute_summary(traces)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Traces", len(traces))
    metric_cols[1].metric("Avg Confidence", f"{summary['avg_confidence']:.2f}")
    metric_cols[2].metric("Conflict Rate", f"{summary['conflict_rate']:.0%}")
    metric_cols[3].metric("Session Skip Rate", f"{summary['session_skip_rate']:.0%}")
    metric_cols[4].metric("Preference Skip Rate", f"{summary['preference_skip_rate']:.0%}")

    st.subheader(f"📋 Memory Trace History ({len(traces)})")
    for index, trace in enumerate(traces):
        overview = _extract_memory_overview(trace)
        trace_id = str(trace.get("trace_id", "unknown"))
        started = str(trace.get("started_at", "—"))
        expander_title = (
            f"🧠 {overview['conversation_id'] or 'unknown'} · "
            f"{overview['mode'] or '—'} · "
            f"confidence={overview['confidence']:.2f} · "
            f"{started[:19]}"
        )
        with st.expander(expander_title, expanded=(index == 0)):
            st.caption(
                " · ".join(
                    [
                        f"trace_id=`{trace_id}`",
                        f"user_id_hash=`{overview['user_id_hash'] or '—'}`",
                        f"configured_mode=`{overview['configured_mode'] or '—'}`",
                    ]
                )
            )
            st.markdown("**Overview**")
            st.json(
                {
                    "mode": overview["mode"],
                    "confidence": round(overview["confidence"], 3),
                    "signal_counts": overview["signal_counts"],
                    "preference_conflicts": overview["preference_conflicts"],
                    "write_decisions": overview["write_decisions"],
                    "session_status": overview["session_status"],
                    "session_reason": overview["session_reason"],
                    "profile_status": overview["profile_status"],
                    "profile_reason": overview["profile_reason"],
                }
            )

            stage_rows = _build_stage_rows(trace)
            if stage_rows:
                st.markdown("**Stage Timings**")
                st.bar_chart({row["Stage"]: row["Elapsed (ms)"] for row in stage_rows}, horizontal=True)

            st.markdown("**Stage Details**")
            for stage in trace.get("stages", []):
                label = str(stage.get("stage", "unknown"))
                with st.expander(label, expanded=False):
                    st.json(stage.get("data", {}))
