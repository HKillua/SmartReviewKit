"""Query Traces page – browse query trace history with stage waterfall.

Layout:
1. Optional keyword search filter
2. Trace list (reverse-chronological, filtered to trace_type=="query")
3. Detail view: stage waterfall + Dense vs Sparse comparison + Rerank delta
4. Per-trace Ragas evaluation button (LLM-as-Judge scoring)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - optional UI dependency
    class _StreamlitStub:
        def __getattr__(self, name: str):
            def _missing(*args, **kwargs):
                raise ModuleNotFoundError(
                    "streamlit is required to render the Query Traces page"
                )

            return _missing

    st = _StreamlitStub()

from src.observability.dashboard.services.trace_service import TraceService

logger = logging.getLogger(__name__)


def _filter_query_traces(
    traces: List[Dict[str, Any]],
    *,
    keyword: str = "",
    source: str = "all",
) -> List[Dict[str, Any]]:
    keyword_norm = keyword.strip().lower()
    source_norm = source.strip().lower()
    filtered: List[Dict[str, Any]] = []
    for trace in traces:
        metadata = trace.get("metadata", {})
        if keyword_norm:
            haystack = f"{metadata} {trace.get('stages', [])}".lower()
            if keyword_norm not in haystack:
                continue
        trace_source = str(metadata.get("source", "") or "").lower()
        if source_norm != "all" and trace_source != source_norm:
            continue
        filtered.append(trace)
    return filtered


def render() -> None:
    """Render the Query Traces page."""
    st.header("🔎 Query Traces")

    svc = TraceService()
    traces = svc.list_traces(trace_type="query")

    if not traces:
        st.info("No query traces recorded yet. Run a query first!")
        return

    # ── Keyword filter ─────────────────────────────────────────────
    filter_cols = st.columns([2, 1])
    with filter_cols[0]:
        keyword = st.text_input(
            "Search by query keyword",
            value="",
            key="qt_keyword",
        )
    sources = sorted(
        {
            str(trace.get("metadata", {}).get("source", "") or "").strip()
            for trace in traces
            if str(trace.get("metadata", {}).get("source", "") or "").strip()
        }
    )
    with filter_cols[1]:
        selected_source = st.selectbox(
            "Source",
            options=["all", *sources],
            index=0,
            key="qt_source_filter",
        )
    traces = _filter_query_traces(
        traces,
        keyword=keyword,
        source=selected_source,
    )

    st.subheader(f"📋 Query History ({len(traces)})")

    for idx, trace in enumerate(traces):
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        total_ms = trace.get("total_elapsed_ms", trace.get("elapsed_ms"))
        total_label = f"{total_ms:.0f} ms" if total_ms is not None else "—"
        meta = trace.get("metadata", {})
        query_text = meta.get("query", "")
        source = meta.get("source", "unknown")

        # ── Expander title: show query text ────────────────────
        query_preview = (
            query_text[:40] + "…" if len(query_text) > 40 else query_text
        ) if query_text else "—"
        expander_title = (
            f"🔍 \"{query_preview}\"  ·  {total_label}  ·  {started[:19]}"
        )

        with st.expander(expander_title, expanded=(idx == 0)):
            # ── 1. Query overview ──────────────────────────────
            st.markdown("#### 💬 Query")
            col_q, col_meta = st.columns([3, 1])
            with col_q:
                st.markdown(f"> {query_text}")
            with col_meta:
                source_emoji = "🤖" if source == "mcp" else "📡"
                st.markdown(f"**Source:** {source_emoji} `{source}`")
                st.markdown(f"**Top-K:** `{meta.get('top_k', '—')}`")
                st.markdown(f"**Collection:** `{meta.get('collection', '—')}`")
                if meta.get("planner_task_intent"):
                    st.markdown(f"**Planner Intent:** `{meta.get('planner_task_intent', '—')}`")
                if meta.get("retrieval_strategy"):
                    st.markdown(f"**Retrieval Strategy:** `{meta.get('retrieval_strategy', '—')}`")
                preferred_sources = meta.get("preferred_sources", [])
                if preferred_sources:
                    st.markdown(f"**Preferred Sources:** `{', '.join(preferred_sources)}`")
                source_weights = meta.get("source_weights", {})
                if source_weights:
                    formatted = ", ".join(
                        f"{key}:{float(value):.2f}"
                        for key, value in source_weights.items()
                    )
                    st.markdown(f"**Source Weights:** `{formatted}`")
                if meta.get("router_match_method"):
                    st.markdown(
                        f"**Router Match:** `{meta.get('router_match_method', '—')}` "
                        f"({meta.get('router_confidence', '—')})"
                    )
                if meta.get("composite_parent_request_id"):
                    st.markdown(
                        f"**Composite Request:** `{meta.get('composite_parent_request_id', '—')}`"
                    )
                if meta.get("composite_subtask_intent"):
                    st.markdown(
                        f"**Composite Subtask:** `#{meta.get('composite_subtask_index', '—')}` "
                        f"`{meta.get('composite_subtask_intent', '—')}`"
                    )

            st.divider()

            # ── 2. Overview metrics ────────────────────────────
            timings = svc.get_stage_timings(trace)
            stages_by_name = {t["stage_name"]: t for t in timings}

            dense_d = (stages_by_name.get("dense_retrieval", {}).get("data") or {})
            sparse_d = (stages_by_name.get("sparse_retrieval", {}).get("data") or {})
            fusion_d = (stages_by_name.get("fusion", {}).get("data") or {})
            rerank_d = (stages_by_name.get("rerank", {}).get("data") or {})
            source_alloc_d = (stages_by_name.get("source_allocation", {}).get("data") or {})
            unit_norm_d = (stages_by_name.get("answer_unit_normalization", {}).get("data") or {})

            dense_count = dense_d.get("result_count", 0)
            sparse_count = sparse_d.get("result_count", 0)
            fusion_count = fusion_d.get("result_count", 0)
            rerank_count = rerank_d.get("output_count", 0)
            normalized_units = sum(
                int(value)
                for value in (meta.get("source_normalized_unit_counts", {}) or {}).values()
                if isinstance(value, (int, float))
            ) or sum(
                int(value)
                for value in unit_norm_d.get("source_normalized_unit_counts", {}).values()
                if isinstance(value, (int, float))
            )

            rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
            with rc1:
                st.metric("Dense Hits", dense_count)
            with rc2:
                st.metric("Sparse Hits", sparse_count)
            with rc3:
                st.metric("Fused", fusion_count or (dense_count + sparse_count))
            with rc4:
                st.metric("After Rerank", rerank_count if rerank_d else "—")
            with rc5:
                st.metric("Answer Units", normalized_units or "—")
            with rc6:
                st.metric("Total Time", total_label)

            st.divider()

            if source_alloc_d or unit_norm_d:
                st.markdown("#### 🧭 Source-Aware Allocation")
                alloc_col1, alloc_col2 = st.columns(2)
                with alloc_col1:
                    if source_alloc_d.get("preferred_sources"):
                        st.markdown(
                            f"**Preferred Sources:** `{', '.join(source_alloc_d.get('preferred_sources', []))}`"
                        )
                    if source_alloc_d.get("source_weights"):
                        st.json(source_alloc_d.get("source_weights", {}), expanded=False)
                    if source_alloc_d.get("source_unit_budgets"):
                        st.markdown("**Unit Budgets**")
                        st.json(source_alloc_d.get("source_unit_budgets", {}), expanded=False)
                with alloc_col2:
                    if source_alloc_d.get("source_raw_overfetch"):
                        st.markdown("**Raw Overfetch**")
                        st.json(source_alloc_d.get("source_raw_overfetch", {}), expanded=False)
                    if meta.get("evidence_source_order"):
                        st.markdown(
                            f"**Evidence Order:** `{', '.join(meta.get('evidence_source_order', []))}`"
                        )
                    if meta.get("normalization_profile"):
                        st.markdown(f"**Normalization:** `{meta.get('normalization_profile', '—')}`")
                if unit_norm_d:
                    stat_cols = st.columns(3)
                    with stat_cols[0]:
                        st.metric("Child Hits", unit_norm_d.get("child_hit_count", 0))
                    with stat_cols[1]:
                        st.metric("Parent Promotions", unit_norm_d.get("parent_promotions", 0))
                    with stat_cols[2]:
                        st.metric("Collapsed Parents", unit_norm_d.get("collapsed_parent_count", 0))
                    if unit_norm_d.get("source_raw_candidate_counts") or meta.get("source_raw_candidate_counts"):
                        st.markdown("**Raw Candidate Counts**")
                        st.json(
                            unit_norm_d.get("source_raw_candidate_counts")
                            or meta.get("source_raw_candidate_counts", {}),
                            expanded=False,
                        )
                    if unit_norm_d.get("source_normalized_unit_counts") or meta.get("source_normalized_unit_counts"):
                        st.markdown("**Normalized Unit Counts**")
                        st.json(
                            unit_norm_d.get("source_normalized_unit_counts")
                            or meta.get("source_normalized_unit_counts", {}),
                            expanded=False,
                        )
                    if meta.get("unit_kind_distribution"):
                        st.markdown("**Unit Kind Distribution**")
                        st.json(meta.get("unit_kind_distribution", {}), expanded=False)

                st.divider()

            # ── 3. Stage timing waterfall ──────────────────────
            main_stage_names = (
                "query_processing",
                "source_allocation",
                "dense_retrieval",
                "sparse_retrieval",
                "fusion",
                "answer_unit_normalization",
                "rerank",
                "answer_unit_rerank",
                "answer_unit_dedup",
            )
            main_timings = [t for t in timings if t["stage_name"] in main_stage_names]
            if main_timings:
                st.markdown("#### ⏱️ Stage Timings")
                chart_data = {t["stage_name"]: t["elapsed_ms"] for t in main_timings}
                st.bar_chart(chart_data, horizontal=True)
                st.table([
                    {
                        "Stage": t["stage_name"],
                        "Elapsed (ms)": round(t["elapsed_ms"], 2),
                    }
                    for t in main_timings
                ])

            st.divider()

            # ── 4. Per-stage detail tabs ───────────────────────
            st.markdown("#### 🔍 Stage Details")

            tab_defs = []
            if "query_processing" in stages_by_name:
                tab_defs.append(("🔤 Query Processing", "query_processing"))
            if "dense_retrieval" in stages_by_name:
                tab_defs.append(("🟦 Dense Retrieval", "dense_retrieval"))
            if "sparse_retrieval" in stages_by_name:
                tab_defs.append(("🟨 Sparse Retrieval", "sparse_retrieval"))
            if "fusion" in stages_by_name:
                tab_defs.append(("🟩 Fusion (RRF)", "fusion"))
            if "source_allocation" in stages_by_name:
                tab_defs.append(("🧭 Source Allocation", "source_allocation"))
            if "answer_unit_normalization" in stages_by_name:
                tab_defs.append(("🧱 Answer Units", "answer_unit_normalization"))
            if "rerank" in stages_by_name:
                tab_defs.append(("🟪 Rerank", "rerank"))
            if "answer_unit_rerank" in stages_by_name:
                tab_defs.append(("🟣 Unit Rerank", "answer_unit_rerank"))
            if "answer_unit_dedup" in stages_by_name:
                tab_defs.append(("🧹 Unit Dedup", "answer_unit_dedup"))

            if tab_defs:
                tabs = st.tabs([label for label, _ in tab_defs])
                for tab, (label, key) in zip(tabs, tab_defs):
                    with tab:
                        stage = stages_by_name[key]
                        data = stage.get("data", {})
                        elapsed = stage.get("elapsed_ms")
                        if elapsed is not None:
                            st.caption(f"⏱️ {elapsed:.1f} ms")

                        if key == "query_processing":
                            _render_query_processing_stage(data)
                        elif key == "dense_retrieval":
                            _render_retrieval_stage(data, "Dense")
                        elif key == "sparse_retrieval":
                            _render_retrieval_stage(data, "Sparse")
                        elif key == "fusion":
                            _render_fusion_stage(data)
                        elif key == "source_allocation":
                            _render_source_allocation_stage(data)
                        elif key == "answer_unit_normalization":
                            _render_answer_unit_stage(data)
                        elif key == "rerank":
                            _render_rerank_stage(data)
                        elif key == "answer_unit_rerank":
                            _render_rerank_stage(data)
                        elif key == "answer_unit_dedup":
                            _render_answer_unit_stage(data)
            else:
                st.info("No stage details available.")

            # ── 5. Ragas Evaluate button ───────────────────────
            _render_evaluate_button(trace, idx)


def _render_evaluate_button(trace: Dict[str, Any], idx: int) -> None:
    """Render a Ragas evaluate button for a single query trace.

    Re-runs retrieval for the stored query and evaluates with
    RagasEvaluator (LLM-as-Judge).  Only works when query text
    is available in trace metadata.
    """
    meta = trace.get("metadata", {})
    query = meta.get("query", "")
    if not query:
        return

    st.divider()
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        clicked = st.button(
            "📏 Ragas Evaluate",
            key=f"eval_trace_{idx}",
            help="Re-run this query and score with Ragas (LLM-as-Judge)",
        )
    with col_info:
        st.caption(
            "Uses Ragas to score faithfulness, answer relevancy, "
            "and context precision. Calls LLM — may take a few seconds."
        )

    # Show previous result from session state
    result_key = f"eval_result_{idx}"
    if result_key in st.session_state and not clicked:
        _display_eval_metrics(st.session_state[result_key])

    if clicked:
        with st.spinner("Running Ragas evaluation…"):
            result = _evaluate_single_trace(query, meta)
        st.session_state[result_key] = result
        _display_eval_metrics(result)


def _evaluate_single_trace(
    query: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-run retrieval and evaluate a single query with Ragas.

    Returns dict with 'metrics' (score dict) or 'error' (str).
    """
    try:
        from dataclasses import replace as dc_replace

        from src.core.settings import load_settings, EvaluationSettings
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        settings = load_settings()

        # Override evaluation settings to force Ragas (frozen dataclass, use replace)
        ragas_eval = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=list(settings.evaluation.metrics) if settings.evaluation.metrics else [],
        )
        settings = dc_replace(settings, evaluation=ragas_eval)
        evaluator = EvaluatorFactory.create(settings)

        # Re-run retrieval
        collection = meta.get("collection", "default")
        top_k = meta.get("top_k", 10)
        chunks = _retrieve_chunks(settings, query, top_k, collection)

        if not chunks:
            return {"error": "No chunks retrieved — is data indexed?"}

        # Build a concise answer from top chunks (limit length to avoid
        # Ragas LLM max_tokens overflow during statement extraction)
        _MAX_ANSWER_CHARS = 1500
        texts = []
        for c in chunks:
            if hasattr(c, "text"):
                texts.append(c.text)
            elif isinstance(c, dict):
                texts.append(c.get("text", str(c)))
            else:
                texts.append(str(c))
        answer = " ".join(texts[:3])
        if len(answer) > _MAX_ANSWER_CHARS:
            answer = answer[:_MAX_ANSWER_CHARS]

        # Evaluate
        metrics = evaluator.evaluate(
            query=query,
            retrieved_chunks=chunks,
            generated_answer=answer,
        )
        return {"metrics": metrics}

    except ImportError as exc:
        return {"error": f"Ragas not installed: {exc}"}
    except Exception as exc:
        logger.exception("Ragas evaluation failed")
        return {"error": str(exc)}


def _retrieve_chunks(
    settings: Any,
    query: str,
    top_k: int,
    collection: str,
) -> list:
    """Re-run HybridSearch to retrieve chunks for evaluation."""
    try:
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        from src.storage.runtime import create_sparse_index

        vector_store = VectorStoreFactory.create(
            settings, collection_name=collection,
        )
        embedding_client = EmbeddingFactory.create(settings)
        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        bm25_indexer = create_sparse_index(settings, collection=collection)
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection
        query_processor = QueryProcessor()
        hybrid_search = create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )

        results = hybrid_search.search(query=query, top_k=top_k)
        return results if isinstance(results, list) else results.results
    except Exception as exc:
        logger.warning("Retrieval for evaluation failed: %s", exc)
        return []


def _display_eval_metrics(result: Dict[str, Any]) -> None:
    """Display evaluation result (metrics or error)."""
    if "error" in result:
        st.error(f"❌ Evaluation failed: {result['error']}")
        return

    metrics = result.get("metrics", {})
    if not metrics:
        st.warning("No metrics returned.")
        return

    st.markdown("**📏 Ragas Scores**")
    cols = st.columns(min(len(metrics), 4))
    for i, (name, value) in enumerate(sorted(metrics.items())):
        with cols[i % len(cols)]:
            st.metric(
                label=name.replace("_", " ").title(),
                value=f"{value:.4f}",
            )


def _extract_pipeline_chunks(
    timings: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract chunk lists from each pipeline stage."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for stage in timings:
        name = stage.get("stage_name", "")
        data = stage.get("data") or {}
        chunks = data.get("chunks")
        if chunks and isinstance(chunks, list):
            result[name] = chunks
    final = meta.get("final_results") or meta.get("results")
    if final and isinstance(final, list):
        result["final"] = final
    return result


# ═══════════════════════════════════════════════════════════════
# Per-stage renderers
# ═══════════════════════════════════════════════════════════════

def _render_query_processing_stage(data: Dict[str, Any]) -> None:
    """Render Query Processing stage: original query → keywords."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original Query**")
        st.info(data.get("original_query", "—"))
    with c2:
        st.markdown("**Method**")
        st.code(data.get("method", "—"))

    keywords = data.get("keywords", [])
    if keywords:
        st.markdown("**Extracted Keywords**")
        st.markdown(" · ".join(f"`{kw}`" for kw in keywords))
    else:
        st.warning("No keywords extracted.")


def _render_retrieval_stage(data: Dict[str, Any], label: str) -> None:
    """Render Dense or Sparse retrieval stage: method, counts, chunk list."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Method", data.get("method", "—"))
    with c2:
        extra = data.get("provider", data.get("keyword_count", "—"))
        extra_label = "Provider" if "provider" in data else "Keywords"
        st.metric(extra_label, extra)
    with c3:
        st.metric("Results", data.get("result_count", 0))

    st.markdown(f"**Top-K requested:** `{data.get('top_k', '—')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"{label.lower().replace(' ', '_')}_chunk")
    else:
        st.info(f"No {label.lower()} results returned.")


def _render_fusion_stage(data: Dict[str, Any]) -> None:
    """Render Fusion (RRF) stage: input lists, fused result count, chunk list."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Method", data.get("method", "rrf"))
    with c2:
        st.metric("Input Lists", data.get("input_lists", "—"))
    with c3:
        st.metric("Fused Results", data.get("result_count", 0))

    st.markdown(f"**Top-K:** `{data.get('top_k', '—')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix="fusion_chunk")
    else:
        st.info("No fusion results.")


def _render_rerank_stage(data: Dict[str, Any]) -> None:
    """Render Rerank stage: method, input/output counts, reranked chunk list."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Method", data.get("method", "—"))
    with c2:
        st.metric("Provider", data.get("provider", "—"))
    with c3:
        st.metric("Input", data.get("input_count", "—"))
    with c4:
        st.metric("Output", data.get("output_count", "—"))

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix="rerank_chunk")
    else:
        st.info("No reranked results.")


def _render_source_allocation_stage(data: Dict[str, Any]) -> None:
    """Render source-aware allocation detail."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Preferred Sources**")
        st.code(", ".join(data.get("preferred_sources", [])) or "—")
        st.markdown("**Source Weights**")
        st.json(data.get("source_weights", {}), expanded=False)
    with c2:
        st.markdown("**Unit Budgets**")
        st.json(data.get("source_unit_budgets", {}), expanded=False)
        st.markdown("**Raw Overfetch**")
        st.json(data.get("source_raw_overfetch", {}), expanded=False)


def _render_answer_unit_stage(data: Dict[str, Any]) -> None:
    """Render answer-unit normalization or dedup stage."""
    cols = st.columns(3)
    with cols[0]:
        st.metric("Child Hits", data.get("child_hit_count", 0))
    with cols[1]:
        st.metric("Parent Promotions", data.get("parent_promotions", 0))
    with cols[2]:
        st.metric("Collapsed Parents", data.get("collapsed_parent_count", 0))

    if data.get("source_raw_candidate_counts"):
        st.markdown("**Raw Candidate Counts**")
        st.json(data.get("source_raw_candidate_counts", {}), expanded=False)
    if data.get("source_normalized_unit_counts"):
        st.markdown("**Normalized Unit Counts**")
        st.json(data.get("source_normalized_unit_counts", {}), expanded=False)
    if data.get("input_count") is not None or data.get("output_count") is not None:
        st.markdown(
            f"**Input / Output:** `{data.get('input_count', '—')}` → `{data.get('output_count', '—')}`"
        )


def _render_chunk_list(chunks: List[Dict[str, Any]], prefix: str = "chunk") -> None:
    """Render a list of chunk dicts as a compact, readable table with expandable text."""
    for ci, chunk in enumerate(chunks):
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "")
        source = chunk.get("source", "")
        title = chunk.get("title", "")

        # Colour-coded score indicator
        if score >= 0.8:
            score_bar = "🟢"
        elif score >= 0.5:
            score_bar = "🟡"
        else:
            score_bar = "🔴"

        header = f"{score_bar} **#{ci + 1}** — Score: `{score:.4f}`"
        if title:
            header += f" — {title}"

        with st.expander(header, expanded=False):
            cols = st.columns([2, 3])
            with cols[0]:
                st.caption(f"Chunk ID: `{chunk_id}`")
            with cols[1]:
                if source:
                    st.caption(f"Source: `{source}`")
            # Show chunk text (scrollable)
            if text:
                st.text_area(
                    f"{prefix}_{ci}",
                    value=text,
                    height=max(80, min(len(text) // 2, 400)),
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.caption("_No text available_")


def _find_stage(timings, name):
    """Find a stage dict by name, or None."""
    for t in timings:
        if t["stage_name"] == name:
            return t
    return None
