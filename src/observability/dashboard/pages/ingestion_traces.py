"""Ingestion Traces page – browse ingestion trace history with per-stage detail.

Layout:
1. Trace list (reverse-chronological, filtered to trace_type=="ingestion")
2. Pipeline overview: source file, total time, stage timing waterfall
3. Per-stage detail tabs:
   📄 Load    – raw document text preview
   ✂️ Split   – chunk list with text
   🔄 Transform – before/after diff, enrichment metadata
   🔢 Embed   – vector stats
   💾 Upsert  – stored IDs
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService

logger = logging.getLogger(__name__)


def render() -> None:
    """Render the Ingestion Traces page."""
    st.header("🔬 Ingestion Traces")

    svc = TraceService()
    traces = svc.list_traces(trace_type="ingestion")

    if not traces:
        st.info("No ingestion traces recorded yet. Run an ingestion first!")
        return

    st.subheader(f"📋 Trace History ({len(traces)})")

    for idx, trace in enumerate(traces):
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        total_ms = trace.get("elapsed_ms")
        total_label = f"{total_ms:.0f} ms" if total_ms is not None else "—"
        meta = trace.get("metadata", {})
        source_path = meta.get("source_path", "—")

        # Build expander title
        file_name = source_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if source_path != "—" else "—"
        expander_title = f"📄 **{file_name}** · {total_label} · {started[:19]}"

        with st.expander(expander_title, expanded=(idx == 0)):
            timings = svc.get_stage_timings(trace)
            stages_by_name = {t["stage_name"]: t for t in timings}

            # ── 1. Overview metrics ────────────────────────────
            st.markdown("#### 📊 Pipeline Overview")
            st.caption(f"Source: `{source_path}`")

            load_d = stages_by_name.get("load", {}).get("data", {})
            split_d = stages_by_name.get("split", {}).get("data", {})
            transform_d = stages_by_name.get("transform", {}).get("data", {})
            embed_d = stages_by_name.get("embed", {}).get("data", {})
            upsert_d = stages_by_name.get("upsert", {}).get("data", {})

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Doc Length", f"{load_d.get('text_length', 0):,} chars")
            with c2:
                st.metric("Chunks", split_d.get("chunk_count", 0))
            with c3:
                st.metric("Images", load_d.get("image_count", 0))
            with c4:
                st.metric("Vectors", upsert_d.get("vector_count", 0))
            with c5:
                st.metric("Total Time", total_label)

            st.divider()

            # ── 2. Stage timing waterfall ──────────────────────
            # Filter to main pipeline stages only (not sub-stages)
            main_stages = [
                t for t in timings
                if t["stage_name"] in ("load", "split", "transform", "embed", "upsert")
            ]
            if main_stages:
                st.markdown("#### ⏱️ Stage Timings")
                chart_data = {t["stage_name"]: t["elapsed_ms"] for t in main_stages}
                st.bar_chart(chart_data, horizontal=True)
                st.table([
                    {
                        "Stage": t["stage_name"],
                        "Elapsed (ms)": round(t["elapsed_ms"], 2),
                    }
                    for t in main_stages
                ])

            st.divider()

            # ── 3. Per-stage detail tabs ───────────────────────
            st.markdown("#### 🔍 Stage Details")

            tab_defs = []
            if "load" in stages_by_name:
                tab_defs.append(("📄 Load", "load"))
            if "split" in stages_by_name:
                tab_defs.append(("✂️ Split", "split"))
            if "transform" in stages_by_name:
                tab_defs.append(("🔄 Transform", "transform"))
            if "embed" in stages_by_name:
                tab_defs.append(("🔢 Embed", "embed"))
            if "upsert" in stages_by_name:
                tab_defs.append(("💾 Upsert", "upsert"))

            if tab_defs:
                tabs = st.tabs([label for label, _ in tab_defs])
                for tab, (label, key) in zip(tabs, tab_defs):
                    with tab:
                        stage = stages_by_name[key]
                        data = stage.get("data", {})
                        elapsed = stage.get("elapsed_ms")
                        if elapsed is not None:
                            st.caption(f"⏱️ {elapsed:.1f} ms")

                        if key == "load":
                            _render_load_stage(data)
                        elif key == "split":
                            _render_split_stage(data)
                        elif key == "transform":
                            _render_transform_stage(data)
                        elif key == "embed":
                            _render_embed_stage(data)
                        elif key == "upsert":
                            _render_upsert_stage(data)
            else:
                st.info("No stage details available.")


# ═══════════════════════════════════════════════════════════════
# Per-stage renderers
# ═══════════════════════════════════════════════════════════════

def _render_load_stage(data: Dict[str, Any]) -> None:
    """Render Load stage: raw document preview."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Doc ID", data.get("doc_id", "—")[:16])
    with c2:
        st.metric("Text Length", f"{data.get('text_length', 0):,}")
    with c3:
        st.metric("Images", data.get("image_count", 0))

    preview = data.get("text_preview", "")
    if preview:
        st.markdown("**Raw Document Text**")
        st.text_area(
            "raw_text",
            value=preview,
            height=max(120, min(len(preview) // 2, 600)),
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.info("No text preview recorded in this trace.")


def _render_split_stage(data: Dict[str, Any]) -> None:
    """Render Split stage: chunk list with texts."""
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Chunks", data.get("chunk_count", 0))
    with c2:
        st.metric("Avg Size", f"{data.get('avg_chunk_size', 0)} chars")

    chunks = data.get("chunks", [])
    if chunks:
        st.markdown("**Chunks after splitting**")
        for i, chunk in enumerate(chunks):
            char_len = chunk.get("char_len", 0)
            chunk_id = chunk.get("chunk_id", "")
            text = chunk.get("text", "")
            header = f"📝 **Chunk #{i+1}** — `{chunk_id[:20]}` — {char_len} chars"
            with st.expander(header, expanded=(i < 2)):
                st.text_area(
                    f"split_{i}",
                    value=text,
                    height=max(100, min(len(text) // 2, 500)),
                    disabled=True,
                    label_visibility="collapsed",
                )
    else:
        st.info("No chunk text recorded. Re-run ingestion to generate new traces.")


def _render_transform_stage(data: Dict[str, Any]) -> None:
    """Render Transform stage: before/after refinement + enrichment metadata."""
    # Summary metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Refined (LLM / Rule)",
            f"{data.get('refined_by_llm', 0)} / {data.get('refined_by_rule', 0)}",
        )
    with c2:
        st.metric(
            "Enriched (LLM / Rule)",
            f"{data.get('enriched_by_llm', 0)} / {data.get('enriched_by_rule', 0)}",
        )
    with c3:
        st.metric("Captioned", data.get("captioned_chunks", 0))

    chunks = data.get("chunks", [])
    if chunks:
        st.markdown("**Per-chunk transform results**")
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", "")
            refined_by = chunk.get("refined_by", "")
            enriched_by = chunk.get("enriched_by", "")
            title = chunk.get("title", "")
            tags = chunk.get("tags", [])
            summary = chunk.get("summary", "")
            text_before = chunk.get("text_before", "")
            text_after = chunk.get("text_after", "")

            badge_parts = []
            if refined_by:
                badge_parts.append(f"refined:`{refined_by}`")
            if enriched_by:
                badge_parts.append(f"enriched:`{enriched_by}`")
            badges = " · ".join(badge_parts)

            header = f"🔄 **Chunk #{i+1}** — `{chunk_id[:20]}` — {badges}"
            with st.expander(header, expanded=(i == 0)):
                # Metadata from enrichment
                if title or tags or summary:
                    st.markdown("**Enriched Metadata**")
                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        st.markdown(f"**Title:** {title}" if title else "_No title_")
                    with meta_cols[1]:
                        if tags:
                            st.markdown("**Tags:** " + ", ".join(f"`{t}`" for t in tags))
                        else:
                            st.markdown("_No tags_")
                    with meta_cols[2]:
                        if summary:
                            st.markdown(f"**Summary:** {summary}")

                # Before / After text comparison
                if text_before or text_after:
                    st.markdown("**Text Comparison**")
                    # Compute a uniform height so both sides match
                    _max_len = max(len(text_before or ""), len(text_after or ""))
                    _h = max(150, min(_max_len // 2, 600))
                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.markdown("*Before refinement:*")
                        st.text_area(
                            f"before_{i}",
                            value=text_before if text_before else "(empty)",
                            height=_h,
                            disabled=True,
                            label_visibility="collapsed",
                        )
                    with col_after:
                        st.markdown("*After refinement + enrichment:*")
                        st.text_area(
                            f"after_{i}",
                            value=text_after if text_after else "(empty)",
                            height=_h,
                            disabled=True,
                            label_visibility="collapsed",
                        )
    else:
        st.info("No per-chunk transform data recorded. Re-run ingestion for new traces.")


def _render_embed_stage(data: Dict[str, Any]) -> None:
    """Render Embed stage: dual-path Dense + Sparse encoding details."""
    # ── Overview metrics ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dense Vectors", data.get("dense_vector_count", 0))
    with c2:
        st.metric("Dimension", data.get("dense_dimension", 0))
    with c3:
        st.metric("Sparse Docs", data.get("sparse_doc_count", 0))
    with c4:
        st.metric("Method", data.get("method", "—"))

    chunks = data.get("chunks", [])
    if not chunks:
        st.info("No chunk encoding data recorded.")
        return

    # ── Dual-path per-chunk table ──
    st.markdown("---")
    dense_tab, sparse_tab = st.tabs(["🟦 Dense Encoding", "🟨 Sparse Encoding (BM25)"])

    with dense_tab:
        st.markdown("Each chunk → **float vector** via embedding model (e.g. `text-embedding-ada-002`)")
        dense_rows = []
        for i, chunk in enumerate(chunks):
            char_len = chunk.get("char_len", 0)
            dense_rows.append({
                "#": i + 1,
                "Chunk ID": chunk.get("chunk_id", ""),
                "Chars": char_len,
                "Est. Tokens": max(1, char_len // 3),
                "Dense Dim": chunk.get("dense_dim", data.get("dense_dimension", "—")),
            })
        st.table(dense_rows)

    with sparse_tab:
        st.markdown("Each chunk → **term frequency stats** for BM25 indexing")
        sparse_rows = []
        for i, chunk in enumerate(chunks):
            sparse_rows.append({
                "#": i + 1,
                "Chunk ID": chunk.get("chunk_id", ""),
                "Doc Length (terms)": chunk.get("doc_length", "—"),
                "Unique Terms": chunk.get("unique_terms", "—"),
            })
        st.table(sparse_rows)

        # Top terms per chunk
        for i, chunk in enumerate(chunks):
            top_terms = chunk.get("top_terms", [])
            if top_terms:
                with st.expander(f"🔤 Chunk {i + 1} — Top Terms", expanded=False):
                    term_rows = [{"Term": t["term"], "Freq": t["freq"]} for t in top_terms]
                    st.table(term_rows)


def _render_upsert_stage(data: Dict[str, Any]) -> None:
    """Render Upsert stage: per-store details with chunk mapping."""
    dense_store = data.get("dense_store", {})
    sparse_store = data.get("sparse_store", {})
    image_store = data.get("image_store", {})

    # ── Overview metrics ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Dense Vectors", dense_store.get("count", data.get("vector_count", 0)))
    with c2:
        st.metric("Sparse (BM25)", sparse_store.get("count", data.get("bm25_docs", 0)))
    with c3:
        st.metric("Images", image_store.get("count", data.get("images_indexed", 0)))

    # ── Dense store details ──
    if dense_store:
        with st.expander("🟦 Dense Vector Store (ChromaDB)", expanded=True):
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(f"**Backend:** `{dense_store.get('backend', '—')}`")
                st.markdown(f"**Collection:** `{dense_store.get('collection', '—')}`")
            with dc2:
                st.markdown(f"**Path:** `{dense_store.get('path', '—')}`")
                st.markdown(f"**Vectors:** {dense_store.get('count', 0)}")

    # ── Sparse store details ──
    if sparse_store:
        with st.expander("🟨 Sparse Index (BM25)", expanded=True):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f"**Backend:** `{sparse_store.get('backend', '—')}`")
                st.markdown(f"**Collection:** `{sparse_store.get('collection', '—')}`")
            with sc2:
                st.markdown(f"**Path:** `{sparse_store.get('path', '—')}`")
                st.markdown(f"**Documents:** {sparse_store.get('count', 0)}")

    # ── Image store details ──
    if image_store and image_store.get("count", 0) > 0:
        with st.expander(f"🖼️ Image Storage ({image_store.get('count', 0)} images)", expanded=True):
            st.markdown(f"**Backend:** `{image_store.get('backend', '—')}`")
            imgs = image_store.get("images", [])
            if imgs:
                img_rows = [
                    {
                        "Image ID": img.get("image_id", ""),
                        "Page": img.get("page", 0),
                        "File": img.get("file_path", ""),
                        "Doc Hash": img.get("doc_hash", "")[:16] + "…",
                    }
                    for img in imgs
                ]
                st.table(img_rows)

    # ── Chunk → Vector ID mapping ──
    chunk_mapping = data.get("chunk_mapping", [])
    if chunk_mapping:
        with st.expander(f"🔗 Chunk → Vector Mapping ({len(chunk_mapping)} entries)", expanded=False):
            mapping_rows = [
                {
                    "#": i + 1,
                    "Chunk ID": m.get("chunk_id", ""),
                    "Vector ID": m.get("vector_id", ""),
                    "Store": m.get("store", ""),
                    "Collection": m.get("collection", ""),
                }
                for i, m in enumerate(chunk_mapping)
            ]
            st.table(mapping_rows)

    # ── Fallback: legacy format with just vector_ids ──
    if not chunk_mapping and not dense_store:
        vector_ids = data.get("vector_ids", [])
        if vector_ids:
            with st.expander("Vector IDs", expanded=False):
                for vid in vector_ids:
                    st.code(vid, language=None)
