"""Modular RAG Dashboard – multi-page Streamlit application.

Entry-point: ``streamlit run src/observability/dashboard/app.py``

Pages are registered via ``st.navigation()`` and rendered by their
respective modules under ``pages/``.  Pages not yet implemented show
a placeholder message.
"""

from __future__ import annotations

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional UI dependency
    STREAMLIT_AVAILABLE = False

    class _StreamlitStub:
        def __getattr__(self, name: str):
            def _missing(*args, **kwargs):
                raise ModuleNotFoundError(
                    "streamlit is required to run the dashboard app"
                )

            return _missing

    st = _StreamlitStub()


# ── Page definitions ─────────────────────────────────────────────────

def _page_overview() -> None:
    from src.observability.dashboard.pages.overview import render
    render()


def _page_data_browser() -> None:
    from src.observability.dashboard.pages.data_browser import render
    render()


def _page_ingestion_manager() -> None:
    from src.observability.dashboard.pages.ingestion_manager import render
    render()


def _page_ingestion_traces() -> None:
    from src.observability.dashboard.pages.ingestion_traces import render
    render()


def _page_query_traces() -> None:
    from src.observability.dashboard.pages.query_traces import render
    render()


def _page_agent_traces() -> None:
    from src.observability.dashboard.pages.agent_traces import render
    render()


def _page_memory_traces() -> None:
    from src.observability.dashboard.pages.memory_traces import render
    render()


def _page_evaluation_panel() -> None:
    from src.observability.dashboard.pages.evaluation_panel import render
    render()


def _page_agent_evaluation_panel() -> None:
    from src.observability.dashboard.pages.agent_evaluation_panel import render
    render()


# ── Navigation ───────────────────────────────────────────────────────

pages = []
if STREAMLIT_AVAILABLE:
    pages = [
        st.Page(_page_overview, title="Overview", icon="📊", default=True),
        st.Page(_page_data_browser, title="Data Browser", icon="🔍"),
        st.Page(_page_ingestion_manager, title="Ingestion Manager", icon="📥"),
        st.Page(_page_ingestion_traces, title="Ingestion Traces", icon="🔬"),
        st.Page(_page_query_traces, title="Query Traces", icon="🔎"),
        st.Page(_page_agent_traces, title="Agent Traces", icon="🤖"),
        st.Page(_page_memory_traces, title="Memory Traces", icon="🧠"),
        st.Page(_page_evaluation_panel, title="Evaluation Panel", icon="📏"),
        st.Page(_page_agent_evaluation_panel, title="Agent Evaluation", icon="🧪"),
    ]


def main() -> None:
    if not STREAMLIT_AVAILABLE:
        raise ModuleNotFoundError("streamlit is required to run the dashboard app")
    st.set_page_config(
        page_title="Modular RAG Dashboard",
        page_icon="📊",
        layout="wide",
    )

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
elif STREAMLIT_AVAILABLE:
    # When run directly via `streamlit run app.py`
    main()
