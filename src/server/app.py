"""FastAPI application factory and startup wiring."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.agent.agent import Agent
from src.agent.config import (
    AgentConfig,
    MemoryConfig,
    ServerConfig,
    load_agent_config,
    load_memory_config,
    load_server_config,
)
from src.agent.llm.factory import create_llm_service
from src.agent.hooks.retry_middleware import RetryWithBackoffMiddleware
from src.agent.hooks.review_schedule import ReviewScheduleHook
from src.agent.memory.context_filter import ContextEngineeringFilter
from src.agent.memory.enhancer import MemoryContextEnhancer, MemoryRecordHook
from src.agent.planner import TaskPlanner
from src.agent.prompt_builder import SystemPromptBuilder
from src.agent.skills.registry import SkillRegistry
from src.agent.skills.workflow import SkillWorkflowHandler
from src.agent.tools.base import ToolRegistry
from src.agent.tools.document_ingest import DocumentIngestTool
from src.agent.tools.knowledge_query import KnowledgeQueryTool
from src.agent.tools.quiz_evaluator import QuizEvaluatorTool
from src.agent.tools.quiz_generator import QuizGeneratorTool
from src.agent.tools.review_summary import ReviewSummaryTool
from src.core.trace.trace_collector import TraceCollector
from src.server.chat_handler import ChatHandler
from src.server.routes import configure_routes, router
from src.storage.runtime import (
    create_conversation_store,
    create_feedback_store,
    create_ingestion_backends,
    create_memory_stores,
    create_rate_limit_hook,
    create_semantic_cache,
    create_sparse_index,
)

logger = logging.getLogger(__name__)


def _resolve_settings_path(path: str = "config/settings.yaml") -> str:
    return os.environ.get("MODULAR_RAG_SETTINGS_PATH", path)


def _load_settings(path: str = "config/settings.yaml") -> dict:
    settings_path = _resolve_settings_path(path)
    with open(settings_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_hybrid_search(collection: str = "computer_network", settings_path: str = "config/settings.yaml") -> tuple:
    """Build a shared HybridSearch instance and return (hybrid, embedding, query_enhancer)."""
    from src.core.settings import load_settings as load_core_settings
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.core.query_engine.fusion import RRFFusion
    from src.core.query_engine.query_enhancer import QueryEnhancer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.embedding.cached_embedding import CachedEmbedding
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory
    from src.libs.reranker.reranker_factory import RerankerFactory

    settings_path = _resolve_settings_path(settings_path)
    core_settings = load_core_settings(settings_path)
    raw_embedding = EmbeddingFactory.create(core_settings)

    cache_size = 4096
    retrieval_cfg = getattr(core_settings, 'retrieval', None)
    if retrieval_cfg:
        cache_size = getattr(retrieval_cfg, 'embedding_cache_size', 4096)
    embedding = CachedEmbedding(raw_embedding, max_size=cache_size)

    vector_store = VectorStoreFactory.create(core_settings, collection_name=collection)
    dense = create_dense_retriever(
        settings=core_settings,
        embedding_client=embedding,
        vector_store=vector_store,
    )
    bm25 = create_sparse_index(core_settings, collection=collection)
    sparse = create_sparse_retriever(
        settings=core_settings,
        bm25_indexer=bm25,
        vector_store=vector_store,
    )
    sparse.default_collection = collection

    rrf_k = getattr(retrieval_cfg, 'rrf_k', 60) if retrieval_cfg else 60
    fusion = RRFFusion(k=rrf_k)

    reranker = None
    try:
        reranker = RerankerFactory.create(core_settings)
        logger.info("Reranker created: %s", type(reranker).__name__)
    except Exception as exc:
        logger.warning("Reranker creation failed, reranking disabled: %s", exc)

    hybrid = HybridSearch(
        settings=core_settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=fusion,
        reranker=reranker,
    )
    hybrid.embedding_client = embedding

    query_enhancer = QueryEnhancer(
        embedding_fn=lambda texts: embedding.embed(texts),
    )

    logger.info("Shared HybridSearch built for collection: %s (reranker=%s, embedding cache=%d)",
                collection, type(reranker).__name__ if reranker else "None", cache_size)
    return hybrid, embedding, query_enhancer


def _auto_ingest(ingest_dir: str, collection: str, settings_path: str = "config/settings.yaml") -> None:
    """Scan a directory and ingest any .pdf/.pptx files that haven't been processed."""
    from src.core.settings import load_settings as load_core_settings, resolve_path

    dir_path = resolve_path(ingest_dir)
    if not dir_path.is_dir():
        logger.warning("Auto-ingest directory not found: %s", dir_path)
        return

    files = sorted(
        p for p in dir_path.iterdir()
        if p.suffix.lower() in (".pdf", ".pptx") and p.is_file()
    )
    if not files:
        logger.info("Auto-ingest: no files found in %s", dir_path)
        return

    logger.info("Auto-ingest: found %d file(s) in %s", len(files), dir_path)

    from src.ingestion.pipeline import IngestionPipeline
    core_settings = load_core_settings(_resolve_settings_path(settings_path))
    pipeline = IngestionPipeline(core_settings, collection=collection)

    try:
        for f in files:
            from src.ingestion.pipeline import IngestionPipeline as _IP
            src_type = _IP.infer_source_type(f)
            result = pipeline.run(str(f), source_type=src_type)
            if result.success:
                if result.stages.get("integrity", {}).get("skipped"):
                    logger.info("  ⏭ %s — already ingested", f.name)
                else:
                    logger.info("  ✓ %s — %d chunks (source_type=%s)", f.name, result.chunk_count, src_type)
            else:
                logger.error("  ✗ %s — %s", f.name, result.error)
    finally:
        pipeline.close()


def create_app(settings_path: str = "config/settings.yaml") -> FastAPI:
    """Wire all components and return a configured FastAPI app."""
    settings_path = _resolve_settings_path(settings_path)
    settings = _load_settings(settings_path)
    from src.core.settings import load_settings as load_core_settings
    core_settings = load_core_settings(settings_path)
    agent_cfg = load_agent_config(settings)
    server_cfg = load_server_config(settings)
    memory_cfg = load_memory_config(settings)
    collection = agent_cfg.default_collection
    ingestion_backends = create_ingestion_backends(core_settings, collection=collection)
    shared_trace_collector = TraceCollector.from_settings(core_settings)

    # --- LLM ---
    llm = create_llm_service(settings)

    # --- Memory stores ---
    memory_stores = create_memory_stores(memory_cfg, settings)
    profile_mem = memory_stores["profile"]
    error_mem = memory_stores["error"]
    kmap_mem = memory_stores["knowledge_map"]
    skill_mem = memory_stores["skill"]
    session_mem = memory_stores["session"]

    memory_enhancer = MemoryContextEnhancer(
        student_profile=profile_mem,
        error_memory=error_mem,
        knowledge_map=kmap_mem,
        skill_memory=skill_mem,
        session_memory=session_mem,
    )
    memory_hook = MemoryRecordHook(
        student_profile=profile_mem,
        skill_memory=skill_mem,
        error_memory=error_mem,
        knowledge_map=kmap_mem,
        session_memory=session_mem,
        llm_service=llm,
        extraction_mode=memory_cfg.extraction_mode,
        write_gating_enabled=memory_cfg.write_gating_enabled,
        session_write_min_confidence=memory_cfg.session_write_min_confidence,
        preference_write_min_confidence=memory_cfg.preference_write_min_confidence,
        preference_conflict_guard=memory_cfg.preference_conflict_guard,
        trace_enabled=bool(settings.get("observability", {}).get("trace_enabled", False)),
        trace_collector=shared_trace_collector,
    )

    # --- Review schedule hook ---
    review_hook = None
    if memory_cfg.review_schedule_enabled:
        review_hook = ReviewScheduleHook(
            knowledge_map=kmap_mem,
            error_memory=error_mem,
            session_memory=session_mem,
            enable_decay=memory_cfg.decay_on_session_start,
        )

    # --- Context filter ---
    context_filter = None
    if memory_cfg.compaction_enabled:
        context_filter = ContextEngineeringFilter(
            max_messages=agent_cfg.max_context_messages,
            max_tokens=agent_cfg.max_context_tokens,
            llm_service=llm,
            compaction_threshold=memory_cfg.compaction_threshold_messages,
            object_store=ingestion_backends.object_store,
            object_prefix=core_settings.object_store.context_prefix,
        )

    # --- Shared HybridSearch ---
    hybrid_search, cached_embedding, query_enhancer = _build_hybrid_search(collection, settings_path=settings_path)
    query_enhancer.llm_service = llm

    # --- Query Router (dual-layer: rule + embedding) ---
    from src.core.query_engine.query_router import QueryRouter
    routing_cfg = settings.get("routing", {})
    if routing_cfg.get("enabled", True):
        query_router = QueryRouter(
            fallback_to_llm=routing_cfg.get("fallback_to_llm", True),
            embedding_fn=cached_embedding.embed if cached_embedding is not None else None,
            similarity_threshold=routing_cfg.get("embedding_threshold", 0.75),
        )
        if query_router.embedding_ready:
            logger.info("QueryRouter: dual-layer mode (rule + embedding, %d prototypes)", query_router.prototype_count)
        else:
            logger.info("QueryRouter: rule-only mode (no embedding_fn)")
    else:
        query_router = None

    # --- Semantic Cache ---
    semantic_cache = None
    cache_cfg = settings.get("semantic_cache", {})
    if cache_cfg.get("enabled", False) and cached_embedding is not None:
        semantic_cache = create_semantic_cache(
            core_settings,
            embedding_fn=lambda texts: cached_embedding.embed(texts),
            similarity_threshold=cache_cfg.get("similarity_threshold", 0.92),
            ttl_seconds=cache_cfg.get("ttl_seconds", 3600),
            max_size=cache_cfg.get("max_size", 500),
        )
        logger.info("SemanticCache enabled (threshold=%.2f, ttl=%ds)",
                     cache_cfg.get("similarity_threshold", 0.92),
                     cache_cfg.get("ttl_seconds", 3600))

    # --- Tools ---
    tool_registry = ToolRegistry()
    tool_registry.register(KnowledgeQueryTool(
        settings=settings,
        hybrid_search=hybrid_search,
        query_enhancer=query_enhancer,
        query_router=query_router,
        semantic_cache=semantic_cache,
        trace_collector=shared_trace_collector,
    ))
    tool_registry.register(DocumentIngestTool(
        settings=core_settings,
        semantic_cache=semantic_cache,
        object_store=ingestion_backends.object_store,
        document_registry=ingestion_backends.document_registry,
        task_store=ingestion_backends.task_store,
    ))
    tool_registry.register(ReviewSummaryTool(
        hybrid_search=hybrid_search,
        llm_service=llm,
        error_memory=error_mem,
        knowledge_map=kmap_mem,
        trace_enabled=bool(settings.get("observability", {}).get("trace_enabled", False)),
        trace_collector=shared_trace_collector,
    ))
    tool_registry.register(QuizGeneratorTool(
        hybrid_search=hybrid_search,
        llm_service=llm,
        error_memory=error_mem,
        knowledge_map=kmap_mem,
        query_router=query_router,
    ))
    tool_registry.register(QuizEvaluatorTool(
        llm_service=llm,
        error_memory=error_mem,
        knowledge_map=kmap_mem,
        hybrid_search=hybrid_search,
        trace_enabled=bool(settings.get("observability", {}).get("trace_enabled", False)),
        trace_collector=shared_trace_collector,
    ))

    # --- Skills ---
    skill_registry = SkillRegistry(agent_cfg.skills_dir)
    skill_workflow = SkillWorkflowHandler(skill_registry)
    task_planner = TaskPlanner(
        embedding_fn=cached_embedding.embed if cached_embedding is not None else None,
    )

    # --- Conversation ---
    conv_store = create_conversation_store(settings, agent_cfg)

    # --- Agent ---
    rate_limit_hook = create_rate_limit_hook(settings, agent_cfg.rate_limit_rpm)
    hooks: list = [rate_limit_hook, memory_hook]
    if review_hook:
        hooks.insert(0, review_hook)

    # --- Guardrails Hook (runs first in chain) ---
    guardrails_cfg = settings.get("guardrails", {})
    if guardrails_cfg.get("enabled", True):
        from src.agent.hooks.guardrails import GuardrailsHook
        guardrails_hook = GuardrailsHook(
            input_filtering=guardrails_cfg.get("input_filtering", True),
            output_redaction=guardrails_cfg.get("output_redaction", True),
            block_on_high_risk=guardrails_cfg.get("block_on_high_risk", True),
        )
        hooks.insert(0, guardrails_hook)

    # --- LLM Middlewares (retry + circuit breaker + reflection) ---
    retry_middleware = RetryWithBackoffMiddleware(llm_service=llm)
    llm_middlewares: list = [retry_middleware]

    reflection_cfg = settings.get("reflection", {})
    if reflection_cfg.get("enabled", True):
        from src.agent.hooks.reflection import ReflectionMiddleware
        reflection_mw = ReflectionMiddleware(
            groundedness_threshold=reflection_cfg.get("groundedness_threshold", 0.3),
            append_warning=reflection_cfg.get("append_warning", True),
        )
        llm_middlewares.append(reflection_mw)

    agent = Agent(
        llm_service=llm,
        tool_registry=tool_registry,
        conversation_store=conv_store,
        config=agent_cfg,
        prompt_builder=SystemPromptBuilder(agent_cfg.system_prompt_path),
        lifecycle_hooks=hooks,
        llm_middlewares=llm_middlewares,
        memory_enhancer=memory_enhancer,
        task_planner=task_planner,
        skill_workflow=skill_workflow,
        context_filter=context_filter,
        review_hook=review_hook,
        trace_enabled=bool(settings.get("observability", {}).get("trace_enabled", False)),
        trace_collector=shared_trace_collector,
    )

    chat_handler = ChatHandler(agent)

    feedback_store = create_feedback_store(settings)

    @asynccontextmanager
    async def _lifespan(_: FastAPI):
        if agent_cfg.auto_ingest_dir:
            logger.info("Running startup auto-ingestion from: %s", agent_cfg.auto_ingest_dir)
            _auto_ingest(agent_cfg.auto_ingest_dir, collection, settings_path=settings_path)
        try:
            yield
        finally:
            logger.info("Shutting down — closing resources...")
            for store in (profile_mem, error_mem, kmap_mem, skill_mem, session_mem):
                if store and hasattr(store, "close"):
                    try:
                        await store.close()
                    except Exception:
                        logger.warning("Failed to close memory store %s", type(store).__name__)
            if feedback_store:
                try:
                    maybe = feedback_store.close()
                    if hasattr(maybe, "__await__"):
                        await maybe
                except Exception:
                    logger.warning("Failed to close FeedbackStore")
            logger.info("Shutdown complete")

    # --- FastAPI ---
    app = FastAPI(title="Course Learning Agent", version="0.1.0", lifespan=_lifespan)
    app.state.agent = agent
    app.state.chat_handler = chat_handler
    app.state.settings_path = settings_path

    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    configure_routes(
        chat_handler, agent,
        server_cfg.upload_dir, server_cfg.max_upload_size_mb,
        feedback_store=feedback_store,
        object_store=ingestion_backends.object_store,
        task_store=ingestion_backends.task_store,
    )
    app.include_router(router)

    web_dir = Path("src/web")
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return (web_dir / "index.html").read_text(encoding="utf-8")

    logger.info(
        "Agent ready — tools: %s, skills: %s",
        tool_registry.tool_names,
        skill_registry.skill_names,
    )
    return app
