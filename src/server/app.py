"""FastAPI application factory and startup wiring."""

from __future__ import annotations

import logging
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
from src.agent.conversation import FileConversationStore
from src.agent.llm.factory import create_llm_service
from src.agent.hooks.rate_limit import RateLimitHook
from src.agent.hooks.retry_middleware import RetryWithBackoffMiddleware
from src.agent.hooks.review_schedule import ReviewScheduleHook
from src.agent.memory.context_filter import ContextEngineeringFilter
from src.agent.memory.enhancer import MemoryContextEnhancer, MemoryRecordHook
from src.agent.memory.error_memory import ErrorMemory
from src.agent.memory.knowledge_map import KnowledgeMapMemory
from src.agent.memory.session_memory import SessionMemory
from src.agent.memory.skill_memory import SkillMemory
from src.agent.memory.student_profile import StudentProfileMemory
from src.agent.prompt_builder import SystemPromptBuilder
from src.agent.skills.registry import SkillRegistry
from src.agent.skills.workflow import SkillWorkflowHandler
from src.agent.tools.base import ToolRegistry
from src.agent.tools.document_ingest import DocumentIngestTool
from src.agent.tools.knowledge_query import KnowledgeQueryTool
from src.agent.tools.quiz_evaluator import QuizEvaluatorTool
from src.agent.tools.quiz_generator import QuizGeneratorTool
from src.agent.tools.review_summary import ReviewSummaryTool
from src.server.chat_handler import ChatHandler
from src.server.routes import configure_routes, router

logger = logging.getLogger(__name__)


def _load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_hybrid_search(collection: str = "computer_network") -> tuple:
    """Build a shared HybridSearch instance and return (hybrid, embedding, query_enhancer)."""
    from src.core.settings import load_settings as load_core_settings, resolve_path
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.hybrid_search import create_hybrid_search
    from src.core.query_engine.dense_retriever import create_dense_retriever
    from src.core.query_engine.sparse_retriever import create_sparse_retriever
    from src.core.query_engine.query_enhancer import QueryEnhancer
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.libs.embedding.cached_embedding import CachedEmbedding
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    core_settings = load_core_settings()
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
    bm25 = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
    sparse = create_sparse_retriever(
        settings=core_settings,
        bm25_indexer=bm25,
        vector_store=vector_store,
    )
    sparse.default_collection = collection

    hybrid = create_hybrid_search(
        settings=core_settings,
        query_processor=QueryProcessor(),
        dense_retriever=dense,
        sparse_retriever=sparse,
    )
    hybrid.embedding_client = embedding

    query_enhancer = QueryEnhancer(
        embedding_fn=lambda texts: embedding.embed(texts),
    )

    logger.info("Shared HybridSearch built for collection: %s (embedding cache=%d)", collection, cache_size)
    return hybrid, embedding, query_enhancer


def _auto_ingest(ingest_dir: str, collection: str) -> None:
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
    core_settings = load_core_settings()
    pipeline = IngestionPipeline(core_settings, collection=collection)

    try:
        for f in files:
            result = pipeline.run(str(f))
            if result.success:
                if result.stages.get("integrity", {}).get("skipped"):
                    logger.info("  ⏭ %s — already ingested", f.name)
                else:
                    logger.info("  ✓ %s — %d chunks", f.name, result.chunk_count)
            else:
                logger.error("  ✗ %s — %s", f.name, result.error)
    finally:
        pipeline.close()


def create_app(settings_path: str = "config/settings.yaml") -> FastAPI:
    """Wire all components and return a configured FastAPI app."""
    settings = _load_settings(settings_path)
    agent_cfg = load_agent_config(settings)
    server_cfg = load_server_config(settings)
    memory_cfg = load_memory_config(settings)

    # --- LLM ---
    llm = create_llm_service(settings)

    # --- Memory stores ---
    profile_mem = StudentProfileMemory(memory_cfg.db_dir) if memory_cfg.profile_enabled else None
    error_mem = ErrorMemory(memory_cfg.db_dir) if memory_cfg.error_memory_enabled else None
    kmap_mem = KnowledgeMapMemory(memory_cfg.db_dir) if memory_cfg.knowledge_map_enabled else None
    skill_mem = SkillMemory(memory_cfg.db_dir) if memory_cfg.skill_memory_enabled else None
    session_mem = SessionMemory(memory_cfg.db_dir) if memory_cfg.session_memory_enabled else None

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
        )

    # --- Shared HybridSearch ---
    collection = agent_cfg.default_collection
    hybrid_search, cached_embedding, query_enhancer = _build_hybrid_search(collection)
    query_enhancer.llm_service = llm

    # --- Tools ---
    tool_registry = ToolRegistry()
    tool_registry.register(KnowledgeQueryTool(
        settings=settings,
        hybrid_search=hybrid_search,
        query_enhancer=query_enhancer,
    ))
    tool_registry.register(DocumentIngestTool(settings=settings))
    tool_registry.register(ReviewSummaryTool(
        hybrid_search=hybrid_search,
        llm_service=llm,
        error_memory=error_mem,
        knowledge_map=kmap_mem,
    ))
    tool_registry.register(QuizGeneratorTool(
        hybrid_search=hybrid_search,
        llm_service=llm,
        error_memory=error_mem,
        knowledge_map=kmap_mem,
    ))
    tool_registry.register(QuizEvaluatorTool(
        llm_service=llm,
        error_memory=error_mem,
        knowledge_map=kmap_mem,
    ))

    # --- Skills ---
    skill_registry = SkillRegistry(agent_cfg.skills_dir)
    skill_workflow = SkillWorkflowHandler(skill_registry)

    # --- Conversation ---
    conv_store = FileConversationStore(agent_cfg.conversation_store_dir)

    # --- Agent ---
    rate_limit_hook = RateLimitHook(requests_per_minute=agent_cfg.rate_limit_rpm)
    hooks: list = [rate_limit_hook, memory_hook]
    if review_hook:
        hooks.insert(0, review_hook)

    # --- LLM Middlewares (retry + circuit breaker) ---
    retry_middleware = RetryWithBackoffMiddleware(llm_service=llm)

    agent = Agent(
        llm_service=llm,
        tool_registry=tool_registry,
        conversation_store=conv_store,
        config=agent_cfg,
        prompt_builder=SystemPromptBuilder(agent_cfg.system_prompt_path),
        lifecycle_hooks=hooks,
        llm_middlewares=[retry_middleware],
        memory_enhancer=memory_enhancer,
        skill_workflow=skill_workflow,
        context_filter=context_filter,
        review_hook=review_hook,
    )

    chat_handler = ChatHandler(agent)

    # --- FastAPI ---
    app = FastAPI(title="Course Learning Agent", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from src.agent.memory.feedback_store import FeedbackStore
    feedback_store = FeedbackStore()
    configure_routes(
        chat_handler, agent,
        server_cfg.upload_dir, server_cfg.max_upload_size_mb,
        feedback_store=feedback_store,
    )
    app.include_router(router)

    web_dir = Path("src/web")
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return (web_dir / "index.html").read_text(encoding="utf-8")

    # --- Startup: auto-ingest course materials ---
    @app.on_event("startup")
    async def _startup_auto_ingest() -> None:
        if agent_cfg.auto_ingest_dir:
            logger.info("Running startup auto-ingestion from: %s", agent_cfg.auto_ingest_dir)
            _auto_ingest(agent_cfg.auto_ingest_dir, collection)

    # --- Shutdown: close DB connections and resources ---
    @app.on_event("shutdown")
    async def _shutdown_cleanup() -> None:
        logger.info("Shutting down — closing resources...")
        for store in (profile_mem, error_mem, kmap_mem, skill_mem, session_mem):
            if store and hasattr(store, "close"):
                try:
                    await store.close()
                except Exception:
                    logger.warning("Failed to close memory store %s", type(store).__name__)
        if feedback_store:
            try:
                feedback_store.close()
            except Exception:
                logger.warning("Failed to close FeedbackStore")
        logger.info("Shutdown complete")

    logger.info(
        "Agent ready — tools: %s, skills: %s",
        tool_registry.tool_names,
        skill_registry.skill_names,
    )
    return app
