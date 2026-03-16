"""Runtime factories for local-vs-production storage selection."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.conversation import FileConversationStore
from src.agent.hooks.rate_limit import CircuitBreaker, CircuitBreakerBackend, RateLimitHook
from src.agent.hooks.redis_circuit_breaker import RedisCircuitBreaker
from src.agent.hooks.redis_rate_limit import RedisRateLimitHook
from src.agent.memory.error_memory import ErrorMemory
from src.agent.memory.feedback_store import FeedbackStore
from src.agent.memory.knowledge_map import KnowledgeMapMemory
from src.agent.memory.session_memory import SessionMemory
from src.agent.memory.skill_memory import SkillMemory
from src.agent.memory.student_profile import StudentProfileMemory
from src.core.cache.redis_semantic_cache import RedisSemanticCache
from src.core.cache.semantic_cache import SemanticCache
from src.core.settings import Settings, resolve_path
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.storage.object_store import LocalObjectStore, ObjectImageStorage, ObjectStore, create_object_store
from src.storage.postgres_backends import (
    IngestionTaskStore,
    PostgresConversationStore,
    PostgresDocumentRegistry,
    PostgresErrorMemory,
    PostgresFeedbackStore,
    PostgresIntegrityChecker,
    PostgresKnowledgeMapMemory,
    PostgresSessionMemory,
    PostgresSkillMemory,
    PostgresStudentProfileMemory,
)
from src.storage.sparse_index import SparseIndex, create_sparse_index as create_sparse_index_backend


@dataclass
class IngestionBackends:
    integrity_checker: Any
    image_storage: Any
    object_store: ObjectStore
    document_registry: PostgresDocumentRegistry | None = None
    task_store: IngestionTaskStore | None = None


def _normalize_scope_segment(value: str) -> str:
    lowered = value.strip().lower() or "default"
    return re.sub(r"[^a-z0-9_.-]+", "-", lowered)


def build_llm_circuit_breaker_scope(settings: Settings) -> str:
    llm = settings.llm
    endpoint_identity = "default"
    if llm.base_url:
        endpoint_identity = f"base_url:{llm.base_url}"
    elif llm.azure_endpoint or llm.deployment_name:
        endpoint_identity = f"azure:{llm.azure_endpoint or ''}:{llm.deployment_name or ''}"
    endpoint_hash = hashlib.sha256(endpoint_identity.encode("utf-8")).hexdigest()[:12]
    provider = _normalize_scope_segment(llm.provider)
    model = _normalize_scope_segment(llm.model)
    return f"circuit_breaker:{provider}:{model}:{endpoint_hash}"


def create_object_store_from_settings(settings: Settings) -> ObjectStore:
    return create_object_store(settings)


def create_conversation_store(raw_settings: dict[str, Any], agent_cfg: Any) -> Any:
    postgres_cfg = raw_settings.get("postgres", {})
    if postgres_cfg.get("enabled") and postgres_cfg.get("dsn"):
        return PostgresConversationStore(postgres_cfg["dsn"])
    return FileConversationStore(agent_cfg.conversation_store_dir)


def create_memory_stores(memory_cfg: Any, raw_settings: dict[str, Any]) -> dict[str, Any]:
    postgres_cfg = raw_settings.get("postgres", {})
    if postgres_cfg.get("enabled") and postgres_cfg.get("dsn"):
        dsn = postgres_cfg["dsn"]
        return {
            "profile": PostgresStudentProfileMemory(dsn) if memory_cfg.profile_enabled else None,
            "error": PostgresErrorMemory(dsn) if memory_cfg.error_memory_enabled else None,
            "knowledge_map": PostgresKnowledgeMapMemory(dsn) if memory_cfg.knowledge_map_enabled else None,
            "skill": PostgresSkillMemory(dsn) if memory_cfg.skill_memory_enabled else None,
            "session": PostgresSessionMemory(dsn) if memory_cfg.session_memory_enabled else None,
        }
    return {
        "profile": StudentProfileMemory(memory_cfg.db_dir) if memory_cfg.profile_enabled else None,
        "error": ErrorMemory(memory_cfg.db_dir) if memory_cfg.error_memory_enabled else None,
        "knowledge_map": KnowledgeMapMemory(memory_cfg.db_dir) if memory_cfg.knowledge_map_enabled else None,
        "skill": SkillMemory(memory_cfg.db_dir) if memory_cfg.skill_memory_enabled else None,
        "session": SessionMemory(memory_cfg.db_dir) if memory_cfg.session_memory_enabled else None,
    }


def create_feedback_store(raw_settings: dict[str, Any]) -> Any:
    postgres_cfg = raw_settings.get("postgres", {})
    if postgres_cfg.get("enabled") and postgres_cfg.get("dsn"):
        return PostgresFeedbackStore(postgres_cfg["dsn"])
    return FeedbackStore()


def create_semantic_cache(
    settings: Settings,
    *,
    embedding_fn: Any,
    similarity_threshold: float,
    ttl_seconds: int,
    max_size: int,
) -> Any:
    provider = (getattr(settings, "_raw", {}) or {}).get("semantic_cache", {}).get("provider", "memory")
    if provider == "redis" and settings.redis.enabled and settings.redis.url:
        return RedisSemanticCache(
            redis_url=settings.redis.url,
            embedding_fn=embedding_fn,
            similarity_threshold=similarity_threshold,
            ttl_seconds=ttl_seconds,
            max_size=max_size,
        )
    return SemanticCache(
        embedding_fn=embedding_fn,
        similarity_threshold=similarity_threshold,
        ttl_seconds=ttl_seconds,
        max_size=max_size,
    )


def create_rate_limit_hook(raw_settings: dict[str, Any], requests_per_minute: int) -> Any:
    redis_cfg = raw_settings.get("redis", {})
    if redis_cfg.get("enabled") and redis_cfg.get("url"):
        return RedisRateLimitHook(redis_cfg["url"], requests_per_minute=requests_per_minute)
    return RateLimitHook(requests_per_minute=requests_per_minute)


def create_circuit_breaker(settings: Settings) -> CircuitBreakerBackend:
    breaker_cfg = settings.llm_resilience.circuit_breaker
    if settings.redis.enabled and settings.redis.url:
        return RedisCircuitBreaker(
            settings.redis.url,
            scope_key=build_llm_circuit_breaker_scope(settings),
            failure_threshold=breaker_cfg.failure_threshold,
            cooldown_seconds=breaker_cfg.cooldown_seconds,
            probe_ttl_seconds=breaker_cfg.cooldown_seconds,
        )
    return CircuitBreaker(
        failure_threshold=breaker_cfg.failure_threshold,
        cooldown_seconds=breaker_cfg.cooldown_seconds,
    )


def create_sparse_index(settings: Settings, collection: str = "default") -> SparseIndex:
    return create_sparse_index_backend(settings, collection=collection)


def create_image_storage(settings: Settings, collection: str = "default") -> Any:
    if settings.postgres.enabled and settings.postgres.dsn:
        object_store = create_object_store_from_settings(settings)
        return ObjectImageStorage(
            dsn=settings.postgres.dsn,
            object_store=object_store,
            images_prefix=settings.object_store.images_prefix,
            cache_root=str(resolve_path("data/images_cache")),
        )
    return ImageStorage(
        db_path=str(resolve_path("data/db/image_index.db")),
        images_root=str(resolve_path("data/images")),
    )


def create_ingestion_backends(settings: Settings, collection: str = "default") -> IngestionBackends:
    object_store = create_object_store_from_settings(settings)
    if settings.postgres.enabled and settings.postgres.dsn:
        return IngestionBackends(
            integrity_checker=PostgresIntegrityChecker(settings.postgres.dsn),
            image_storage=create_image_storage(settings, collection),
            object_store=object_store,
            document_registry=PostgresDocumentRegistry(settings.postgres.dsn),
            task_store=IngestionTaskStore(settings.postgres.dsn),
        )
    return IngestionBackends(
        integrity_checker=SQLiteIntegrityChecker(db_path=str(resolve_path("data/db/ingestion_history.db"))),
        image_storage=create_image_storage(settings, collection),
        object_store=object_store if not isinstance(object_store, LocalObjectStore) else object_store,
    )
