"""Shared storage adapters and factories for local and production backends."""

from src.storage.object_store import (
    LocalObjectStore,
    MinioObjectStore,
    ObjectImageStorage,
    ObjectStore,
    create_object_store,
)
from src.storage.postgres import PostgresExecutor, PostgresUnavailableError
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
from src.storage.runtime import (
    build_llm_circuit_breaker_scope,
    create_circuit_breaker,
    create_conversation_store,
    create_feedback_store,
    create_image_storage,
    create_ingestion_backends,
    create_memory_stores,
    create_object_store_from_settings,
    create_rate_limit_hook,
    create_semantic_cache,
    create_sparse_index,
)
from src.storage.sparse_index import (
    BM25SparseIndex,
    OpenSearchSparseIndex,
    SparseIndex,
)

__all__ = [
    "BM25SparseIndex",
    "IngestionTaskStore",
    "LocalObjectStore",
    "MinioObjectStore",
    "ObjectImageStorage",
    "ObjectStore",
    "OpenSearchSparseIndex",
    "PostgresConversationStore",
    "PostgresDocumentRegistry",
    "PostgresErrorMemory",
    "PostgresExecutor",
    "PostgresFeedbackStore",
    "PostgresIntegrityChecker",
    "PostgresKnowledgeMapMemory",
    "PostgresSessionMemory",
    "PostgresSkillMemory",
    "PostgresStudentProfileMemory",
    "PostgresUnavailableError",
    "SparseIndex",
    "build_llm_circuit_breaker_scope",
    "create_circuit_breaker",
    "create_conversation_store",
    "create_feedback_store",
    "create_image_storage",
    "create_ingestion_backends",
    "create_memory_stores",
    "create_object_store",
    "create_object_store_from_settings",
    "create_rate_limit_hook",
    "create_semantic_cache",
    "create_sparse_index",
]
