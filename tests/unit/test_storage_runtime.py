"""Tests for storage runtime factories."""

from __future__ import annotations

import textwrap
from pathlib import Path

from src.agent.config import load_agent_config, load_memory_config
from src.agent.conversation import FileConversationStore
from src.core.cache.semantic_cache import SemanticCache
from src.core.settings import load_settings
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.storage.object_store import LocalObjectStore
from src.storage.runtime import (
    create_conversation_store,
    create_feedback_store,
    create_image_storage,
    create_ingestion_backends,
    create_memory_stores,
    create_object_store_from_settings,
    create_semantic_cache,
    create_sparse_index,
)
from src.storage.sparse_index import BM25SparseIndex


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def _build_settings(tmp_path: Path):
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(
        settings_path,
        """
        llm:
          provider: openai
          model: gpt-4o-mini
          temperature: 0.0
          max_tokens: 1024
        embedding:
          provider: openai
          model: text-embedding-3-small
          dimensions: 1536
        vector_store:
          provider: chroma
          persist_directory: ./data/db/chroma
          collection_name: knowledge_hub
        retrieval:
          dense_top_k: 20
          sparse_top_k: 20
          fusion_top_k: 10
          rrf_k: 60
        rerank:
          enabled: false
          provider: none
          model: cross-encoder/ms-marco-MiniLM-L-6-v2
          top_k: 5
        evaluation:
          enabled: false
          provider: custom
          metrics: [hit_rate]
        observability:
          log_level: INFO
          trace_enabled: true
          trace_file: ./logs/traces.jsonl
          structured_logging: true
        agent:
          conversation_store_dir: data/conversations
        memory:
          db_dir: data/memory
          profile_enabled: true
          error_memory_enabled: true
          knowledge_map_enabled: true
          skill_memory_enabled: true
          session_memory_enabled: true
        semantic_cache:
          provider: memory
        """,
    )
    return settings_path, load_settings(settings_path)


def test_runtime_factories_default_to_local_backends(tmp_path: Path) -> None:
    settings_path, core_settings = _build_settings(tmp_path)
    raw_settings = {
        "agent": {"conversation_store_dir": "data/conversations"},
        "memory": {"db_dir": "data/memory", "profile_enabled": True, "error_memory_enabled": True, "knowledge_map_enabled": True, "skill_memory_enabled": True, "session_memory_enabled": True},
        "semantic_cache": {"provider": "memory"},
    }
    agent_cfg = load_agent_config(raw_settings)
    memory_cfg = load_memory_config(raw_settings)

    conv_store = create_conversation_store(raw_settings, agent_cfg)
    memory_stores = create_memory_stores(memory_cfg, raw_settings)
    feedback_store = create_feedback_store(raw_settings)
    object_store = create_object_store_from_settings(core_settings)
    sparse_index = create_sparse_index(core_settings, collection="test")
    image_storage = create_image_storage(core_settings, collection="test")
    ingestion_backends = create_ingestion_backends(core_settings, collection="test")
    semantic_cache = create_semantic_cache(
        core_settings,
        embedding_fn=lambda text: [0.1, 0.2, 0.3],
        similarity_threshold=0.9,
        ttl_seconds=10,
        max_size=10,
    )

    assert isinstance(conv_store, FileConversationStore)
    assert memory_stores["profile"] is not None
    assert feedback_store is not None
    assert isinstance(object_store, LocalObjectStore)
    assert isinstance(sparse_index, BM25SparseIndex)
    assert isinstance(image_storage, ImageStorage)
    assert isinstance(ingestion_backends.integrity_checker, SQLiteIntegrityChecker)
    assert isinstance(semantic_cache, SemanticCache)
