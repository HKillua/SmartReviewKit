"""Tests for settings loading and validation."""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from src.core.settings import SettingsError, load_settings


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def test_load_settings_success(tmp_path: Path) -> None:
    config = """
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
      metrics:
        - hit_rate
        - mrr
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    ingestion:
      chunk_size: 1000
      chunk_overlap: 200
      splitter: recursive
      batch_size: 100
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    settings = load_settings(settings_path)

    assert settings.llm.provider == "openai"
    assert settings.embedding.dimensions == 1536
    assert settings.vector_store.collection_name == "knowledge_hub"
    assert settings.retrieval.rrf_k == 60
    assert settings.rerank.provider == "none"
    assert settings.evaluation.metrics == ["hit_rate", "mrr"]
    assert settings.observability.log_level == "INFO"
    assert settings.ingestion is not None
    assert settings.llm_resilience.retry.max_retries == 3
    assert settings.llm_resilience.circuit_breaker.cooldown_seconds == 30.0
    assert settings.grounding.mode == "balanced"
    assert settings.grounding.low_evidence_threshold == 0.4
    assert settings.retrieval.query_rewrite_policy == "followup_only"
    assert settings.retrieval.query_rewrite_enabled is True
    assert settings.retrieval.post_rerank_min_score == 0.0
    assert settings.retrieval.empty_result_fallback_enabled is True


def test_missing_required_field_raises_error(tmp_path: Path) -> None:
    config = """
    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.0
      max_tokens: 1024
    embedding:
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
      metrics:
        - hit_rate
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    with pytest.raises(SettingsError, match="embedding.provider"):
        load_settings(settings_path)


def test_load_settings_with_production_storage_sections(tmp_path: Path) -> None:
    config = """
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
      provider: milvus
      persist_directory: ./data/db/chroma
      collection_name: knowledge_hub
      milvus:
        mode: service
        uri: http://milvus:19530
        dim: 1536
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
      metrics:
        - hit_rate
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    postgres:
      enabled: true
      dsn: postgresql://postgres:postgres@localhost:5432/modular_rag
    redis:
      enabled: true
      url: redis://localhost:6379/0
      ttl_seconds: 120
    object_store:
      provider: minio
      endpoint: localhost:9000
      bucket: modular-rag
      access_key: minio
      secret_key: secret
      secure: false
    sparse_store:
      provider: opensearch
      index_dir: ./data/db/bm25
    opensearch:
      hosts:
        - http://localhost:9200
      index_prefix: modular-rag
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    settings = load_settings(settings_path)

    assert settings.milvus.mode == "service"
    assert settings.postgres.enabled is True
    assert settings.redis.ttl_seconds == 120
    assert settings.object_store.provider == "minio"
    assert settings.sparse_store.provider == "opensearch"
    assert settings.opensearch.index_prefix == "modular-rag"


def test_load_settings_with_llm_resilience_overrides(tmp_path: Path) -> None:
    config = """
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
      metrics:
        - hit_rate
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    llm_resilience:
      retry:
        max_retries: 7
        base_delay_seconds: 0.5
        max_delay_seconds: 9.0
      circuit_breaker:
        failure_threshold: 8
        cooldown_seconds: 45.0
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    settings = load_settings(settings_path)

    assert settings.llm_resilience.retry.max_retries == 7
    assert settings.llm_resilience.retry.base_delay_seconds == 0.5
    assert settings.llm_resilience.retry.max_delay_seconds == 9.0
    assert settings.llm_resilience.circuit_breaker.failure_threshold == 8
    assert settings.llm_resilience.circuit_breaker.cooldown_seconds == 45.0


def test_load_settings_with_query_rewrite_policy_and_grounding_overrides(tmp_path: Path) -> None:
    config = """
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
      query_rewrite_policy: followup_only
    grounding:
      mode: strict
      low_evidence_threshold: 0.55
    rerank:
      enabled: false
      provider: none
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k: 5
    evaluation:
      enabled: false
      provider: custom
      metrics:
        - hit_rate
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    settings = load_settings(settings_path)

    assert settings.retrieval.query_rewrite_policy == "followup_only"
    assert settings.retrieval.query_rewrite_enabled is True
    assert settings.grounding.mode == "strict"
    assert settings.grounding.low_evidence_threshold == 0.55


def test_load_settings_with_post_rerank_threshold_and_empty_result_fallback(tmp_path: Path) -> None:
    config = """
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
      min_score: 0.15
      post_rerank_min_score: 0.33
      empty_result_fallback_enabled: false
    rerank:
      enabled: false
      provider: none
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      top_k: 5
    evaluation:
      enabled: false
      provider: custom
      metrics:
        - hit_rate
    observability:
      log_level: INFO
      trace_enabled: true
      trace_file: ./logs/traces.jsonl
      structured_logging: true
    """
    settings_path = tmp_path / "settings.yaml"
    _write_yaml(settings_path, config)

    settings = load_settings(settings_path)

    assert settings.retrieval.min_score == 0.15
    assert settings.retrieval.post_rerank_min_score == 0.33
    assert settings.retrieval.empty_result_fallback_enabled is False
