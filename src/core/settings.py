"""Configuration loading and validation for the Modular RAG MCP Server."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# ---------------------------------------------------------------------------
# Repo root & path resolution
# ---------------------------------------------------------------------------
# Anchored to this file's location: <repo>/src/core/settings.py → parents[2]
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Default absolute path to settings.yaml
DEFAULT_SETTINGS_PATH: Path = REPO_ROOT / "config" / "settings.yaml"


def resolve_path(relative: Union[str, Path]) -> Path:
    """Resolve a repo-relative path to an absolute path.

    If *relative* is already absolute it is returned as-is.  Otherwise
    it is resolved against :data:`REPO_ROOT`.

    >>> resolve_path("config/settings.yaml")  # doctest: +SKIP
    PosixPath('/home/user/Modular-RAG-MCP-Server/config/settings.yaml')
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


class SettingsError(ValueError):
    """Raised when settings validation fails."""


def _require_mapping(data: Dict[str, Any], key: str, path: str) -> Dict[str, Any]:
    value = data.get(key)
    if value is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    if not isinstance(value, dict):
        raise SettingsError(f"Expected mapping for field: {path}.{key}")
    return value


def _require_value(data: Dict[str, Any], key: str, path: str) -> Any:
    if key not in data or data.get(key) is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    return data[key]


def _require_str(data: Dict[str, Any], key: str, path: str) -> str:
    value = _require_value(data, key, path)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Expected non-empty string for field: {path}.{key}")
    return value


def _require_int(data: Dict[str, Any], key: str, path: str) -> int:
    value = _require_value(data, key, path)
    if not isinstance(value, int):
        raise SettingsError(f"Expected integer for field: {path}.{key}")
    return value


def _require_number(data: Dict[str, Any], key: str, path: str) -> float:
    value = _require_value(data, key, path)
    if not isinstance(value, (int, float)):
        raise SettingsError(f"Expected number for field: {path}.{key}")
    return float(value)


def _require_bool(data: Dict[str, Any], key: str, path: str) -> bool:
    value = _require_value(data, key, path)
    if not isinstance(value, bool):
        raise SettingsError(f"Expected boolean for field: {path}.{key}")
    return value


def _require_list(data: Dict[str, Any], key: str, path: str) -> List[Any]:
    value = _require_value(data, key, path)
    if not isinstance(value, list):
        raise SettingsError(f"Expected list for field: {path}.{key}")
    return value


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    # Azure/OpenAI-specific optional fields
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    # Ollama-specific optional fields
    base_url: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int
    # Azure-specific optional fields
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    # Ollama-specific optional fields
    base_url: Optional[str] = None


@dataclass(frozen=True)
class VectorStoreSettings:
    provider: str
    persist_directory: str
    collection_name: str


@dataclass(frozen=True)
class MilvusServiceSettings:
    uri: str
    mode: str = "lite"
    host: str = ""
    port: int = 19530
    token: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    db_name: Optional[str] = None
    dim: int = 768


@dataclass(frozen=True)
class RetrievalSettings:
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    rerank_enabled: bool = False
    rerank_top_k: int = 5
    mmr_enabled: bool = False
    mmr_lambda: float = 0.7
    query_rewrite_enabled: bool = False
    hyde_enabled: bool = False
    multi_query_enabled: bool = False
    contextual_enrichment: str = "off"
    embedding_cache_size: int = 4096
    dedup_enabled: bool = True
    dedup_threshold: int = 3
    min_score: float = 0.0
    post_dedup_enabled: bool = True


@dataclass(frozen=True)
class RerankSettings:
    enabled: bool
    provider: str
    model: str
    top_k: int


@dataclass(frozen=True)
class EvaluationSettings:
    enabled: bool
    provider: str
    metrics: List[str]


@dataclass(frozen=True)
class ObservabilitySettings:
    log_level: str
    trace_enabled: bool
    trace_file: str
    structured_logging: bool
    trace_sink: str = "file"
    trace_source: str = "auto"


@dataclass(frozen=True)
class PostgresSettings:
    enabled: bool = False
    dsn: str = ""


@dataclass(frozen=True)
class RedisSettings:
    enabled: bool = False
    url: str = ""
    ttl_seconds: int = 3600


@dataclass(frozen=True)
class RetryPolicySettings:
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 16.0


@dataclass(frozen=True)
class CircuitBreakerSettings:
    failure_threshold: int = 5
    cooldown_seconds: float = 30.0


@dataclass(frozen=True)
class LLMResilienceSettings:
    retry: RetryPolicySettings = field(default_factory=RetryPolicySettings)
    circuit_breaker: CircuitBreakerSettings = field(default_factory=CircuitBreakerSettings)


@dataclass(frozen=True)
class ObjectStoreSettings:
    provider: str = "local"
    local_root: str = "./data/object_store"
    endpoint: str = ""
    bucket: str = "modular-rag"
    access_key: str = ""
    secret_key: str = ""
    secure: bool = False
    uploads_prefix: str = "uploads"
    images_prefix: str = "images"
    context_prefix: str = "context"


@dataclass(frozen=True)
class SparseStoreSettings:
    provider: str = "bm25"
    index_dir: str = "./data/db/bm25"


@dataclass(frozen=True)
class OpenSearchSettings:
    hosts: List[str]
    index_prefix: str = "modular-rag"
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass(frozen=True)
class VisionLLMSettings:
    enabled: bool
    provider: str
    model: str
    max_image_size: int
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    base_url: Optional[str] = None


@dataclass(frozen=True)
class IngestionSettings:
    chunk_size: int
    chunk_overlap: int
    splitter: str
    batch_size: int
    chunk_refiner: Optional[Dict[str, Any]] = None  # 动态配置
    metadata_enricher: Optional[Dict[str, Any]] = None  # 动态配置


@dataclass(frozen=True)
class IngestionWorkerSettings:
    enabled: bool = False
    mode: str = "separate_process"
    poll_interval_seconds: int = 2
    lease_seconds: int = 120
    heartbeat_interval_seconds: int = 30
    max_attempts: int = 3
    dev_fallback_sync: bool = True


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    milvus: MilvusServiceSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    postgres: PostgresSettings
    redis: RedisSettings
    object_store: ObjectStoreSettings
    sparse_store: SparseStoreSettings
    opensearch: OpenSearchSettings
    llm_resilience: LLMResilienceSettings = field(default_factory=LLMResilienceSettings)
    ingestion: Optional[IngestionSettings] = None
    ingestion_worker: Optional[IngestionWorkerSettings] = None
    vision_llm: Optional[VisionLLMSettings] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        if not isinstance(data, dict):
            raise SettingsError("Settings root must be a mapping")

        llm = _require_mapping(data, "llm", "settings")
        embedding = _require_mapping(data, "embedding", "settings")
        vector_store = _require_mapping(data, "vector_store", "settings")
        retrieval = _require_mapping(data, "retrieval", "settings")
        rerank = _require_mapping(data, "rerank", "settings")
        evaluation = _require_mapping(data, "evaluation", "settings")
        observability = _require_mapping(data, "observability", "settings")
        milvus_cfg = vector_store.get("milvus", {}) if isinstance(vector_store.get("milvus", {}), dict) else {}
        postgres_cfg = data.get("postgres", {}) if isinstance(data.get("postgres", {}), dict) else {}
        redis_cfg = data.get("redis", {}) if isinstance(data.get("redis", {}), dict) else {}
        llm_resilience_cfg = {}
        if "llm_resilience" in data:
            llm_resilience_cfg = _require_mapping(data, "llm_resilience", "settings")
        retry_cfg = llm_resilience_cfg.get("retry", {})
        if retry_cfg and not isinstance(retry_cfg, dict):
            raise SettingsError("Expected mapping for field: llm_resilience.retry")
        circuit_breaker_cfg = llm_resilience_cfg.get("circuit_breaker", {})
        if circuit_breaker_cfg and not isinstance(circuit_breaker_cfg, dict):
            raise SettingsError("Expected mapping for field: llm_resilience.circuit_breaker")
        object_store_cfg = data.get("object_store", {}) if isinstance(data.get("object_store", {}), dict) else {}
        sparse_store_cfg = data.get("sparse_store", {}) if isinstance(data.get("sparse_store", {}), dict) else {}
        opensearch_cfg = data.get("opensearch", {}) if isinstance(data.get("opensearch", {}), dict) else {}

        ingestion_settings = None
        if "ingestion" in data:
            ingestion = _require_mapping(data, "ingestion", "settings")
            ingestion_settings = IngestionSettings(
                chunk_size=_require_int(ingestion, "chunk_size", "ingestion"),
                chunk_overlap=_require_int(ingestion, "chunk_overlap", "ingestion"),
                splitter=_require_str(ingestion, "splitter", "ingestion"),
                batch_size=_require_int(ingestion, "batch_size", "ingestion"),
                chunk_refiner=ingestion.get("chunk_refiner"),  # 可选配置
                metadata_enricher=ingestion.get("metadata_enricher"),  # 可选配置
            )

        vision_llm_settings = None
        if "vision_llm" in data:
            vision_llm = _require_mapping(data, "vision_llm", "settings")
            vision_llm_settings = VisionLLMSettings(
                enabled=_require_bool(vision_llm, "enabled", "vision_llm"),
                provider=_require_str(vision_llm, "provider", "vision_llm"),
                model=_require_str(vision_llm, "model", "vision_llm"),
                max_image_size=_require_int(vision_llm, "max_image_size", "vision_llm"),
                api_key=vision_llm.get("api_key"),
                api_version=vision_llm.get("api_version"),
                azure_endpoint=vision_llm.get("azure_endpoint"),
                deployment_name=vision_llm.get("deployment_name"),
                base_url=vision_llm.get("base_url"),
            )

        ingestion_worker_settings = None
        if "ingestion_worker" in data:
            ingestion_worker = _require_mapping(data, "ingestion_worker", "settings")
            ingestion_worker_settings = IngestionWorkerSettings(
                enabled=bool(ingestion_worker.get("enabled", False)),
                mode=str(ingestion_worker.get("mode", "separate_process")),
                poll_interval_seconds=int(ingestion_worker.get("poll_interval_seconds", 2)),
                lease_seconds=int(ingestion_worker.get("lease_seconds", 120)),
                heartbeat_interval_seconds=int(ingestion_worker.get("heartbeat_interval_seconds", 30)),
                max_attempts=int(ingestion_worker.get("max_attempts", 3)),
                dev_fallback_sync=bool(ingestion_worker.get("dev_fallback_sync", True)),
            )

        settings = cls(
            llm=LLMSettings(
                provider=_require_str(llm, "provider", "llm"),
                model=_require_str(llm, "model", "llm"),
                temperature=_require_number(llm, "temperature", "llm"),
                max_tokens=_require_int(llm, "max_tokens", "llm"),
                api_key=llm.get("api_key"),
                api_version=llm.get("api_version"),
                azure_endpoint=llm.get("azure_endpoint"),
                deployment_name=llm.get("deployment_name"),
                base_url=llm.get("base_url"),
            ),
            embedding=EmbeddingSettings(
                provider=_require_str(embedding, "provider", "embedding"),
                model=_require_str(embedding, "model", "embedding"),
                dimensions=_require_int(embedding, "dimensions", "embedding"),
                api_key=embedding.get("api_key"),
                api_version=embedding.get("api_version"),
                azure_endpoint=embedding.get("azure_endpoint"),
                deployment_name=embedding.get("deployment_name"),
                base_url=embedding.get("base_url"),
            ),
            vector_store=VectorStoreSettings(
                provider=_require_str(vector_store, "provider", "vector_store"),
                persist_directory=_require_str(vector_store, "persist_directory", "vector_store"),
                collection_name=_require_str(vector_store, "collection_name", "vector_store"),
            ),
            milvus=MilvusServiceSettings(
                uri=str(milvus_cfg.get("uri", "./data/db/milvus.db")),
                mode=str(milvus_cfg.get("mode", "lite")),
                host=str(milvus_cfg.get("host", "")),
                port=int(milvus_cfg.get("port", 19530)),
                token=milvus_cfg.get("token"),
                user=milvus_cfg.get("user"),
                password=milvus_cfg.get("password"),
                db_name=milvus_cfg.get("db_name"),
                dim=int(milvus_cfg.get("dim", 768)),
            ),
            retrieval=RetrievalSettings(
                dense_top_k=_require_int(retrieval, "dense_top_k", "retrieval"),
                sparse_top_k=_require_int(retrieval, "sparse_top_k", "retrieval"),
                fusion_top_k=_require_int(retrieval, "fusion_top_k", "retrieval"),
                rrf_k=_require_int(retrieval, "rrf_k", "retrieval"),
                dense_weight=float(retrieval.get("dense_weight", 1.0)),
                sparse_weight=float(retrieval.get("sparse_weight", 1.0)),
                rerank_enabled=bool(retrieval.get("rerank_enabled", False)),
                rerank_top_k=int(retrieval.get("rerank_top_k", 5)),
                mmr_enabled=bool(retrieval.get("mmr_enabled", False)),
                mmr_lambda=float(retrieval.get("mmr_lambda", 0.7)),
                query_rewrite_enabled=bool(retrieval.get("query_rewrite_enabled", False)),
                hyde_enabled=bool(retrieval.get("hyde_enabled", False)),
                multi_query_enabled=bool(retrieval.get("multi_query_enabled", False)),
                contextual_enrichment=str(retrieval.get("contextual_enrichment", "off")),
                embedding_cache_size=int(retrieval.get("embedding_cache_size", 4096)),
                dedup_enabled=bool(retrieval.get("dedup_enabled", True)),
                dedup_threshold=int(retrieval.get("dedup_threshold", 3)),
                min_score=float(retrieval.get("min_score", 0.0)),
                post_dedup_enabled=bool(retrieval.get("post_dedup_enabled", True)),
            ),
            rerank=RerankSettings(
                enabled=_require_bool(rerank, "enabled", "rerank"),
                provider=_require_str(rerank, "provider", "rerank"),
                model=_require_str(rerank, "model", "rerank"),
                top_k=_require_int(rerank, "top_k", "rerank"),
            ),
            evaluation=EvaluationSettings(
                enabled=_require_bool(evaluation, "enabled", "evaluation"),
                provider=_require_str(evaluation, "provider", "evaluation"),
                metrics=[str(item) for item in _require_list(evaluation, "metrics", "evaluation")],
            ),
            observability=ObservabilitySettings(
                log_level=_require_str(observability, "log_level", "observability"),
                trace_enabled=_require_bool(observability, "trace_enabled", "observability"),
                trace_file=_require_str(observability, "trace_file", "observability"),
                structured_logging=_require_bool(observability, "structured_logging", "observability"),
                trace_sink=str(observability.get("trace_sink", "file")),
                trace_source=str(observability.get("trace_source", "auto")),
            ),
            postgres=PostgresSettings(
                enabled=bool(postgres_cfg.get("enabled", False)),
                dsn=str(postgres_cfg.get("dsn", "")),
            ),
            redis=RedisSettings(
                enabled=bool(redis_cfg.get("enabled", False)),
                url=str(redis_cfg.get("url", "")),
                ttl_seconds=int(redis_cfg.get("ttl_seconds", 3600)),
            ),
            llm_resilience=LLMResilienceSettings(
                retry=RetryPolicySettings(
                    max_retries=int(retry_cfg.get("max_retries", 3)),
                    base_delay_seconds=float(retry_cfg.get("base_delay_seconds", 1.0)),
                    max_delay_seconds=float(retry_cfg.get("max_delay_seconds", 16.0)),
                ),
                circuit_breaker=CircuitBreakerSettings(
                    failure_threshold=int(circuit_breaker_cfg.get("failure_threshold", 5)),
                    cooldown_seconds=float(circuit_breaker_cfg.get("cooldown_seconds", 30.0)),
                ),
            ),
            object_store=ObjectStoreSettings(
                provider=str(object_store_cfg.get("provider", "local")),
                local_root=str(object_store_cfg.get("local_root", "./data/object_store")),
                endpoint=str(object_store_cfg.get("endpoint", "")),
                bucket=str(object_store_cfg.get("bucket", "modular-rag")),
                access_key=str(object_store_cfg.get("access_key", "")),
                secret_key=str(object_store_cfg.get("secret_key", "")),
                secure=bool(object_store_cfg.get("secure", False)),
                uploads_prefix=str(object_store_cfg.get("uploads_prefix", "uploads")),
                images_prefix=str(object_store_cfg.get("images_prefix", "images")),
                context_prefix=str(object_store_cfg.get("context_prefix", "context")),
            ),
            sparse_store=SparseStoreSettings(
                provider=str(sparse_store_cfg.get("provider", "bm25")),
                index_dir=str(sparse_store_cfg.get("index_dir", "./data/db/bm25")),
            ),
            opensearch=OpenSearchSettings(
                hosts=[str(item) for item in opensearch_cfg.get("hosts", ["http://localhost:9200"])],
                index_prefix=str(opensearch_cfg.get("index_prefix", "modular-rag")),
                username=opensearch_cfg.get("username"),
                password=opensearch_cfg.get("password"),
            ),
            ingestion=ingestion_settings,
            ingestion_worker=ingestion_worker_settings,
            vision_llm=vision_llm_settings,
        )

        return settings


def validate_settings(settings: Settings) -> None:
    """Validate settings and raise SettingsError if invalid."""

    if not settings.llm.provider:
        raise SettingsError("Missing required field: llm.provider")
    if not settings.embedding.provider:
        raise SettingsError("Missing required field: embedding.provider")
    if not settings.vector_store.provider:
        raise SettingsError("Missing required field: vector_store.provider")
    if not settings.retrieval.rrf_k:
        raise SettingsError("Missing required field: retrieval.rrf_k")
    if not settings.rerank.provider:
        raise SettingsError("Missing required field: rerank.provider")
    if not settings.evaluation.provider:
        raise SettingsError("Missing required field: evaluation.provider")
    if not settings.observability.log_level:
        raise SettingsError("Missing required field: observability.log_level")


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from a YAML file and validate required fields.

    Args:
        path: Path to settings YAML.  Defaults to
            ``<repo>/config/settings.yaml`` (absolute, CWD-independent).
    """
    resolved_input = path if path is not None else os.environ.get("MODULAR_RAG_SETTINGS_PATH", DEFAULT_SETTINGS_PATH)
    settings_path = Path(resolved_input)
    if not settings_path.is_absolute():
        settings_path = resolve_path(settings_path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    settings = Settings.from_dict(data or {})
    validate_settings(settings)
    object.__setattr__(settings, "_raw", data or {})
    return settings
