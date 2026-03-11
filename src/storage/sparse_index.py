"""Sparse retrieval adapters for local BM25 and production OpenSearch."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ingestion.storage.bm25_indexer import BM25Indexer

logger = logging.getLogger(__name__)
_DOC_HASH_PATTERN = re.compile(r"^doc_([0-9a-f]{16,64})(?:_|$)")

try:
    from opensearchpy import OpenSearch
    from opensearchpy.helpers import bulk as opensearch_bulk

    OPENSEARCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional in dev
    OpenSearch = None
    opensearch_bulk = None
    OPENSEARCH_AVAILABLE = False


class SparseIndex(ABC):
    """Abstract sparse-index interface used by retrieval and ingestion."""

    @abstractmethod
    def ensure_collection_ready(self, collection: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build(
        self,
        term_stats: List[Dict[str, Any]],
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def remove_document(self, doc_id: str, collection: str = "default") -> bool:
        raise NotImplementedError


class BM25SparseIndex(SparseIndex):
    """Adapter around the existing local BM25Indexer."""

    def __init__(self, index_dir: str = "data/db/bm25") -> None:
        self._indexer = BM25Indexer(index_dir=index_dir)

    @property
    def indexer(self) -> BM25Indexer:
        return self._indexer

    def ensure_collection_ready(self, collection: str) -> bool:
        path = self._indexer._get_index_path(collection)
        legacy = self._indexer._get_legacy_index_path(collection)
        if path.exists() or legacy.exists():
            return self._indexer.load(collection)
        return False

    def build(
        self,
        term_stats: List[Dict[str, Any]],
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> None:
        self._indexer.build(term_stats, collection=collection, trace=trace)

    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        if not self.ensure_collection_ready(collection):
            return []
        return self._indexer.query(query_terms=query_terms, top_k=top_k, trace=trace)

    def remove_document(self, doc_id: str, collection: str = "default") -> bool:
        return self._indexer.remove_document(doc_id, collection)


class OpenSearchSparseIndex(SparseIndex):
    """OpenSearch-backed sparse index for production BM25 retrieval."""

    def __init__(
        self,
        *,
        hosts: List[str],
        index_prefix: str,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        if not OPENSEARCH_AVAILABLE:
            raise ImportError(
                "opensearch-py is required for OpenSearchSparseIndex. "
                "Install with: pip install opensearch-py"
            )
        kwargs: Dict[str, Any] = {"hosts": hosts}
        if username:
            kwargs["http_auth"] = (username, password or "")
        self._client = OpenSearch(**kwargs)
        self._index_prefix = index_prefix

    def _index_name(self, collection: str) -> str:
        safe = collection.replace("/", "-").replace("_", "-")
        return f"{self._index_prefix}-{safe}-chunks"

    @staticmethod
    def _extract_doc_hash(chunk_id: str) -> str:
        match = _DOC_HASH_PATTERN.match(chunk_id or "")
        if match:
            return match.group(1)
        if "_chunk" in chunk_id:
            return chunk_id.split("_chunk", 1)[0]
        return chunk_id

    def ensure_collection_ready(self, collection: str) -> bool:
        index_name = self._index_name(collection)
        if self._client.indices.exists(index=index_name):
            return True

        body = {
            "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "doc_hash": {"type": "keyword"},
                    "collection": {"type": "keyword"},
                    "text": {"type": "text"},
                    "doc_length": {"type": "integer"},
                }
            },
        }
        self._client.indices.create(index=index_name, body=body)
        return True

    def build(
        self,
        term_stats: List[Dict[str, Any]],
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> None:
        if not term_stats:
            return
        index_name = self._index_name(collection)
        self.ensure_collection_ready(collection)

        actions = []
        for stat in term_stats:
            tokens: list[str] = []
            for term, freq in stat.get("term_frequencies", {}).items():
                tokens.extend([term] * int(freq))
            chunk_id = stat["chunk_id"]
            doc_hash = str(stat.get("doc_hash") or self._extract_doc_hash(chunk_id))
            doc_id = doc_hash
            actions.append(
                {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": chunk_id,
                    "_source": {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "doc_hash": doc_hash,
                        "collection": collection,
                        "text": " ".join(tokens),
                        "doc_length": int(stat.get("doc_length", 0)),
                    },
                }
            )
        if actions:
            opensearch_bulk(self._client, actions, refresh=True)

    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        if not query_terms:
            return []
        if not self.ensure_collection_ready(collection):
            return []
        index_name = self._index_name(collection)
        query_text = " ".join(query_terms)
        response = self._client.search(
            index=index_name,
            body={
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [{"match": {"text": {"query": query_text}}}],
                    }
                },
            },
        )
        hits = response.get("hits", {}).get("hits", [])
        return [
            {"chunk_id": hit["_source"]["chunk_id"], "score": float(hit["_score"])}
            for hit in hits
        ]

    def remove_document(self, doc_id: str, collection: str = "default") -> bool:
        index_name = self._index_name(collection)
        if not self._client.indices.exists(index=index_name):
            return False
        legacy_chunk_prefix = f"doc_{doc_id[:16]}" if doc_id and len(doc_id) > 16 else f"doc_{doc_id}"
        response = self._client.delete_by_query(
            index=index_name,
            body={
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"doc_hash": doc_id}},
                            {"prefix": {"doc_id": doc_id}},
                            {"prefix": {"chunk_id": legacy_chunk_prefix}},
                            {"prefix": {"chunk_id": f"doc_{doc_id}"}} if doc_id and not doc_id.startswith("doc_") else {"prefix": {"chunk_id": doc_id}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            },
            refresh=True,
        )
        return int(response.get("deleted", 0)) > 0


def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
    if hasattr(cfg, name):
        value = getattr(cfg, name)
        return default if value is None else value
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return default


def create_sparse_index(settings: Any, collection: str = "default") -> SparseIndex:
    """Create sparse index backend from settings."""
    cfg = getattr(settings, "sparse_store", settings)
    provider = _cfg_value(cfg, "provider", "bm25")
    if provider == "opensearch":
        os_cfg = getattr(settings, "opensearch", None) if hasattr(settings, "opensearch") else settings.get("opensearch", {})
        return OpenSearchSparseIndex(
            hosts=list(_cfg_value(os_cfg, "hosts", ["http://localhost:9200"])),
            index_prefix=str(_cfg_value(os_cfg, "index_prefix", "modular-rag")),
            username=_cfg_value(os_cfg, "username", None),
            password=_cfg_value(os_cfg, "password", None),
        )
    index_dir = _cfg_value(cfg, "index_dir", "data/db/bm25")
    return BM25SparseIndex(index_dir=str(Path(index_dir)))
