"""Milvus Lite vector store implementation (embedded / local mode).

Uses ``pymilvus.MilvusClient`` with a local SQLite-based file to avoid
any external server dependency.  Provides the same interface as ChromaStore.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)

try:
    from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False

_DEFAULT_DIM = 768
_DEFAULT_URI = "./data/db/milvus.db"


class MilvusStore(BaseVectorStore):
    """Milvus Lite (embedded) vector store implementation."""

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        if not PYMILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is required for MilvusStore. "
                "Install with: pip install pymilvus"
            )

        vs_cfg = settings.vector_store
        self.collection_name = kwargs.get(
            "collection_name",
            getattr(vs_cfg, "collection_name", "knowledge_hub"),
        )

        raw = getattr(settings, "_raw", None) or {}
        milvus_cfg = raw.get("vector_store", {}).get("milvus", {}) if isinstance(raw, dict) else {}
        self._dim = int(milvus_cfg.get("dim", _DEFAULT_DIM))

        from src.core.settings import resolve_path
        uri_str = milvus_cfg.get("uri", _DEFAULT_URI)
        uri_path = resolve_path(uri_str)
        uri_path.parent.mkdir(parents=True, exist_ok=True)
        self._uri = str(uri_path)

        logger.info(f"Initializing MilvusStore: uri={self._uri}, collection={self.collection_name}")

        self.client = MilvusClient(uri=self._uri)
        self._ensure_collection(self.collection_name)

        logger.info("MilvusStore initialized successfully")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self, name: str) -> None:
        if self.client.has_collection(name):
            return
        schema = self._build_schema()
        self.client.create_collection(
            collection_name=name,
            schema=schema,
        )
        self.client.create_index(
            collection_name=name,
            field_name="vector",
            index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 32, "efConstruction": 200}},
        )
        logger.info(f"Created Milvus collection '{name}' with HNSW index")

    def _build_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="has_formula", dtype=DataType.BOOL),
            FieldSchema(name="is_parent", dtype=DataType.BOOL),
            FieldSchema(name="parent_chunk_id", dtype=DataType.VARCHAR, max_length=512),
        ]
        return CollectionSchema(fields=fields, description="RAG knowledge store")

    def get_or_switch_collection(self, collection_name: str) -> None:
        self._ensure_collection(collection_name)
        self.collection_name = collection_name
        logger.info(f"Switched to Milvus collection '{collection_name}'")

    def list_collections(self) -> List[str]:
        return self.client.list_collections()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.validate_records(records)

        rows = []
        for r in records:
            meta = r.get("metadata", {})
            rows.append({
                "id": str(r["id"]),
                "vector": r["vector"],
                "text": meta.get("text", str(r["id"])),
                "metadata_json": json.dumps(meta, ensure_ascii=False, default=str),
                "source_path": str(meta.get("source_path", "")),
                "content_type": str(meta.get("content_type", "")),
                "has_formula": bool(meta.get("has_formula", False)),
                "is_parent": bool(meta.get("is_parent", False)),
                "parent_chunk_id": str(meta.get("parent_chunk_id", "")),
            })

        self.client.upsert(collection_name=self.collection_name, data=rows)
        logger.debug(f"Upserted {len(rows)} records to Milvus")

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        self.validate_query_vector(vector, top_k)

        search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
        filter_expr = self._build_filter_expr(filters) if filters else ""

        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=top_k,
            output_fields=["text", "metadata_json", "source_path", "content_type",
                           "has_formula", "is_parent", "parent_chunk_id"],
            search_params=search_params,
            filter=filter_expr or "",
        )

        output: List[Dict[str, Any]] = []
        for hit in results[0] if results else []:
            entity = hit.get("entity", {})
            meta_str = entity.get("metadata_json", "{}")
            try:
                meta = json.loads(meta_str)
            except (json.JSONDecodeError, TypeError):
                meta = {}
            output.append({
                "id": hit["id"],
                "score": float(hit.get("distance", 0.0)),
                "text": entity.get("text", ""),
                "metadata": meta,
            })
        return output

    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if not ids:
            raise ValueError("IDs list cannot be empty")
        self.client.delete(collection_name=self.collection_name, ids=ids)
        logger.debug(f"Deleted {len(ids)} records from Milvus")

    def clear(
        self,
        collection_name: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        name = collection_name or self.collection_name
        if self.client.has_collection(name):
            self.client.drop_collection(name)
        self._ensure_collection(name)
        logger.info(f"Cleared Milvus collection '{name}'")

    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if not ids:
            raise ValueError("IDs list cannot be empty")

        results = self.client.get(
            collection_name=self.collection_name,
            ids=ids,
            output_fields=["text", "metadata_json"],
        )

        id_map: Dict[str, Dict[str, Any]] = {}
        for r in results:
            meta_str = r.get("metadata_json", "{}")
            try:
                meta = json.loads(meta_str)
            except (json.JSONDecodeError, TypeError):
                meta = {}
            id_map[r["id"]] = {
                "id": r["id"],
                "text": r.get("text", ""),
                "metadata": meta,
            }

        return [id_map.get(i, {}) for i in ids]

    def delete_by_metadata(
        self,
        filter_dict: Dict[str, Any],
        trace: Optional[Any] = None,
    ) -> int:
        if not filter_dict:
            raise ValueError("filter_dict cannot be empty")
        expr = self._build_filter_expr(filter_dict)
        if not expr:
            return 0
        results = self.client.query(
            collection_name=self.collection_name,
            filter=expr,
            output_fields=["id"],
        )
        ids = [r["id"] for r in results]
        if ids:
            self.client.delete(collection_name=self.collection_name, ids=ids)
        return len(ids)

    def get_collection_stats(self) -> Dict[str, Any]:
        stats = self.client.get_collection_stats(self.collection_name)
        return {
            "count": stats.get("row_count", 0),
            "name": self.collection_name,
        }

    # ------------------------------------------------------------------
    # Filter expression builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter_expr(filters: Dict[str, Any]) -> str:
        """Build a Milvus boolean filter expression from a dict."""
        if not filters:
            return ""

        parts: List[str] = []
        for key, value in filters.items():
            if isinstance(value, dict):
                for op, val in value.items():
                    milvus_op = {"$eq": "==", "$ne": "!=", "$gt": ">", "$gte": ">=",
                                 "$lt": "<", "$lte": "<=", "$in": "in"}.get(op, "==")
                    if milvus_op == "in":
                        parts.append(f'{key} in {val}')
                    elif isinstance(val, str):
                        parts.append(f'{key} {milvus_op} "{val}"')
                    else:
                        parts.append(f'{key} {milvus_op} {val}')
            elif isinstance(value, bool):
                parts.append(f'{key} == {"true" if value else "false"}')
            elif isinstance(value, str):
                parts.append(f'{key} == "{value}"')
            else:
                parts.append(f'{key} == {value}')

        return " and ".join(parts)
