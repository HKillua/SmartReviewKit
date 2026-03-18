"""Milvus vector store implementation for both lite and service modes."""

from __future__ import annotations

import hashlib
import json
import logging
import re
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
_VALID_COLLECTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_COLLECTION_NAME = 255


class MilvusStore(BaseVectorStore):
    """Milvus vector store implementation for local lite and remote service modes."""

    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        if not PYMILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is required for MilvusStore. "
                "Install with: pip install pymilvus"
            )

        vs_cfg = settings.vector_store
        requested_collection_name = kwargs.get(
            "collection_name",
            getattr(vs_cfg, "collection_name", "knowledge_hub"),
        )
        self.requested_collection_name = str(requested_collection_name or "knowledge_hub")
        self.collection_name = self._normalize_collection_name(self.requested_collection_name)
        self._collection_aliases: Dict[str, str] = {self.collection_name: self.requested_collection_name}
        self._loaded_collections: set[str] = set()
        if self.collection_name != self.requested_collection_name:
            logger.info(
                "Normalized Milvus collection name '%s' -> '%s'",
                self.requested_collection_name,
                self.collection_name,
            )

        milvus_cfg = getattr(settings, "milvus", None)
        self._dim = int(getattr(milvus_cfg, "dim", _DEFAULT_DIM))
        self._load_timeout_s = float(getattr(milvus_cfg, "load_timeout_s", 30.0))
        mode = getattr(milvus_cfg, "mode", "lite")

        from src.core.settings import resolve_path

        if mode == "service":
            uri = getattr(milvus_cfg, "uri", "") or ""
            if uri:
                client_kwargs: Dict[str, Any] = {"uri": uri}
            else:
                host = getattr(milvus_cfg, "host", "")
                port = int(getattr(milvus_cfg, "port", 19530))
                client_kwargs = {"uri": f"http://{host}:{port}"}
            token = getattr(milvus_cfg, "token", None)
            user = getattr(milvus_cfg, "user", None)
            password = getattr(milvus_cfg, "password", None)
            db_name = getattr(milvus_cfg, "db_name", None)
            if token:
                client_kwargs["token"] = token
            elif user:
                client_kwargs["user"] = user
                client_kwargs["password"] = password or ""
            if db_name:
                client_kwargs["db_name"] = db_name
            self._uri = str(client_kwargs["uri"])
            logger.info("Initializing MilvusStore in service mode: uri=%s, collection=%s", self._uri, self.collection_name)
            self.client = MilvusClient(**client_kwargs)
        else:
            uri_str = getattr(milvus_cfg, "uri", _DEFAULT_URI)
            uri_path = resolve_path(uri_str)
            uri_path.parent.mkdir(parents=True, exist_ok=True)
            self._uri = str(uri_path)
            logger.info("Initializing MilvusStore in lite mode: uri=%s, collection=%s", self._uri, self.collection_name)
            self.client = MilvusClient(uri=self._uri)
        self._ensure_collection(self.collection_name)

        logger.info("MilvusStore initialized successfully")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_collection_name(name: str) -> str:
        raw = str(name or "knowledge_hub").strip() or "knowledge_hub"
        if _VALID_COLLECTION_NAME.fullmatch(raw) and len(raw) <= _MAX_COLLECTION_NAME:
            return raw

        sanitized = re.sub(r"[^A-Za-z0-9_]", "_", raw)
        if not sanitized:
            sanitized = "knowledge_hub"
        if not re.match(r"^[A-Za-z_]", sanitized):
            sanitized = f"c_{sanitized}"
        sanitized = sanitized[:200].rstrip("_") or "knowledge_hub"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        max_base_len = _MAX_COLLECTION_NAME - len(digest) - 1
        base = sanitized[:max_base_len].rstrip("_") or "knowledge_hub"
        return f"{base}_{digest}"

    def _ensure_collection(self, name: str) -> None:
        if self.client.has_collection(name):
            self._ensure_index(name)
            self._ensure_loaded(name)
            return
        schema = self._build_schema()
        self.client.create_collection(
            collection_name=name,
            schema=schema,
        )
        self._ensure_index(name)
        self._ensure_loaded(name)
        logger.info(f"Created Milvus collection '{name}' with HNSW index")

    def _ensure_index(self, name: str) -> None:
        existing = self.client.list_indexes(name)
        if existing:
            return
        index_params = self.client.prepare_index_params(
            field_name="vector",
            index_name="vector_idx",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 32, "efConstruction": 200},
        )
        self.client.create_index(
            collection_name=name,
            index_params=index_params,
        )

    def _ensure_loaded(self, name: Optional[str] = None) -> None:
        collection_name = name or self.collection_name
        if collection_name in self._loaded_collections:
            return
        try:
            state = self.client.get_load_state(collection_name)
        except Exception:
            state = {}
        if self._is_loaded_state(state):
            self._loaded_collections.add(collection_name)
            return
        self.client.load_collection(
            collection_name=collection_name,
            timeout=self._load_timeout_s,
        )
        self._loaded_collections.add(collection_name)

    @staticmethod
    def _is_loaded_state(state: Any) -> bool:
        if not isinstance(state, dict):
            return False
        value = state.get("state")
        if value is None:
            return False
        return str(value).endswith("Loaded")

    def _build_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="source_ref", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="doc_hash", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="has_formula", dtype=DataType.BOOL),
            FieldSchema(name="is_parent", dtype=DataType.BOOL),
            FieldSchema(name="parent_chunk_id", dtype=DataType.VARCHAR, max_length=512),
        ]
        return CollectionSchema(fields=fields, description="RAG knowledge store")

    def get_or_switch_collection(self, collection_name: str) -> None:
        requested = str(collection_name or "knowledge_hub")
        normalized = self._normalize_collection_name(requested)
        self._ensure_collection(normalized)
        self.collection_name = normalized
        self.requested_collection_name = requested
        self._collection_aliases[normalized] = requested
        self._ensure_loaded(normalized)
        if normalized != requested:
            logger.info("Switched to Milvus collection '%s' (physical='%s')", requested, normalized)
        else:
            logger.info("Switched to Milvus collection '%s'", requested)

    def list_collections(self) -> List[str]:
        return [self._collection_aliases.get(name, name) for name in self.client.list_collections()]

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
        self._ensure_loaded(self.collection_name)

        rows = []
        for r in records:
            meta = r.get("metadata", {})
            rows.append({
                "id": str(r["id"]),
                "vector": r["vector"],
                "text": meta.get("text", str(r["id"])),
                "metadata_json": json.dumps(meta, ensure_ascii=False, default=str),
                "source_path": str(meta.get("source_path", "")),
                "source_ref": str(meta.get("source_ref", "")),
                "source_type": str(meta.get("source_type", "")),
                "content_type": str(meta.get("content_type", "")),
                "doc_hash": str(meta.get("doc_hash", "")),
                "page_num": int(meta.get("page_num", meta.get("page", 0)) or 0),
                "has_formula": bool(meta.get("has_formula", False)),
                "is_parent": bool(meta.get("is_parent", False)),
                "parent_chunk_id": str(meta.get("parent_chunk_id", "")),
            })

        self.client.upsert(collection_name=self.collection_name, data=rows)
        self.client.flush(self.collection_name)
        self._ensure_loaded(self.collection_name)
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
        self._ensure_loaded(self.collection_name)

        search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
        filter_expr = self._build_filter_expr(filters) if filters else ""

        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=top_k,
            output_fields=["text", "metadata_json", "source_path", "source_ref", "source_type", "content_type",
                           "doc_hash", "page_num",
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
        self.client.flush(self.collection_name)
        self._ensure_loaded(self.collection_name)
        logger.debug(f"Deleted {len(ids)} records from Milvus")

    def clear(
        self,
        collection_name: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        requested = collection_name or self.requested_collection_name
        name = self._normalize_collection_name(requested)
        if self.client.has_collection(name):
            self.client.drop_collection(name)
            self._loaded_collections.discard(name)
        self._ensure_collection(name)
        self._ensure_loaded(name)
        self.collection_name = name
        self.requested_collection_name = requested
        self._collection_aliases[name] = requested
        logger.info("Cleared Milvus collection '%s' (physical='%s')", requested, name)

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
            output_fields=["text", "metadata_json", "source_path", "source_ref", "source_type", "content_type", "doc_hash", "page_num"],
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

    def list_by_metadata(self, filter_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        expr = self._build_filter_expr(filter_dict)
        if not expr:
            return []
        results = self.client.query(
            collection_name=self.collection_name,
            filter=expr,
            output_fields=["id", "text", "metadata_json"],
        )
        records: List[Dict[str, Any]] = []
        for record in results:
            meta_str = record.get("metadata_json", "{}")
            try:
                meta = json.loads(meta_str)
            except (json.JSONDecodeError, TypeError):
                meta = {}
            records.append({
                "id": record.get("id", ""),
                "text": record.get("text", ""),
                "metadata": meta,
            })
        return records

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
            self.client.flush(self.collection_name)
            self._ensure_loaded(self.collection_name)
        return len(ids)

    def get_collection_stats(self) -> Dict[str, Any]:
        stats = self.client.get_collection_stats(self.collection_name)
        return {
            "count": stats.get("row_count", 0),
            "name": self.requested_collection_name,
            "physical_name": self.collection_name,
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
            if key == "collection":
                continue
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
