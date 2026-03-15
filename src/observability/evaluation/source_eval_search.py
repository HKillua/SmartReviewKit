"""Lightweight retrieval adapter for evaluation on top of sparse index only.

This adapter is intentionally optimized for stable offline evaluation:
- it uses the configured sparse backend (OpenSearch/BM25)
- it resolves source labels from the document registry
- it avoids vector-store startup and embedding latency when all we need is a
  source-level retrieval baseline
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.query_engine.query_processor import QueryProcessor
from src.core.types import RetrievalResult
from src.core.settings import resolve_path
from src.storage.runtime import create_sparse_index


class SparseSourceEvalSearch:
    """Sparse-only evaluation adapter with source-label enrichment."""

    def __init__(self, settings: Any, collection: str) -> None:
        self._settings = settings
        self._default_collection = collection
        self._query_processor = QueryProcessor()
        self._sparse_index = create_sparse_index(settings, collection=collection)
        self._source_lookup = self._load_source_lookup(collection)

    def search(
        self,
        *,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        collection = str((filters or {}).get("collection") or self._default_collection)
        processed = self._query_processor.process(query)
        query_terms = list(processed.keywords or [query])
        hits = self._sparse_index.query(query_terms=query_terms, top_k=top_k, collection=collection)
        results: List[RetrievalResult] = []
        for hit in hits:
            doc_hash = str(hit.get("doc_hash") or "")
            source_path = self._source_lookup.get(doc_hash, "")
            source_label = Path(source_path).name if source_path else ""
            metadata = {
                "doc_hash": doc_hash,
                "source_path": source_path,
                "source_label": source_label,
                "retrieval_backend": "sparse_eval",
            }
            results.append(
                RetrievalResult(
                    chunk_id=str(hit["chunk_id"]),
                    score=float(hit["score"]),
                    text="",
                    metadata=metadata,
                )
            )
        return results

    def _load_source_lookup(self, collection: str) -> Dict[str, str]:
        if getattr(self._settings.postgres, "enabled", False) and getattr(self._settings.postgres, "dsn", ""):
            try:
                import psycopg

                with psycopg.connect(self._settings.postgres.dsn) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT file_hash, file_path
                            FROM document_registry
                            WHERE collection = %s
                            """,
                            (collection,),
                        )
                        return {str(file_hash): str(file_path) for file_hash, file_path in cur.fetchall()}
            except Exception:
                return {}

        db_path = resolve_path("data/db/ingestion_history.db")
        if not db_path.exists():
            return {}

        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT file_hash, file_path
                FROM ingestion_history
                WHERE collection = ?
                """,
                (collection,),
            ).fetchall()
        return {str(file_hash): str(file_path) for file_hash, file_path in rows}
