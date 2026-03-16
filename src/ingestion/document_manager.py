"""Cross-store document lifecycle management.

This module provides a single entry-point for listing, inspecting, and
deleting documents across all four storage backends (ChromaDB, BM25,
ImageStorage, FileIntegrityChecker).

Design Principles:
- Coordinated: one call cascades into all relevant stores.
- Fail-safe: partial failures are reported but do not abort remaining stores.
- Read-only safe: list / stats / detail methods never mutate data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data-classes
# ---------------------------------------------------------------------------

@dataclass
class DocumentInfo:
    """Summary information about an ingested document."""

    source_path: str
    source_hash: str
    collection: Optional[str] = None
    chunk_count: int = 0
    image_count: int = 0
    processed_at: Optional[str] = None


@dataclass
class DocumentDetail(DocumentInfo):
    """Extended document info including chunk IDs and image IDs."""

    chunk_ids: List[str] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)


@dataclass
class DeleteResult:
    """Outcome of a delete_document operation."""

    success: bool
    chunks_deleted: int = 0
    bm25_removed: bool = False
    images_deleted: int = 0
    integrity_removed: bool = False
    registry_removed: bool = False
    object_removed: bool = False
    tasks_removed: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class CollectionStats:
    """Aggregate statistics for a collection."""

    collection: Optional[str] = None
    document_count: int = 0
    chunk_count: int = 0
    image_count: int = 0


# ---------------------------------------------------------------------------
# DocumentManager
# ---------------------------------------------------------------------------

class DocumentManager:
    """Coordinate document lifecycle across all storage backends.

    Args:
        chroma_store: ChromaStore instance (vector store).
        bm25_indexer: BM25Indexer instance (sparse index).
        image_storage: ImageStorage instance (image files + SQLite index).
        file_integrity: SQLiteIntegrityChecker instance (ingestion history).
    """

    def __init__(
        self,
        chroma_store: Any,
        bm25_indexer: Any,
        image_storage: Any,
        file_integrity: Any,
        *,
        document_registry: Any = None,
        object_store: Any = None,
        task_store: Any = None,
    ) -> None:
        self.chroma = chroma_store
        self.bm25 = bm25_indexer
        self.images = image_storage
        self.integrity = file_integrity
        self.document_registry = document_registry
        self.object_store = object_store
        self.task_store = task_store

    # ------------------------------------------------------------------
    # list_documents
    # ------------------------------------------------------------------

    def list_documents(
        self, collection: Optional[str] = None
    ) -> List[DocumentInfo]:
        """Return a list of ingested documents.

        Combines information from the integrity checker (source_path,
        hash, processed_at) with counts from ChromaDB and ImageStorage.

        Args:
            collection: Optional collection filter.

        Returns:
            List of ``DocumentInfo`` objects.
        """
        records = self.integrity.list_processed(collection)
        if not records and self.document_registry is not None:
            try:
                registry_docs = self.document_registry.list_documents(collection)
                records = [
                    {
                        "file_hash": doc["file_hash"],
                        "file_path": doc["file_path"],
                        "collection": doc.get("collection"),
                        "processed_at": "",
                        "updated_at": doc.get("metadata", {}).get("updated_at", ""),
                    }
                    for doc in registry_docs
                    if doc.get("status") in {"ready", "queued", "running", "failed"}
                ]
            except Exception:
                logger.debug("Failed to read document registry during list_documents", exc_info=True)

        docs: List[DocumentInfo] = []
        for rec in records:
            source_hash = rec["file_hash"]
            source_path = rec["file_path"]
            coll = rec.get("collection")

            # Count chunks in Chroma
            chunk_count = self._count_chunks(source_hash)

            # Count images
            image_count = self._count_images(source_hash)

            docs.append(
                DocumentInfo(
                    source_path=source_path,
                    source_hash=source_hash,
                    collection=coll,
                    chunk_count=chunk_count,
                    image_count=image_count,
                    processed_at=rec.get("processed_at"),
                )
            )

        return docs

    # ------------------------------------------------------------------
    # get_document_detail
    # ------------------------------------------------------------------

    def get_document_detail(self, doc_id: str) -> Optional[DocumentDetail]:
        """Get detailed information about a single document.

        *doc_id* is matched against the ``source_hash`` stored in the
        integrity checker.

        Args:
            doc_id: The document's source_hash.

        Returns:
            ``DocumentDetail`` with chunk/image IDs, or *None* if not found.
        """
        # Look up integrity record
        all_records = self.integrity.list_processed()
        record = None
        for rec in all_records:
            if rec["file_hash"] == doc_id:
                record = rec
                break

        if record is None:
            return None

        source_hash = record["file_hash"]

        # Collect chunk IDs from Chroma
        chunk_ids = self._get_chunk_ids(source_hash)

        # Collect image IDs
        image_ids = self._get_image_ids(source_hash)

        return DocumentDetail(
            source_path=record["file_path"],
            source_hash=source_hash,
            collection=record.get("collection"),
            chunk_count=len(chunk_ids),
            image_count=len(image_ids),
            processed_at=record.get("processed_at"),
            chunk_ids=chunk_ids,
            image_ids=image_ids,
        )

    # ------------------------------------------------------------------
    # delete_document
    # ------------------------------------------------------------------

    def delete_document(
        self,
        source_path: str,
        collection: str = "default",
        source_hash: Optional[str] = None,
        *,
        remove_original_object: bool = True,
        remove_task_records: bool = True,
    ) -> DeleteResult:
        """Delete a document from all storage backends.

        Coordinates deletion across ChromaDB, BM25, ImageStorage, and
        FileIntegrity.  Partial failures are captured in
        ``DeleteResult.errors`` but do not prevent remaining stores
        from being cleaned.

        The document is identified by its *source_hash*.  When the hash
        is not supplied the method tries to compute it from the file;
        if the file no longer exists it falls back to looking up the
        hash from the integrity records by path.

        Args:
            source_path: Original filesystem path of the document.
            collection: Collection the document belongs to.
            source_hash: Pre-computed SHA-256 hash.  When provided the
                method will not attempt to read the source file.

        Returns:
            ``DeleteResult`` summarising what was cleaned.
        """
        result = DeleteResult(success=True)

        # Resolve hash – prefer caller-supplied, then file, then DB lookup
        if source_hash is None:
            try:
                source_hash = self.integrity.compute_sha256(source_path)
            except Exception as e:
                source_hash = self._hash_from_path(source_path)
                if source_hash is None:
                    result.success = False
                    result.errors.append(f"Cannot identify document: {e}")
                    return result

        # 1. ChromaDB – delete chunks matching source_hash
        try:
            count = self.chroma.delete_by_metadata(
                {"doc_hash": source_hash}
            )
            result.chunks_deleted = count
        except Exception as e:
            result.errors.append(f"ChromaDB delete failed: {e}")

        # 2. BM25 – remove postings for this document
        try:
            result.bm25_removed = self.bm25.remove_document(
                source_hash, collection
            )
        except Exception as e:
            result.errors.append(f"BM25 remove failed: {e}")

        # 3. ImageStorage – delete images by doc_hash
        try:
            images = self.images.list_images(doc_hash=source_hash)
            deleted_imgs = 0
            for img in images:
                if self.images.delete_image(img["image_id"]):
                    deleted_imgs += 1
            result.images_deleted = deleted_imgs
        except Exception as e:
            result.errors.append(f"ImageStorage delete failed: {e}")

        # 4. FileIntegrity – remove the ingestion record
        try:
            result.integrity_removed = self.integrity.remove_record(
                source_hash,
                collection,
            )
        except Exception as e:
            result.errors.append(f"FileIntegrity remove failed: {e}")

        object_uri = ""
        if self.document_registry is not None:
            try:
                registry_doc = self.document_registry.get_document(source_hash, collection)
                object_uri = (registry_doc or {}).get("object_uri", "") or ""
                result.registry_removed = self.document_registry.delete_document(source_hash, collection)
            except Exception as e:
                result.errors.append(f"Document registry delete failed: {e}")
        if not object_uri and self.task_store is not None:
            try:
                tasks = self.task_store.list_tasks(file_hash=source_hash, collection=collection, limit=10)
                object_uri = next((task.get("object_uri", "") or "" for task in tasks if task.get("object_uri")), "")
            except Exception:
                logger.debug("Failed to read task object URI during delete", exc_info=True)

        if remove_original_object and self.object_store is not None:
            object_key = ""
            if object_uri:
                object_key = self.object_store.key_from_uri(object_uri) or ""
            if object_key:
                try:
                    result.object_removed = self.object_store.delete_object(object_key)
                except Exception as e:
                    result.errors.append(f"Object store delete failed: {e}")

        if remove_task_records and self.task_store is not None:
            try:
                result.tasks_removed = int(self.task_store.delete_tasks_by_file(source_hash, collection))
            except Exception as e:
                result.errors.append(f"Task store delete failed: {e}")

        if result.errors:
            result.success = False

        return result

    def rollback_document(
        self,
        *,
        source_hash: str,
        source_path: str,
        collection: str,
        purge_original_object: bool = False,
        remove_task_records: bool = False,
    ) -> DeleteResult:
        """Rollback partially written ingestion state for a document."""
        return self.delete_document(
            source_path=source_path,
            collection=collection,
            source_hash=source_hash,
            remove_original_object=purge_original_object,
            remove_task_records=remove_task_records,
        )

    # ------------------------------------------------------------------
    # get_collection_stats
    # ------------------------------------------------------------------

    def get_collection_stats(
        self, collection: Optional[str] = None
    ) -> CollectionStats:
        """Return aggregate statistics for a collection.

        Args:
            collection: Collection name.  When *None*, stats span
                all collections.

        Returns:
            ``CollectionStats`` dataclass.
        """
        docs = self.list_documents(collection)
        chunk_total = sum(d.chunk_count for d in docs)
        image_total = sum(d.image_count for d in docs)

        return CollectionStats(
            collection=collection,
            document_count=len(docs),
            chunk_count=chunk_total,
            image_count=image_total,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_chunks(self, source_hash: str) -> int:
        """Count chunks in Chroma that belong to *source_hash*."""
        try:
            results = self.chroma.collection.get(
                where={"doc_hash": source_hash}, include=[]
            )
            return len(results.get("ids", []))
        except Exception:
            return 0

    def _get_chunk_ids(self, source_hash: str) -> List[str]:
        """Return chunk IDs from Chroma matching *source_hash*."""
        try:
            results = self.chroma.collection.get(
                where={"doc_hash": source_hash}, include=[]
            )
            return results.get("ids", [])
        except Exception:
            return []

    def _count_images(self, source_hash: str) -> int:
        """Count images belonging to *source_hash*."""
        try:
            return len(self.images.list_images(doc_hash=source_hash))
        except Exception:
            return 0

    def _get_image_ids(self, source_hash: str) -> List[str]:
        """Return image IDs belonging to *source_hash*."""
        try:
            imgs = self.images.list_images(doc_hash=source_hash)
            return [img["image_id"] for img in imgs]
        except Exception:
            return []

    def _hash_from_path(self, source_path: str) -> Optional[str]:
        """Try to find a source_hash from integrity records by path."""
        try:
            for rec in self.integrity.list_processed():
                if rec["file_path"] == source_path:
                    return rec["file_hash"]
        except Exception:
            pass
        if self.document_registry is not None:
            try:
                for rec in self.document_registry.list_documents():
                    if rec["file_path"] == source_path:
                        return rec["file_hash"]
            except Exception:
                pass
        return None
