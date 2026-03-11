"""Separate-process ingestion worker backed by PostgreSQL task leasing."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import socket
import tempfile
from pathlib import Path
from typing import Any, Optional

from src.core.settings import Settings, load_settings
from src.ingestion.document_manager import DocumentManager
from src.ingestion.pipeline import IngestionPipeline
from src.storage.runtime import create_ingestion_backends, create_sparse_index
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(
        self,
        settings: Settings,
        *,
        poll_interval_seconds: int = 2,
        lease_seconds: int = 120,
        heartbeat_interval_seconds: int = 30,
        worker_id: str = "",
    ) -> None:
        self.settings = settings
        worker_cfg = settings.ingestion_worker
        self.poll_interval_seconds = poll_interval_seconds or int(getattr(worker_cfg, "poll_interval_seconds", 2) if worker_cfg else 2)
        self.lease_seconds = lease_seconds or int(getattr(worker_cfg, "lease_seconds", 120) if worker_cfg else 120)
        self.heartbeat_interval_seconds = heartbeat_interval_seconds or int(getattr(worker_cfg, "heartbeat_interval_seconds", 30) if worker_cfg else 30)
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self._backends = create_ingestion_backends(settings, collection=settings.vector_store.collection_name)
        self._task_store = self._backends.task_store
        if self._task_store is None:
            raise RuntimeError("Separate-process ingestion worker requires a configured task store")
        self._stop_event = asyncio.Event()

    def request_stop(self) -> None:
        logger.info("Ingestion worker stop requested: worker_id=%s", self.worker_id)
        self._stop_event.set()

    async def run_forever(self) -> None:
        logger.info("Ingestion worker started: worker_id=%s", self.worker_id)
        while not self._stop_event.is_set():
            task = self._task_store.claim_task(self.worker_id, lease_seconds=self.lease_seconds)
            if task is None:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval_seconds)
                except asyncio.TimeoutError:
                    pass
                continue
            await self._process_task(task)
        logger.info("Ingestion worker stopped: worker_id=%s", self.worker_id)

    async def run_once(self) -> bool:
        task = self._task_store.claim_task(self.worker_id, lease_seconds=self.lease_seconds)
        if task is None:
            return False
        await self._process_task(task)
        return True

    async def _process_task(self, task: dict[str, Any]) -> None:
        payload = task.get("payload") or {}
        collection = task["collection"]
        file_hash = task.get("file_hash") or ""
        source_path = str(payload.get("local_file_path") or task.get("file_path") or "")
        source_type = str(payload.get("source_type") or "auto")
        original_filename = str(payload.get("original_filename") or Path(source_path).name or "ingestion.bin")
        object_key = str(payload.get("object_key") or "")
        object_uri = str(task.get("object_uri", "") or payload.get("object_uri", "") or "")
        source_label = str(payload.get("source_label") or original_filename or Path(source_path).name or "ingestion.bin")
        tmp_path: Optional[Path] = None

        if self._backends.document_registry is not None and file_hash:
            try:
                current = self._backends.document_registry.get_document(file_hash, collection)
                merged_metadata = dict((current or {}).get("metadata", {}))
                merged_metadata.update(
                    {
                        "task_id": task["task_id"],
                        "worker_id": self.worker_id,
                        "source_label": source_label,
                        "original_filename": original_filename,
                        "object_key": object_key,
                    }
                )
                self._backends.document_registry.upsert_document(
                    file_hash=file_hash,
                    file_path=source_path,
                    collection=collection,
                    status="running",
                    object_uri=object_uri,
                    metadata=merged_metadata,
                )
            except Exception:
                logger.warning("Failed to mark registry entry running for task %s", task["task_id"], exc_info=True)

        heartbeat_stop = asyncio.Event()
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(task["task_id"], heartbeat_stop))

        try:
            if object_key:
                suffix = Path(original_filename).suffix or Path(source_path).suffix or ".bin"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                self._backends.object_store.download_to_path(object_key, tmp_path)
                pipeline_path = str(tmp_path)
            else:
                pipeline_path = source_path

            pipeline = IngestionPipeline(self.settings, collection=collection, force=bool(payload.get("force", False)))
            try:
                run_kwargs: dict[str, Any] = {"source_type": source_type}
                try:
                    run_signature = inspect.signature(pipeline.run)
                except (TypeError, ValueError):
                    run_signature = None
                if run_signature is None or "metadata_overrides" in run_signature.parameters:
                    run_kwargs["metadata_overrides"] = {
                        "source_path": source_path or original_filename,
                        "source_label": source_label,
                        "original_filename": original_filename,
                        "object_uri": object_uri,
                        "object_key": object_key,
                    }
                if run_signature is None or "record_file_path" in run_signature.parameters:
                    run_kwargs["record_file_path"] = source_path or original_filename
                result = await asyncio.to_thread(
                    pipeline.run,
                    pipeline_path,
                    **run_kwargs,
                )
            finally:
                pipeline.close()

            if result.success:
                result_stages = getattr(result, "stages", {}) or {}
                integrity_stage = result_stages.get("integrity", {})
                skipped_existing = bool(integrity_stage.get("skipped")) and integrity_stage.get("reason") == "already_processed"
                if skipped_existing:
                    self._task_store.update_status(
                        task["task_id"],
                        "skipped_existing",
                        "already_processed_for_collection",
                        result={
                            "doc_id": result.doc_id,
                            "chunk_count": result.chunk_count,
                            "image_count": result.image_count,
                            "vector_ids_count": len(result.vector_ids),
                        },
                        worker_id=self.worker_id,
                    )
                    if self._backends.document_registry is not None and result.doc_id:
                        try:
                            current = self._backends.document_registry.get_document(result.doc_id, collection)
                            merged_metadata = dict((current or {}).get("metadata", {}))
                            merged_metadata.update(
                                {
                                    "task_id": task["task_id"],
                                    "chunk_count": result.chunk_count,
                                    "image_count": result.image_count,
                                    "worker_id": self.worker_id,
                                    "source_label": source_label,
                                    "original_filename": original_filename,
                                    "object_key": object_key,
                                }
                            )
                            self._backends.document_registry.upsert_document(
                                file_hash=result.doc_id,
                                file_path=source_path,
                                collection=collection,
                                status="ready",
                                object_uri=object_uri,
                                metadata=merged_metadata,
                            )
                        except Exception:
                            logger.warning(
                                "Failed to finalize document registry for skipped task %s",
                                task["task_id"],
                                exc_info=True,
                            )
                    return

                if result.chunk_count <= 0 or len(result.vector_ids) <= 0:
                    failure = self._task_store.mark_failed(
                        task["task_id"],
                        self.worker_id,
                        "ingestion_completed_without_chunks_or_vectors",
                    )
                    self._rollback_attempt(
                        file_hash=file_hash or (result.doc_id or ""),
                        source_path=source_path,
                        collection=collection,
                        purge_original_object=bool(failure.get("terminal")),
                    )
                    return

                if self._backends.document_registry is not None and result.doc_id:
                    try:
                        current = self._backends.document_registry.get_document(result.doc_id, collection)
                        merged_metadata = dict((current or {}).get("metadata", {}))
                        merged_metadata.update(
                            {
                                "task_id": task["task_id"],
                                "chunk_count": result.chunk_count,
                                "image_count": result.image_count,
                                "worker_id": self.worker_id,
                                "source_label": source_label,
                                "original_filename": original_filename,
                                "object_key": object_key,
                            }
                        )
                        self._backends.document_registry.upsert_document(
                            file_hash=result.doc_id,
                            file_path=source_path,
                            collection=collection,
                            status="ready",
                            object_uri=object_uri,
                            metadata=merged_metadata,
                        )
                    except Exception:
                        logger.warning("Failed to finalize document registry for task %s", task["task_id"], exc_info=True)
                self._task_store.mark_succeeded(
                    task["task_id"],
                    self.worker_id,
                    result={
                        "doc_id": result.doc_id,
                        "chunk_count": result.chunk_count,
                        "image_count": result.image_count,
                        "vector_ids_count": len(result.vector_ids),
                    },
                )
            else:
                failure = self._task_store.mark_failed(task["task_id"], self.worker_id, result.error or "ingestion_failed")
                self._rollback_attempt(
                    file_hash=file_hash or (result.doc_id or ""),
                    source_path=source_path,
                    collection=collection,
                    purge_original_object=bool(failure.get("terminal")),
                )
        except Exception as exc:
            logger.exception("Worker failed to process task %s", task["task_id"])
            failure = self._task_store.mark_failed(task["task_id"], self.worker_id, str(exc))
            self._rollback_attempt(
                file_hash=file_hash,
                source_path=source_path,
                collection=collection,
                purge_original_object=bool(failure.get("terminal")),
            )
        finally:
            heartbeat_stop.set()
            try:
                await heartbeat_task
            except Exception:
                logger.debug("Heartbeat task ended with an error", exc_info=True)
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    logger.debug("Failed to remove worker temp file %s", tmp_path, exc_info=True)

    async def _heartbeat_loop(self, task_id: str, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.heartbeat_interval_seconds)
                break
            except asyncio.TimeoutError:
                self._task_store.heartbeat(task_id, self.worker_id, lease_seconds=self.lease_seconds)

    def _rollback_attempt(
        self,
        *,
        file_hash: str,
        source_path: str,
        collection: str,
        purge_original_object: bool,
    ) -> None:
        if not file_hash:
            return
        vector_store = VectorStoreFactory.create(self.settings, collection_name=collection)
        sparse_index = create_sparse_index(self.settings, collection=collection)
        manager = DocumentManager(
            vector_store,
            sparse_index,
            self._backends.image_storage,
            self._backends.integrity_checker,
            document_registry=self._backends.document_registry,
            object_store=self._backends.object_store,
            task_store=None,
        )
        manager.rollback_document(
            source_hash=file_hash,
            source_path=source_path,
            collection=collection,
            purge_original_object=purge_original_object,
            remove_task_records=False,
        )


async def run_worker_from_settings(settings_path: str) -> None:
    worker = IngestionWorker(load_settings(settings_path))
    await worker.run_forever()
