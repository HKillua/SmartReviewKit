from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agent.tools.document_ingest import DocumentIngestArgs, DocumentIngestTool
from src.agent.types import ToolContext
from src.ingestion.worker import IngestionWorker


@pytest.mark.asyncio
async def test_document_ingest_enqueues_when_worker_enabled(tmp_path) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"fake pdf")

    pipeline = MagicMock()
    task_store = MagicMock()
    task_store.create_task.return_value = "task123"
    object_store = MagicMock()
    object_store.uri_for.return_value = "minio://modular-rag/uploads/u1/sample.pdf"
    registry = MagicMock()
    settings = SimpleNamespace(
        ingestion_worker=SimpleNamespace(enabled=True, max_attempts=4, dev_fallback_sync=False),
    )

    tool = DocumentIngestTool(
        settings=settings,
        pipeline=pipeline,
        allowed_dirs=[str(tmp_path)],
        task_store=task_store,
        object_store=object_store,
        document_registry=registry,
    )

    result = await tool.execute(
        ToolContext(user_id="u1", conversation_id="c1", metadata={"uploaded_object_key": "uploads/u1/sample.pdf"}),
        DocumentIngestArgs(file_path=str(file_path), collection="demo"),
    )

    assert result.success is True
    assert result.metadata["task_id"] == "task123"
    assert result.metadata["status"] == "queued"
    pipeline.run.assert_not_called()
    task_store.create_task.assert_called_once()
    registry.upsert_document.assert_called_once()


@pytest.mark.asyncio
async def test_worker_run_once_marks_success(monkeypatch, tmp_path) -> None:
    task_store = MagicMock()
    task_store.claim_task.return_value = {
        "task_id": "task123",
        "file_hash": "hash123",
        "file_path": str(tmp_path / "sample.pdf"),
        "collection": "demo",
        "payload": {"local_file_path": str(tmp_path / "sample.pdf"), "source_type": "auto", "original_filename": "sample.pdf"},
        "object_uri": "",
    }
    backends = SimpleNamespace(
        task_store=task_store,
        document_registry=MagicMock(),
        object_store=MagicMock(),
        image_storage=MagicMock(),
        integrity_checker=MagicMock(),
    )

    class _FakePipeline:
        def __init__(self, settings, collection, force=False) -> None:
            self.collection = collection

        def run(self, path, *, source_type="auto"):
            return SimpleNamespace(success=True, doc_id="hash123", chunk_count=3, image_count=1, vector_ids=["v1", "v2", "v3"])

        def close(self) -> None:
            return None

    monkeypatch.setattr("src.ingestion.worker.create_ingestion_backends", lambda settings, collection: backends)
    monkeypatch.setattr("src.ingestion.worker.IngestionPipeline", _FakePipeline)

    settings = SimpleNamespace(
        vector_store=SimpleNamespace(collection_name="demo"),
        ingestion_worker=SimpleNamespace(poll_interval_seconds=1, lease_seconds=30, heartbeat_interval_seconds=1),
    )
    worker = IngestionWorker(settings)
    processed = await worker.run_once()

    assert processed is True
    task_store.mark_succeeded.assert_called_once()


@pytest.mark.asyncio
async def test_worker_failure_triggers_rollback(monkeypatch, tmp_path) -> None:
    task_store = MagicMock()
    task_store.claim_task.return_value = {
        "task_id": "task123",
        "file_hash": "hash123",
        "file_path": str(tmp_path / "sample.pdf"),
        "collection": "demo",
        "payload": {"local_file_path": str(tmp_path / "sample.pdf"), "source_type": "auto", "original_filename": "sample.pdf"},
        "object_uri": "",
    }
    task_store.mark_failed.return_value = {"terminal": False}
    backends = SimpleNamespace(
        task_store=task_store,
        document_registry=MagicMock(),
        object_store=MagicMock(),
        image_storage=MagicMock(),
        integrity_checker=MagicMock(),
    )
    rollback_spy = MagicMock()

    class _FakePipeline:
        def __init__(self, settings, collection, force=False) -> None:
            self.collection = collection

        def run(self, path, *, source_type="auto"):
            raise RuntimeError("boom")

        def close(self) -> None:
            return None

    class _FakeManager:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def rollback_document(self, **kwargs):
            rollback_spy(**kwargs)

    monkeypatch.setattr("src.ingestion.worker.create_ingestion_backends", lambda settings, collection: backends)
    monkeypatch.setattr("src.ingestion.worker.IngestionPipeline", _FakePipeline)
    monkeypatch.setattr("src.ingestion.worker.DocumentManager", _FakeManager)
    monkeypatch.setattr("src.ingestion.worker.VectorStoreFactory.create", lambda settings, collection_name: MagicMock())
    monkeypatch.setattr("src.ingestion.worker.create_sparse_index", lambda settings, collection: MagicMock())

    settings = SimpleNamespace(
        vector_store=SimpleNamespace(collection_name="demo"),
        ingestion_worker=SimpleNamespace(poll_interval_seconds=1, lease_seconds=30, heartbeat_interval_seconds=1),
    )
    worker = IngestionWorker(settings)
    processed = await worker.run_once()

    assert processed is True
    task_store.mark_failed.assert_called_once()
    rollback_spy.assert_called_once()
