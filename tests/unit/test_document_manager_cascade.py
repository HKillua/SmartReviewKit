from __future__ import annotations

from unittest.mock import MagicMock

from src.ingestion.document_manager import DocumentManager


def test_document_manager_deletes_object_and_tasks() -> None:
    chroma = MagicMock()
    chroma.delete_by_metadata.return_value = 2
    bm25 = MagicMock()
    bm25.remove_document.return_value = True
    images = MagicMock()
    images.list_images.return_value = []
    integrity = MagicMock()
    integrity.remove_record.return_value = True
    registry = MagicMock()
    registry.get_document.return_value = {"object_uri": "minio://bucket/uploads/user/doc.pdf"}
    registry.delete_document.return_value = True
    object_store = MagicMock()
    object_store.key_from_uri.return_value = "uploads/user/doc.pdf"
    object_store.delete_object.return_value = True
    task_store = MagicMock()
    task_store.delete_tasks_by_file.return_value = 3

    manager = DocumentManager(
        chroma,
        bm25,
        images,
        integrity,
        document_registry=registry,
        object_store=object_store,
        task_store=task_store,
    )

    result = manager.delete_document(
        source_path="/tmp/doc.pdf",
        collection="demo",
        source_hash="hash123",
    )

    assert result.object_removed is True
    assert result.registry_removed is True
    assert result.tasks_removed == 3
    object_store.delete_object.assert_called_once_with("uploads/user/doc.pdf")
