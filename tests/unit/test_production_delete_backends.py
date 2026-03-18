from __future__ import annotations

from types import SimpleNamespace

from src.libs.vector_store.milvus_store import MilvusStore
from src.storage.sparse_index import OpenSearchSparseIndex


class _FakeMilvusDeleteClient:
    def __init__(self, **kwargs) -> None:
        self.deleted_ids = []
        self.flushed = []
        self.loaded = []
        self.states = {}

    def has_collection(self, name: str) -> bool:
        return True

    def list_indexes(self, name: str):
        return ["vector_idx"]

    def get_load_state(self, collection_name: str):
        return {"state": self.states.get(collection_name, "NotLoad")}

    def load_collection(self, collection_name: str, timeout=None) -> None:
        self.loaded.append(collection_name)
        self.states[collection_name] = "Loaded"

    def query(self, **kwargs):
        return [{"id": "chunk-1"}]

    def delete(self, collection_name: str, ids):
        self.deleted_ids.append((collection_name, list(ids)))

    def flush(self, collection_name: str) -> None:
        self.flushed.append(collection_name)


class _FakeIndices:
    def exists(self, index: str) -> bool:
        return True


class _FakeOpenSearchClient:
    def __init__(self, **kwargs) -> None:
        self.indices = _FakeIndices()
        self.last_delete = None

    def delete_by_query(self, *, index: str, body, refresh: bool):
        self.last_delete = {"index": index, "body": body, "refresh": refresh}
        return {"deleted": 1}


def test_milvus_delete_by_metadata_flushes(monkeypatch) -> None:
    monkeypatch.setattr("src.libs.vector_store.milvus_store.PYMILVUS_AVAILABLE", True)
    monkeypatch.setattr("src.libs.vector_store.milvus_store.MilvusClient", _FakeMilvusDeleteClient)

    settings = SimpleNamespace(
        vector_store=SimpleNamespace(collection_name="delete-smoke"),
        milvus=SimpleNamespace(mode="service", uri="http://127.0.0.1:19530", host="127.0.0.1", port=19530, token="", user="", password="", db_name="", dim=768),
    )

    store = MilvusStore(settings)
    deleted = store.delete_by_metadata({"doc_hash": "abc123"})

    assert deleted == 1
    assert store.client.deleted_ids
    assert store.client.flushed == [store.collection_name]
    assert store.collection_name in store.client.loaded


def test_opensearch_sparse_delete_targets_doc_hash(monkeypatch) -> None:
    monkeypatch.setattr("src.storage.sparse_index.OPENSEARCH_AVAILABLE", True)
    monkeypatch.setattr("src.storage.sparse_index.OpenSearch", _FakeOpenSearchClient)

    index = OpenSearchSparseIndex(hosts=["http://localhost:9200"], index_prefix="modular-rag")
    doc_hash = "b6cc294aacce6c495bf4cee5ecd99b5c6508214bcc4375288b9c5917900384f8"
    removed = index.remove_document(doc_hash, collection="demo")

    assert removed is True
    payload = index._client.last_delete["body"]["query"]["bool"]["should"]
    assert {"term": {"doc_hash": doc_hash}} in payload
    assert {"prefix": {"chunk_id": f"doc_{doc_hash[:16]}"}} in payload
