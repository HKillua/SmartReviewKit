from __future__ import annotations

from types import SimpleNamespace

from src.libs.vector_store.milvus_store import MilvusStore


class _FakeMilvusClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def has_collection(self, name: str) -> bool:
        return True

    def list_indexes(self, name: str):
        return ["vector_idx"]

    def load_collection(self, collection_name: str) -> None:
        return None

    def list_collections(self):
        return ["storage_delete_smoke_43e1a87efe"]

    def get_collection_stats(self, collection_name: str):
        return {"row_count": 0}


def test_milvus_store_normalizes_invalid_collection_name(monkeypatch) -> None:
    monkeypatch.setattr("src.libs.vector_store.milvus_store.PYMILVUS_AVAILABLE", True)
    monkeypatch.setattr("src.libs.vector_store.milvus_store.MilvusClient", _FakeMilvusClient)

    settings = SimpleNamespace(
        vector_store=SimpleNamespace(collection_name="storage-delete-smoke"),
        milvus=SimpleNamespace(mode="service", uri="http://127.0.0.1:19530", host="127.0.0.1", port=19530, token="", user="", password="", db_name="", dim=768),
    )

    store = MilvusStore(settings)

    assert store.requested_collection_name == "storage-delete-smoke"
    assert store.collection_name.startswith("storage_delete_smoke_")
    assert "-" not in store.collection_name
    assert store.list_collections() == ["storage-delete-smoke"]


def test_milvus_store_switch_collection_tracks_requested_name(monkeypatch) -> None:
    monkeypatch.setattr("src.libs.vector_store.milvus_store.PYMILVUS_AVAILABLE", True)
    monkeypatch.setattr("src.libs.vector_store.milvus_store.MilvusClient", _FakeMilvusClient)

    settings = SimpleNamespace(
        vector_store=SimpleNamespace(collection_name="default"),
        milvus=SimpleNamespace(mode="service", uri="http://127.0.0.1:19530", host="127.0.0.1", port=19530, token="", user="", password="", db_name="", dim=768),
    )

    store = MilvusStore(settings)
    store.get_or_switch_collection("tenant-prod/network-v1")

    assert store.requested_collection_name == "tenant-prod/network-v1"
    assert store.collection_name.startswith("tenant_prod_network_v1_")
    assert store.get_collection_stats()["name"] == "tenant-prod/network-v1"


def test_milvus_filter_builder_ignores_collection_key() -> None:
    expr = MilvusStore._build_filter_expr({"collection": "demo", "doc_hash": "abc123"})
    assert "collection" not in expr
    assert 'doc_hash == "abc123"' in expr
