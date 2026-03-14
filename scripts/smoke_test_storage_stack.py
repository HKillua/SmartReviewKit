#!/usr/bin/env python3
"""Smoke-test the standalone production storage stack."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.config import load_agent_config, load_memory_config
from src.agent.hooks.rate_limit import RateLimitExceeded
from src.agent.memory.error_memory import ErrorRecord
from src.agent.memory.session_memory import SessionSummary
from src.agent.memory.skill_memory import ToolUsageRecord
from src.agent.types import Message
from src.core.settings import load_settings
from src.core.trace.trace_context import TraceContext
from src.libs.vector_store import VectorStoreFactory
from src.storage.postgres import PostgresExecutor
from src.storage.runtime import (
    create_conversation_store,
    create_feedback_store,
    create_image_storage,
    create_ingestion_backends,
    create_memory_stores,
    create_rate_limit_hook,
    create_semantic_cache,
    create_sparse_index,
)


def _load_raw_settings(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _wait_for(name: str, check: Any, timeout: float = 180.0, interval: float = 2.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            if check():
                print(f"[ready] {name}")
                return
        except Exception as exc:  # pragma: no cover - exercised in integration
            last_error = exc
        time.sleep(interval)
    if last_error:
        raise RuntimeError(f"{name} not ready: {last_error}") from last_error
    raise RuntimeError(f"{name} not ready before timeout")


async def _exercise_conversations(raw_settings: dict[str, Any]) -> dict[str, Any]:
    agent_cfg = load_agent_config(raw_settings)
    store = create_conversation_store(raw_settings, agent_cfg)
    conv = await store.create("storage-smoke-user")
    conv.messages.append(Message(role="user", content="storage smoke message"))
    await store.update(conv)
    fetched = await store.get(conv.id, conv.user_id)
    listed = await store.list_conversations(conv.user_id, limit=5)
    deleted = await store.delete(conv.id, conv.user_id)
    return {
        "conversation_id": conv.id,
        "fetched": fetched is not None,
        "listed_count": len(listed),
        "deleted": deleted,
    }


async def _exercise_memory(raw_settings: dict[str, Any]) -> dict[str, Any]:
    memory_cfg = load_memory_config(raw_settings)
    stores = create_memory_stores(memory_cfg, raw_settings)
    user_id = "storage-smoke-user"

    await stores["profile"].update_profile(user_id, {"preferences": {"answer_style": "concise"}})
    profile = await stores["profile"].get_profile(user_id)

    summary = SessionSummary(
        session_id="storage-smoke-session",
        user_id=user_id,
        topics=["tcp", "udp"],
        key_questions=["什么是三次握手？"],
        summary_text="讨论了 TCP 和 UDP 的差异",
        extraction_metadata={"mode": "rule", "confidence": 0.91},
    )
    await stores["session"].save_session(user_id, summary)
    recent_sessions = await stores["session"].get_recent_sessions(user_id, limit=5)

    error = ErrorRecord(
        question="TCP 是什么？",
        topic="tcp",
        concepts=["tcp"],
        user_answer="不知道",
        correct_answer="传输控制协议",
        explanation="一个传输层协议",
    )
    await stores["error"].add_error(user_id, error)
    errors = await stores["error"].get_errors(user_id, limit=5)

    await stores["knowledge_map"].update_mastery(user_id, "tcp", correct=True)
    node = await stores["knowledge_map"].get_node(user_id, "tcp")

    usage = ToolUsageRecord(question_pattern="总结 tcp", tool_chain=["review_summary"], quality_score=0.8)
    await stores["skill"].save_usage(user_id, usage)
    similar = await stores["skill"].search_similar(user_id, "tcp 总结", limit=3)

    for store in stores.values():
        if store is not None and hasattr(store, "close"):
            await store.close()

    return {
        "profile_pref": profile.preferences.get("answer_style"),
        "session_count": len(recent_sessions),
        "error_count": len(errors),
        "knowledge_node_found": node is not None,
        "skill_match_count": len(similar),
    }


async def _exercise_feedback(raw_settings: dict[str, Any]) -> dict[str, Any]:
    store = create_feedback_store(raw_settings)
    feedback_id = await store.add(
        user_id="storage-smoke-user",
        conversation_id="storage-smoke-conv",
        rating="up",
        query="tcp",
        response_preview="TCP 是一种可靠传输协议",
    )
    stats = await store.stats()
    recent = await store.list_recent(limit=5)
    if hasattr(store, "close"):
        await store.close()
    return {
        "feedback_id": feedback_id,
        "stats": stats,
        "recent_count": len(recent),
    }


def _exercise_postgres(settings: Any) -> dict[str, Any]:
    executor = PostgresExecutor(settings.postgres.dsn)
    with executor.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
        row = cur.fetchone()
    return {"select_one": int(row[0])}


def _exercise_object_and_ingestion(settings: Any) -> dict[str, Any]:
    backends = create_ingestion_backends(settings, collection="storage-smoke")
    object_store = backends.object_store

    key = "uploads/storage-smoke/hello.txt"
    object_store.put_bytes(key, b"hello storage stack", content_type="text/plain")
    data = object_store.read_bytes(key).decode("utf-8")

    image_storage = create_image_storage(settings, collection="storage-smoke")
    image_storage.save_image(
        image_id="storage-smoke-image",
        image_data=b"\x89PNG\r\n\x1a\nfake",
        collection="storage-smoke",
        doc_hash="doc-hash-1",
        page_num=1,
        extension="png",
    )
    image_path = image_storage.get_image_path("storage-smoke-image")
    image_listing = image_storage.list_images(collection="storage-smoke")
    image_storage.delete_image("storage-smoke-image")

    registry = backends.document_registry
    task_store = backends.task_store
    registry.upsert_document(
        file_hash="storage-smoke-hash",
        file_path="/tmp/storage-smoke.pdf",
        collection="storage-smoke",
        status="succeeded",
        object_uri=object_store.uri_for(key),
        metadata={"smoke": True},
    )
    doc = registry.get_document("storage-smoke-hash")

    task_id = task_store.create_task("/tmp/storage-smoke.pdf", "storage-smoke", object_uri=object_store.uri_for(key))
    task_store.update_status(task_id, "succeeded")
    task = task_store.get_task(task_id)

    return {
        "object_store_readback": data,
        "image_path_cached": bool(image_path),
        "image_listing_count": len(image_listing),
        "document_registry_status": doc["status"] if doc else "",
        "ingestion_task_status": task["status"] if task else "",
    }


def _exercise_sparse_index(settings: Any) -> dict[str, Any]:
    collection = "storage-smoke-sparse"
    index = create_sparse_index(settings, collection=collection)
    trace = TraceContext(trace_type="query", metadata={"source": "storage_stack_smoke"})
    term_stats = [
        {
            "chunk_id": "doc1_chunk0",
            "term_frequencies": {"tcp": 2, "three": 1, "handshake": 1},
            "doc_length": 4,
        },
        {
            "chunk_id": "doc2_chunk0",
            "term_frequencies": {"udp": 2, "datagram": 1},
            "doc_length": 3,
        },
    ]
    index.build(term_stats, collection=collection, trace=trace)
    results = index.query(["tcp", "handshake"], top_k=5, collection=collection, trace=trace)
    removed = index.remove_document("doc1", collection=collection)
    results_after_delete = index.query(["tcp"], top_k=5, collection=collection, trace=trace)
    return {
        "initial_hits": len(results),
        "removed": removed,
        "hits_after_delete": len(results_after_delete),
    }


def _exercise_vector_store(settings: Any) -> dict[str, Any]:
    collection = "storage_smoke_vectors"
    store = VectorStoreFactory.create(settings, collection_name=collection)
    dim = int(settings.embedding.dimensions)
    vec_a = [1.0] + [0.0] * (dim - 1)
    vec_b = [0.0, 1.0] + [0.0] * (dim - 2)
    store.upsert(
        [
            {"id": "vec-doc-1", "vector": vec_a, "metadata": {"text": "tcp handshake", "source_ref": "tcp"}},
            {"id": "vec-doc-2", "vector": vec_b, "metadata": {"text": "udp datagram", "source_ref": "udp"}},
        ]
    )
    results = store.query(vec_a, top_k=2)
    store.delete(["vec-doc-1", "vec-doc-2"])
    results_after_delete = store.query(vec_a, top_k=2)
    return {
        "initial_hits": len(results),
        "top_hit": results[0]["id"] if results else "",
        "hits_after_delete": len(results_after_delete),
    }


def _milvus_ready(settings: Any) -> bool:
    # Fresh CI environments legitimately start with zero collections. Readiness
    # should validate connectivity, not pre-existing data.
    store = VectorStoreFactory.create(settings, collection_name="storage_smoke_health")
    store.list_collections()
    return True


async def _exercise_cache_and_rate_limit(settings: Any, raw_settings: dict[str, Any]) -> dict[str, Any]:
    def _fake_embed(text: Any) -> list[float]:
        if isinstance(text, list):
            joined = " ".join(str(item) for item in text)
        else:
            joined = str(text)
        return [1.0, 0.0] if "tcp" in joined.lower() else [0.0, 1.0]

    cache = create_semantic_cache(
        settings,
        embedding_fn=_fake_embed,
        similarity_threshold=0.9,
        ttl_seconds=300,
        max_size=50,
    )
    await cache.put("tcp handshake", "ok", {"collection": "storage-smoke"})
    hit = await cache.get("tcp 三次握手")

    hook = create_rate_limit_hook(raw_settings, requests_per_minute=1)
    rate_limit_user = f"storage-smoke-rate-{int(time.time() * 1000)}"
    await hook.before_message(rate_limit_user, "hello")
    blocked = False
    try:
        await hook.before_message(rate_limit_user, "hello again")
    except RateLimitExceeded:
        blocked = True

    return {
        "cache_hit": hit is not None,
        "cache_stats": getattr(cache, "stats", {}),
        "rate_limit_blocked": blocked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the standalone production storage stack.")
    parser.add_argument("--settings", default="config/settings.storage_stack.yaml", help="Settings file to use")
    args = parser.parse_args()

    raw_settings = _load_raw_settings(args.settings)
    settings = load_settings(args.settings)

    _wait_for("postgres", lambda: _exercise_postgres(settings)["select_one"] == 1)

    import redis

    _wait_for("redis", lambda: redis.Redis.from_url(settings.redis.url, decode_responses=True).ping())
    _wait_for("minio", lambda: create_ingestion_backends(settings, collection="storage-smoke").object_store is not None)
    _wait_for("opensearch", lambda: create_sparse_index(settings, collection="storage-smoke-check").ensure_collection_ready("storage-smoke-check"))
    _wait_for("milvus", lambda: _milvus_ready(settings))

    summary: dict[str, Any] = {}
    summary["postgres"] = _exercise_postgres(settings)
    summary["conversation"] = asyncio.run(_exercise_conversations(raw_settings))
    summary["memory"] = asyncio.run(_exercise_memory(raw_settings))
    summary["feedback"] = asyncio.run(_exercise_feedback(raw_settings))
    summary["objects_and_ingestion"] = _exercise_object_and_ingestion(settings)
    summary["sparse_index"] = _exercise_sparse_index(settings)
    summary["vector_store"] = _exercise_vector_store(settings)
    summary["cache_and_rate_limit"] = asyncio.run(_exercise_cache_and_rate_limit(settings, raw_settings))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
