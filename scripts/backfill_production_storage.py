"""Backfill local JSON/SQLite storage into production backends.

Usage:
    python scripts/backfill_production_storage.py --settings config/settings.yaml --reindex-documents
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from pathlib import Path

from src.agent.memory.session_memory import SessionSummary
from src.agent.types import Conversation
from src.core.settings import load_settings, resolve_path
from src.core.trace.trace_collector import PostgresTraceSink
from src.ingestion.pipeline import IngestionPipeline
from src.observability.evaluation.migration_compare import MigrationCompareRunner
from src.storage.object_store import ObjectImageStorage
from src.storage.postgres_backends import (
    PostgresConversationStore,
    PostgresDocumentRegistry,
    PostgresFeedbackStore,
    PostgresIntegrityChecker,
    PostgresSessionMemory,
    PostgresStudentProfileMemory,
)
from src.storage.runtime import create_ingestion_backends


def _iter_sqlite_rows(db_path: Path, query: str):
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return list(conn.execute(query))


async def _backfill_conversations(base_dir: Path, dsn: str) -> int:
    store = PostgresConversationStore(dsn)
    count = 0
    for file_path in sorted(base_dir.glob("*/*.json")):
        try:
            conversation = Conversation.model_validate_json(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        await store.update(conversation)
        count += 1
    return count


async def _backfill_memory(db_dir: Path, dsn: str) -> dict[str, int]:
    results = {"profiles": 0, "sessions": 0}
    profile_store = PostgresStudentProfileMemory(dsn)
    session_store = PostgresSessionMemory(dsn)

    for row in _iter_sqlite_rows(db_dir / "profiles.db", "SELECT user_id, data FROM student_profiles"):
        payload = json.loads(row["data"])
        await profile_store.update_profile(row["user_id"], payload)
        results["profiles"] += 1

    for row in _iter_sqlite_rows(db_dir / "sessions.db", "SELECT user_id, data FROM session_summaries"):
        payload = json.loads(row["data"])
        await session_store.save_session(row["user_id"], SessionSummary.model_validate(payload))
        results["sessions"] += 1

    return results


async def _backfill_feedback(db_path: Path, dsn: str) -> int:
    store = PostgresFeedbackStore(dsn)
    count = 0
    for row in _iter_sqlite_rows(db_path, "SELECT * FROM feedback ORDER BY id ASC"):
        await store.add(
            user_id=row["user_id"],
            conversation_id=row["conversation_id"],
            rating=row["rating"],
            message_index=row["message_index"],
            comment=row["comment"],
            query=row["query"],
            response_preview=row["response_preview"],
        )
        count += 1
    return count


def _backfill_images(settings, backends) -> int:
    image_db = resolve_path("data/db/image_index.db")
    if not image_db.exists() or not isinstance(backends.image_storage, ObjectImageStorage):
        return 0
    count = 0
    for row in _iter_sqlite_rows(image_db, "SELECT image_id, file_path, collection, doc_hash, page_num FROM image_index"):
        file_path = Path(row["file_path"])
        if not file_path.exists():
            continue
        backends.image_storage.register_image(
            image_id=row["image_id"],
            file_path=file_path,
            collection=row["collection"],
            doc_hash=row["doc_hash"],
            page_num=row["page_num"],
        )
        count += 1
    return count


def _backfill_ingestion_registry(dsn: str) -> int:
    local_checker = resolve_path("data/db/ingestion_history.db")
    if not local_checker.exists():
        return 0
    prod_checker = PostgresIntegrityChecker(dsn)
    registry = PostgresDocumentRegistry(dsn)
    count = 0
    with sqlite3.connect(local_checker) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM ingestion_history").fetchall()
    for row in rows:
        if row["status"] == "success":
            prod_checker.mark_success(row["file_hash"], row["file_path"], row["collection"])
            registry.upsert_document(
                file_hash=row["file_hash"],
                file_path=row["file_path"],
                collection=row["collection"] or "default",
                status="ready",
                metadata={"migrated_from": "sqlite"},
            )
        else:
            prod_checker.mark_failed(row["file_hash"], row["file_path"], row["error_msg"] or "")
        count += 1
    return count


def _reindex_documents(settings) -> int:
    local_checker = resolve_path("data/db/ingestion_history.db")
    if not local_checker.exists():
        return 0
    with sqlite3.connect(local_checker) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT file_path, collection FROM ingestion_history WHERE status = 'success' ORDER BY updated_at ASC"
        ).fetchall()

    count = 0
    for row in rows:
        file_path = Path(row["file_path"])
        if not file_path.exists():
            continue
        collection = row["collection"] or settings.vector_store.collection_name
        pipeline = IngestionPipeline(settings=settings, collection=collection, force=True)
        try:
            result = pipeline.run(str(file_path))
            if result.success:
                count += 1
        finally:
            pipeline.close()
    return count


def _backfill_traces(trace_path: Path, dsn: str) -> int:
    if not trace_path.exists():
        return 0
    sink = PostgresTraceSink(dsn)
    count = 0
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                sink.collect(json.loads(line))
            except Exception:
                continue
            count += 1
    return count


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill local storage to production backends.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to settings file")
    parser.add_argument(
        "--legacy-settings",
        default="config/settings.yaml",
        help="Legacy retrieval settings path for migration compare.",
    )
    parser.add_argument(
        "--reindex-documents",
        action="store_true",
        help="Re-ingest successful local documents into the configured production vector/sparse stack.",
    )
    parser.add_argument(
        "--skip-trace-backfill",
        action="store_true",
        help="Skip importing JSONL traces into PostgreSQL.",
    )
    parser.add_argument(
        "--skip-compare-eval",
        action="store_true",
        help="Skip retrieval compare evaluation after migration/reindex.",
    )
    parser.add_argument(
        "--compare-report",
        default="logs/migration_compare_report.json",
        help="Where to write the retrieval migration compare report.",
    )
    args = parser.parse_args()

    settings = load_settings(args.settings)
    if not settings.postgres.enabled or not settings.postgres.dsn:
        raise SystemExit("postgres.enabled must be true and postgres.dsn must be configured before backfill")

    backends = create_ingestion_backends(settings, collection=settings.vector_store.collection_name)

    conversations = await _backfill_conversations(resolve_path("data/conversations"), settings.postgres.dsn)
    feedback = await _backfill_feedback(resolve_path("data/db/feedback.db"), settings.postgres.dsn)
    memory = await _backfill_memory(resolve_path("data/memory"), settings.postgres.dsn)
    registry = _backfill_ingestion_registry(settings.postgres.dsn)
    images = _backfill_images(settings, backends)
    reindexed = _reindex_documents(settings) if args.reindex_documents else 0
    traces = 0 if args.skip_trace_backfill else _backfill_traces(resolve_path(settings.observability.trace_file), settings.postgres.dsn)
    compare_report_path = resolve_path(args.compare_report)
    compare_report = None
    if not args.skip_compare_eval:
        comparator = MigrationCompareRunner(
            legacy_settings_path=args.legacy_settings,
            production_settings_path=args.settings,
        )
        compare_report = comparator.run(
            test_set_path=resolve_path("tests/fixtures/golden_test_set.json"),
            top_k=settings.retrieval.fusion_top_k,
            collection=settings.vector_store.collection_name,
        )
        compare_report_path.parent.mkdir(parents=True, exist_ok=True)
        compare_report_path.write_text(compare_report.to_json(), encoding="utf-8")

    print("Backfill complete")
    print(f"- conversations: {conversations}")
    print(f"- memory profiles: {memory['profiles']}")
    print(f"- memory sessions: {memory['sessions']}")
    print(f"- feedback rows: {feedback}")
    print(f"- ingestion records: {registry}")
    print(f"- images: {images}")
    print(f"- reindexed documents: {reindexed}")
    print(f"- traces imported: {traces}")
    if compare_report is not None:
        print(f"- compare hard failures: {len(compare_report.hard_failures)}")
        print(f"- compare warnings: {len(compare_report.warnings)}")
        print(f"- compare report: {compare_report_path}")


if __name__ == "__main__":
    asyncio.run(main())
