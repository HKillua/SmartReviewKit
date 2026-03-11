"""PostgreSQL-backed storage adapters for production deployments."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agent.conversation import ConversationStore
from src.agent.memory.error_memory import ErrorMemory, ErrorRecord
from src.agent.memory.feedback_store import FeedbackStore
from src.agent.memory.knowledge_map import KnowledgeMapMemory, KnowledgeNode
from src.agent.memory.session_memory import SessionMemory, SessionSummary
from src.agent.memory.skill_memory import SkillMemory, ToolUsageRecord
from src.agent.memory.student_profile import StudentProfile, StudentProfileMemory
from src.agent.types import Conversation
from src.libs.loader.file_integrity import FileIntegrityChecker
from src.storage.postgres import PostgresExecutor, utcnow

logger = logging.getLogger(__name__)


def _to_json(model_or_value: Any) -> str:
    if hasattr(model_or_value, "model_dump_json"):
        return model_or_value.model_dump_json()
    return json.dumps(model_or_value, ensure_ascii=False, default=str)


class PostgresConversationStore(ConversationStore):
    """PostgreSQL implementation of conversation persistence."""

    _DDL = [
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            data TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC)",
    ]

    def __init__(self, dsn: str) -> None:
        self._db = PostgresExecutor(dsn)
        self._db.execute_ddl(self._DDL)
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _get_write_lock(self, conversation_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(conversation_id, asyncio.Lock())

    async def create(self, user_id: str) -> Conversation:
        now = utcnow()
        conv = Conversation(
            id=uuid.uuid4().hex[:16],
            user_id=user_id,
            messages=[],
            created_at=now,
            updated_at=now,
        )
        await self.update(conv)
        return conv

    async def get(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        def _get() -> Optional[Conversation]:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT data FROM conversations WHERE id = %s AND user_id = %s",
                    (conversation_id, user_id),
                )
                row = cur.fetchone()
            if not row:
                return None
            return Conversation.model_validate_json(row[0])

        try:
            return await self._db.run(_get)
        except Exception:
            logger.exception("Failed to load conversation %s", conversation_id)
            return None

    async def update(self, conversation: Conversation) -> None:
        lock = await self._get_write_lock(conversation.id)
        async with lock:
            conversation.updated_at = utcnow()

            def _update() -> None:
                with self._db.connect() as conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO conversations (id, user_id, data, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                        SET user_id = EXCLUDED.user_id,
                            data = EXCLUDED.data,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            conversation.id,
                            conversation.user_id,
                            conversation.model_dump_json(indent=2),
                            conversation.created_at,
                            conversation.updated_at,
                        ),
                    )

            await self._db.run(_update)

    async def list_conversations(self, user_id: str, limit: int = 20) -> list[Conversation]:
        def _list() -> list[Conversation]:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT data FROM conversations
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
            return [Conversation.model_validate_json(row[0]) for row in rows]

        return await self._db.run(_list)

    async def delete(self, conversation_id: str, user_id: str) -> bool:
        def _delete() -> bool:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversations WHERE id = %s AND user_id = %s",
                    (conversation_id, user_id),
                )
                return cur.rowcount > 0

        return await self._db.run(_delete)


class PostgresStudentProfileMemory(StudentProfileMemory):
    """PostgreSQL-backed student profile store."""

    _DDL = [
        """
        CREATE TABLE IF NOT EXISTS student_profiles (
            user_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    ]

    def __init__(self, dsn: str) -> None:
        self._db = PostgresExecutor(dsn)
        self._db.execute_ddl(self._DDL)

    async def get_profile(self, user_id: str) -> StudentProfile:
        def _get() -> StudentProfile:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT data FROM student_profiles WHERE user_id = %s", (user_id,))
                row = cur.fetchone()
            if row:
                try:
                    return StudentProfile.model_validate_json(row[0])
                except Exception:
                    logger.warning("Corrupted profile data for %s, returning default", user_id)
            return StudentProfile(user_id=user_id)

        return await self._db.run(_get)

    async def update_profile(self, user_id: str, updates: dict) -> None:
        profile = await self.get_profile(user_id)
        for key, value in updates.items():
            if key in self._ALLOWED_UPDATE_FIELDS:
                setattr(profile, key, value)
        profile.last_active = utcnow()

        def _update() -> None:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO student_profiles (user_id, data, updated_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE
                    SET data = EXCLUDED.data,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (user_id, profile.model_dump_json(), utcnow()),
                )

        await self._db.run(_update)

    async def close(self) -> None:
        return None


class PostgresSessionMemory(SessionMemory):
    """PostgreSQL-backed session summary store."""

    _DDL = [
        """
        CREATE TABLE IF NOT EXISTS session_summaries (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            data TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_pg_sess_user ON session_summaries(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_pg_sess_time ON session_summaries(created_at DESC)",
    ]

    def __init__(self, dsn: str) -> None:
        self._db = PostgresExecutor(dsn)
        self._db.execute_ddl(self._DDL)

    async def save_session(self, user_id: str, summary: SessionSummary) -> None:
        summary.user_id = user_id

        def _save() -> None:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO session_summaries (id, user_id, data, created_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET user_id = EXCLUDED.user_id,
                        data = EXCLUDED.data,
                        created_at = EXCLUDED.created_at
                    """,
                    (summary.session_id, user_id, summary.model_dump_json(), summary.timestamp),
                )

        await self._db.run(_save)

    async def get_recent_sessions(self, user_id: str, limit: int = 5) -> list[SessionSummary]:
        def _get() -> list[SessionSummary]:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT data FROM session_summaries
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
            return [SessionSummary.model_validate_json(row[0]) for row in rows]

        return await self._db.run(_get)

    async def get_topic_history(self, user_id: str, topic: str, limit: int = 10) -> list[SessionSummary]:
        sessions = await self.get_recent_sessions(user_id, limit=limit * 4)
        topic_lower = topic.lower()
        matched = [s for s in sessions if any(topic_lower in t.lower() for t in s.topics)]
        return matched[:limit]

    async def search_sessions(self, user_id: str, query: str, limit: int = 5) -> list[SessionSummary]:
        sessions = await self.get_recent_sessions(user_id, limit=limit * 10)
        query_lower = query.lower()
        scored: list[tuple[float, SessionSummary]] = []
        for session in sessions:
            searchable = " ".join(session.topics + session.key_questions) + " " + session.summary_text
            searchable_lower = searchable.lower()
            score = 0.0
            for token in query_lower.split():
                if token in searchable_lower:
                    score += 1.0
            if score > 0:
                scored.append((score, session))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    async def get_session_count(self, user_id: str) -> int:
        def _count() -> int:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM session_summaries WHERE user_id = %s", (user_id,))
                row = cur.fetchone()
            return int(row[0]) if row else 0

        return await self._db.run(_count)

    async def close(self) -> None:
        return None


class PostgresErrorMemory(ErrorMemory):
    """PostgreSQL-backed incorrect answer store."""

    _DDL = [
        """
        CREATE TABLE IF NOT EXISTS error_records (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            data TEXT NOT NULL,
            mastered BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_pg_err_user ON error_records(user_id)",
    ]

    def __init__(self, dsn: str) -> None:
        self._db = PostgresExecutor(dsn)
        self._db.execute_ddl(self._DDL)

    async def add_error(self, user_id: str, record: ErrorRecord) -> None:
        record.user_id = user_id

        def _add() -> None:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO error_records (id, user_id, data, mastered, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET user_id = EXCLUDED.user_id,
                        data = EXCLUDED.data,
                        mastered = EXCLUDED.mastered,
                        created_at = EXCLUDED.created_at
                    """,
                    (record.id, user_id, record.model_dump_json(), record.mastered, record.created_at),
                )

        await self._db.run(_add)

    async def get_errors(
        self,
        user_id: str,
        topic: Optional[str] = None,
        mastered: Optional[bool] = None,
        limit: int = 50,
    ) -> list[ErrorRecord]:
        def _get() -> list[ErrorRecord]:
            query = "SELECT data FROM error_records WHERE user_id = %s"
            params: list[Any] = [user_id]
            if mastered is not None:
                query += " AND mastered = %s"
                params.append(mastered)
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit * 4 if topic else limit)

            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

            records = [ErrorRecord.model_validate_json(row[0]) for row in rows]
            if topic:
                topic_lower = topic.lower()
                records = [r for r in records if topic_lower in (r.topic or "").lower()]
            return records[:limit]

        return await self._db.run(_get)

    async def mark_mastered(self, error_id: str) -> None:
        def _mark() -> None:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT data FROM error_records WHERE id = %s", (error_id,))
                row = cur.fetchone()
                if not row:
                    return
                record = ErrorRecord.model_validate_json(row[0])
                record.mastered = True
                record.mastered_at = datetime.now()
                cur.execute(
                    "UPDATE error_records SET data = %s, mastered = TRUE WHERE id = %s",
                    (record.model_dump_json(), error_id),
                )

        await self._db.run(_mark)

    async def get_weak_concepts(self, user_id: str) -> list[str]:
        errors = await self.get_errors(user_id, mastered=False, limit=200)
        counter: Counter[str] = Counter()
        for record in errors:
            for concept in record.concepts:
                counter[concept] += 1
        return [concept for concept, _ in counter.most_common(20)]

    async def close(self) -> None:
        return None


class PostgresKnowledgeMapMemory(KnowledgeMapMemory):
    """PostgreSQL-backed concept mastery store."""

    _DDL = [
        """
        CREATE TABLE IF NOT EXISTS knowledge_nodes (
            user_id TEXT NOT NULL,
            concept TEXT NOT NULL,
            data TEXT NOT NULL,
            PRIMARY KEY (user_id, concept)
        )
        """
    ]

    def __init__(self, dsn: str) -> None:
        self._db = PostgresExecutor(dsn)
        self._db.execute_ddl(self._DDL)

    async def get_node(self, user_id: str, concept: str) -> Optional[KnowledgeNode]:
        def _get() -> Optional[KnowledgeNode]:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT data FROM knowledge_nodes WHERE user_id = %s AND concept = %s",
                    (user_id, concept),
                )
                row = cur.fetchone()
            if not row:
                return None
            return KnowledgeNode.model_validate_json(row[0])

        return await self._db.run(_get)

    async def update_mastery(self, user_id: str, concept: str, correct: bool) -> None:
        node = await self.get_node(user_id, concept)
        if node is None:
            node = KnowledgeNode(concept=concept)

        node.quiz_count += 1
        if correct:
            node.correct_count += 1
            node.mastery_level = min(1.0, node.mastery_level + 0.1)
            node.review_interval_days = min(30.0, node.review_interval_days * 2.0)
        else:
            node.mastery_level = max(0.0, node.mastery_level - 0.15)
            node.review_interval_days = max(1.0, node.review_interval_days / 2.0)
        node.last_reviewed = utcnow()

        def _save() -> None:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO knowledge_nodes (user_id, concept, data)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, concept) DO UPDATE
                    SET data = EXCLUDED.data
                    """,
                    (user_id, concept, node.model_dump_json()),
                )

        await self._db.run(_save)

    async def _get_all_nodes(self, user_id: str) -> list[KnowledgeNode]:
        def _get() -> list[KnowledgeNode]:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT data FROM knowledge_nodes WHERE user_id = %s", (user_id,))
                rows = cur.fetchall()
            return [KnowledgeNode.model_validate_json(row[0]) for row in rows]

        return await self._db.run(_get)

    async def close(self) -> None:
        return None


class PostgresSkillMemory(SkillMemory):
    """PostgreSQL-backed tool-usage memory."""

    _DDL = [
        """
        CREATE TABLE IF NOT EXISTS tool_usage (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            question_pattern TEXT NOT NULL,
            data TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_pg_tool_usage_user ON tool_usage(user_id)",
    ]

    def __init__(self, dsn: str) -> None:
        self._db = PostgresExecutor(dsn)
        self._db.execute_ddl(self._DDL)

    async def save_usage(self, user_id: str, record: ToolUsageRecord) -> None:
        def _save() -> None:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tool_usage (user_id, question_pattern, data, created_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (user_id, record.question_pattern, record.model_dump_json(), utcnow()),
                )

        await self._db.run(_save)

    async def search_similar(self, user_id: str, question: str, limit: int = 3) -> list[ToolUsageRecord]:
        keywords = set(question.lower().split())

        def _search() -> list[ToolUsageRecord]:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT data FROM tool_usage
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT 100
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()
            scored: list[tuple[float, ToolUsageRecord]] = []
            for (data,) in rows:
                record = ToolUsageRecord.model_validate_json(data)
                overlap = len(keywords & set(record.question_pattern.lower().split()))
                if overlap > 0:
                    scored.append((overlap, record))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [item[1] for item in scored[:limit]]

        return await self._db.run(_search)

    async def close(self) -> None:
        return None


class PostgresFeedbackStore(FeedbackStore):
    """PostgreSQL-backed feedback store."""

    _DDL = [
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            message_index INTEGER DEFAULT -1,
            rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
            comment TEXT DEFAULT '',
            query TEXT DEFAULT '',
            response_preview TEXT DEFAULT '',
            created_at DOUBLE PRECISION NOT NULL
        )
        """
    ]

    def __init__(self, dsn: str) -> None:
        self._db = PostgresExecutor(dsn)
        self._db.execute_ddl(self._DDL)

    async def add(
        self,
        user_id: str,
        conversation_id: str,
        rating: str,
        message_index: int = -1,
        comment: str = "",
        query: str = "",
        response_preview: str = "",
    ) -> int:
        def _add() -> int:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feedback
                    (user_id, conversation_id, message_index, rating, comment, query, response_preview, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, EXTRACT(EPOCH FROM NOW()))
                    RETURNING id
                    """,
                    (user_id, conversation_id, message_index, rating, comment, query, response_preview),
                )
                row = cur.fetchone()
            return int(row[0])

        return await self._db.run(_add)

    async def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        def _list() -> List[Dict[str, Any]]:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT * FROM feedback ORDER BY created_at DESC LIMIT %s", (limit,))
                columns = [desc.name for desc in cur.description]
                rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]

        return await self._db.run(_list)

    async def stats(self) -> Dict[str, int]:
        def _stats() -> Dict[str, int]:
            with self._db.connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS total,
                        SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END) AS up,
                        SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END) AS down
                    FROM feedback
                    """
                )
                row = cur.fetchone()
            return {"total": int(row[0]), "up": int(row[1] or 0), "down": int(row[2] or 0)}

        return await self._db.run(_stats)

    async def close(self) -> None:
        return None


class PostgresIntegrityChecker(FileIntegrityChecker):
    """PostgreSQL-backed ingestion history checker."""

    _DDL = [
        """
        CREATE TABLE IF NOT EXISTS ingestion_history (
            file_hash TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            status TEXT NOT NULL,
            collection TEXT,
            error_msg TEXT,
            object_uri TEXT,
            processed_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_pg_ingestion_status ON ingestion_history(status)",
    ]

    def __init__(self, dsn: str) -> None:
        self._db = PostgresExecutor(dsn)
        self._db.execute_ddl(self._DDL)

    def compute_sha256(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise IOError(f"Path is not a file: {file_path}")
        sha256_hash = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT status FROM ingestion_history WHERE file_hash = %s", (file_hash,))
            row = cur.fetchone()
        return bool(row and row[0] == "success")

    def mark_success(self, file_hash: str, file_path: str, collection: Optional[str] = None) -> None:
        now = utcnow()
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_history
                (file_hash, file_path, status, collection, error_msg, object_uri, processed_at, updated_at)
                VALUES (%s, %s, 'success', %s, '', NULL, %s, %s)
                ON CONFLICT (file_hash) DO UPDATE
                SET file_path = EXCLUDED.file_path,
                    status = EXCLUDED.status,
                    collection = EXCLUDED.collection,
                    error_msg = EXCLUDED.error_msg,
                    updated_at = EXCLUDED.updated_at
                """,
                (file_hash, file_path, collection, now, now),
            )

    def mark_failed(self, file_hash: str, file_path: str, error_msg: str) -> None:
        now = utcnow()
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_history
                (file_hash, file_path, status, collection, error_msg, object_uri, processed_at, updated_at)
                VALUES (%s, %s, 'failed', NULL, %s, NULL, %s, %s)
                ON CONFLICT (file_hash) DO UPDATE
                SET file_path = EXCLUDED.file_path,
                    status = EXCLUDED.status,
                    error_msg = EXCLUDED.error_msg,
                    updated_at = EXCLUDED.updated_at
                """,
                (file_hash, file_path, error_msg, now, now),
            )

    def remove_record(self, file_hash: str) -> bool:
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM ingestion_history WHERE file_hash = %s", (file_hash,))
            return cur.rowcount > 0

    def list_processed(self, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT file_hash, file_path, collection, processed_at, updated_at, object_uri
            FROM ingestion_history
            WHERE status = 'success'
        """
        params: list[Any] = []
        if collection:
            query += " AND collection = %s"
            params.append(collection)
        query += " ORDER BY updated_at DESC"

        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [
            {
                "file_hash": row[0],
                "file_path": row[1],
                "collection": row[2],
                "processed_at": row[3].isoformat() if hasattr(row[3], "isoformat") else str(row[3]),
                "updated_at": row[4].isoformat() if hasattr(row[4], "isoformat") else str(row[4]),
                "object_uri": row[5],
            }
            for row in rows
        ]


class PostgresDocumentRegistry:
    """Document metadata registry for production ingestion lifecycle."""

    _DDL = [
        """
        CREATE TABLE IF NOT EXISTS document_registry (
            file_hash TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            collection TEXT NOT NULL,
            status TEXT NOT NULL,
            object_uri TEXT,
            metadata_json TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_pg_document_collection ON document_registry(collection)",
    ]

    def __init__(self, dsn: str) -> None:
        self._db = PostgresExecutor(dsn)
        self._db.execute_ddl(self._DDL)

    def upsert_document(
        self,
        file_hash: str,
        file_path: str,
        collection: str,
        status: str,
        object_uri: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = utcnow()
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_registry
                (file_hash, file_path, collection, status, object_uri, metadata_json, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (file_hash) DO UPDATE
                SET file_path = EXCLUDED.file_path,
                    collection = EXCLUDED.collection,
                    status = EXCLUDED.status,
                    object_uri = EXCLUDED.object_uri,
                    metadata_json = EXCLUDED.metadata_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    file_hash,
                    file_path,
                    collection,
                    status,
                    object_uri,
                    _to_json(metadata or {}),
                    now,
                    now,
                ),
            )

    def delete_document(self, file_hash: str) -> bool:
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM document_registry WHERE file_hash = %s", (file_hash,))
            return cur.rowcount > 0

    def get_document(self, file_hash: str) -> Optional[Dict[str, Any]]:
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT file_hash, file_path, collection, status, object_uri, metadata_json
                FROM document_registry
                WHERE file_hash = %s
                """,
                (file_hash,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "file_hash": row[0],
            "file_path": row[1],
            "collection": row[2],
            "status": row[3],
            "object_uri": row[4],
            "metadata": json.loads(row[5] or "{}"),
        }

    def list_documents(self, collection: Optional[str] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT file_hash, file_path, collection, status, object_uri, metadata_json
            FROM document_registry
        """
        params: list[Any] = []
        if collection:
            query += " WHERE collection = %s"
            params.append(collection)
        query += " ORDER BY updated_at DESC"
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [
            {
                "file_hash": row[0],
                "file_path": row[1],
                "collection": row[2],
                "status": row[3],
                "object_uri": row[4],
                "metadata": json.loads(row[5] or "{}"),
            }
            for row in rows
        ]


class IngestionTaskStore:
    """PostgreSQL-backed ingestion task status store."""

    _DDL = [
        """
        CREATE TABLE IF NOT EXISTS ingestion_tasks (
            task_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            collection TEXT NOT NULL,
            status TEXT NOT NULL,
            object_uri TEXT,
            error_msg TEXT,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_pg_ingestion_tasks_status ON ingestion_tasks(status)",
    ]

    def __init__(self, dsn: str) -> None:
        self._db = PostgresExecutor(dsn)
        self._db.execute_ddl(self._DDL)

    def create_task(self, file_path: str, collection: str, object_uri: str = "") -> str:
        task_id = uuid.uuid4().hex[:16]
        now = utcnow()
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_tasks
                (task_id, file_path, collection, status, object_uri, error_msg, created_at, updated_at)
                VALUES (%s, %s, %s, 'queued', %s, '', %s, %s)
                """,
                (task_id, file_path, collection, object_uri, now, now),
            )
        return task_id

    def update_status(self, task_id: str, status: str, error_msg: str = "") -> None:
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingestion_tasks
                SET status = %s, error_msg = %s, updated_at = %s
                WHERE task_id = %s
                """,
                (status, error_msg, utcnow(), task_id),
            )

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT task_id, file_path, collection, status, object_uri, error_msg, created_at, updated_at
                FROM ingestion_tasks
                WHERE task_id = %s
                """,
                (task_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "task_id": row[0],
            "file_path": row[1],
            "collection": row[2],
            "status": row[3],
            "object_uri": row[4],
            "error_msg": row[5],
            "created_at": row[6].isoformat() if hasattr(row[6], "isoformat") else str(row[6]),
            "updated_at": row[7].isoformat() if hasattr(row[7], "isoformat") else str(row[7]),
        }
