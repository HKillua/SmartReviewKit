"""Session memory — stores per-session summaries for cross-session recall.

Inspired by CoPaw ReMe's daily-log concept (memory/YYYY-MM-DD.md) but
backed by SQLite for structured queries.  Enables features such as:
  - "你上次问了 TCP 三次握手"
  - Topic-based session history lookup
  - Preference trend tracking across sessions
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SessionSummary(BaseModel):
    """Structured summary of a single learning session."""

    session_id: str = ""
    user_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    topics: list[str] = Field(default_factory=list)
    key_questions: list[str] = Field(default_factory=list)
    mastery_observations: dict[str, str] = Field(default_factory=dict)
    preference_snapshot: dict = Field(default_factory=dict)
    summary_text: str = ""
    quiz_count: int = 0
    quiz_accuracy: float = 0.0


_CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS session_summaries (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
"""


class SessionMemory:
    """SQLite-backed session summary store (async via aiosqlite).

    Maintains a persistent connection to avoid per-query open/close overhead.
    """

    def __init__(self, db_dir: str = "data/memory") -> None:
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = str(Path(db_dir) / "sessions.db")
        self._conn: Optional[aiosqlite.Connection] = None
        self._init_db_sync()

    def _init_db_sync(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_CREATE_SQL)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sess_user ON session_summaries(user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sess_time ON session_summaries(created_at)"
            )

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(self._db_path)
        return self._conn

    async def save_session(self, user_id: str, summary: SessionSummary) -> None:
        summary.user_id = user_id
        db = await self._get_conn()
        await db.execute(
            "INSERT OR REPLACE INTO session_summaries (id, user_id, data, created_at) VALUES (?, ?, ?, ?)",
            (
                summary.session_id,
                user_id,
                summary.model_dump_json(),
                summary.timestamp.isoformat(),
            ),
        )
        await db.commit()
        logger.debug("Saved session summary %s for user %s", summary.session_id, user_id)

    async def get_recent_sessions(
        self, user_id: str, limit: int = 5
    ) -> list[SessionSummary]:
        db = await self._get_conn()
        async with db.execute(
            "SELECT data FROM session_summaries WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        results: list[SessionSummary] = []
        for r in rows:
            try:
                results.append(SessionSummary.model_validate_json(r[0]))
            except Exception:
                logger.warning("Skipping corrupted session summary")
        return results

    async def get_topic_history(
        self, user_id: str, topic: str, limit: int = 10
    ) -> list[SessionSummary]:
        """Return sessions where *topic* appeared (SQL LIKE + in-memory verification)."""
        db = await self._get_conn()
        async with db.execute(
            "SELECT data FROM session_summaries WHERE user_id = ? AND data LIKE ? ORDER BY created_at DESC LIMIT ?",
            (user_id, f"%{topic}%", limit * 3),
        ) as cursor:
            rows = await cursor.fetchall()
        topic_lower = topic.lower()
        matched: list[SessionSummary] = []
        for r in rows:
            try:
                s = SessionSummary.model_validate_json(r[0])
            except Exception:
                continue
            if any(topic_lower in t.lower() for t in s.topics):
                matched.append(s)
                if len(matched) >= limit:
                    break
        return matched

    async def search_sessions(
        self, user_id: str, query: str, limit: int = 5
    ) -> list[SessionSummary]:
        """Keyword search across session summaries and topics (SQL pre-filter)."""
        db = await self._get_conn()
        async with db.execute(
            "SELECT data FROM session_summaries WHERE user_id = ? AND data LIKE ? ORDER BY created_at DESC LIMIT ?",
            (user_id, f"%{query}%", limit * 5),
        ) as cursor:
            rows = await cursor.fetchall()
        query_lower = query.lower()
        scored: list[tuple[float, SessionSummary]] = []
        for r in rows:
            try:
                s = SessionSummary.model_validate_json(r[0])
            except Exception:
                continue
            score = 0.0
            searchable = " ".join(s.topics + s.key_questions) + " " + s.summary_text
            searchable_lower = searchable.lower()
            for token in query_lower.split():
                if token in searchable_lower:
                    score += 1.0
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:limit]]

    async def get_session_count(self, user_id: str) -> int:
        db = await self._get_conn()
        async with db.execute(
            "SELECT COUNT(*) FROM session_summaries WHERE user_id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
