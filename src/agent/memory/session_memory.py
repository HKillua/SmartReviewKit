"""Session memory — stores per-session summaries for cross-session recall.

Inspired by CoPaw ReMe's daily-log concept (memory/YYYY-MM-DD.md) but
backed by SQLite for structured queries.  Enables features such as:
  - "你上次问了 TCP 三次握手"
  - Topic-based session history lookup
  - Preference trend tracking across sessions
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

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


class SessionMemory:
    """SQLite-backed session summary store."""

    def __init__(self, db_dir: str = "data/memory") -> None:
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = str(Path(db_dir) / "sessions.db")
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_summaries (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sess_user ON session_summaries(user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sess_time ON session_summaries(created_at)"
            )

    async def save_session(self, user_id: str, summary: SessionSummary) -> None:
        summary.user_id = user_id
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO session_summaries (id, user_id, data, created_at) VALUES (?, ?, ?, ?)",
                (
                    summary.session_id,
                    user_id,
                    summary.model_dump_json(),
                    summary.timestamp.isoformat(),
                ),
            )
        logger.debug("Saved session summary %s for user %s", summary.session_id, user_id)

    async def get_recent_sessions(
        self, user_id: str, limit: int = 5
    ) -> list[SessionSummary]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT data FROM session_summaries WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [SessionSummary.model_validate_json(r[0]) for r in rows]

    async def get_topic_history(
        self, user_id: str, topic: str, limit: int = 10
    ) -> list[SessionSummary]:
        """Return sessions where *topic* appeared (case-insensitive substring match)."""
        all_sessions = await self.get_recent_sessions(user_id, limit=100)
        topic_lower = topic.lower()
        matched = [
            s
            for s in all_sessions
            if any(topic_lower in t.lower() for t in s.topics)
        ]
        return matched[:limit]

    async def search_sessions(
        self, user_id: str, query: str, limit: int = 5
    ) -> list[SessionSummary]:
        """Keyword search across session summaries and topics."""
        all_sessions = await self.get_recent_sessions(user_id, limit=100)
        query_lower = query.lower()
        scored: list[tuple[float, SessionSummary]] = []
        for s in all_sessions:
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
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM session_summaries WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row[0] if row else 0
