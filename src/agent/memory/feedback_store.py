"""FeedbackStore — persists user feedback (thumbs up/down, comments) to SQLite."""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeedbackStore:
    """SQLite-backed store for user feedback on agent responses."""

    def __init__(self, db_path: str = "data/db/feedback.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                message_index INTEGER DEFAULT -1,
                rating TEXT NOT NULL CHECK(rating IN ('up', 'down')),
                comment TEXT DEFAULT '',
                query TEXT DEFAULT '',
                response_preview TEXT DEFAULT '',
                created_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def add(
        self,
        user_id: str,
        conversation_id: str,
        rating: str,
        message_index: int = -1,
        comment: str = "",
        query: str = "",
        response_preview: str = "",
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO feedback
               (user_id, conversation_id, message_index, rating, comment, query, response_preview, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, conversation_id, message_index, rating, comment, query, response_preview, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict[str, int]:
        row = self._conn.execute(
            """SELECT
                 COUNT(*) AS total,
                 SUM(CASE WHEN rating='up' THEN 1 ELSE 0 END) AS up,
                 SUM(CASE WHEN rating='down' THEN 1 ELSE 0 END) AS down
               FROM feedback"""
        ).fetchone()
        return {"total": row["total"], "up": row["up"] or 0, "down": row["down"] or 0}

    def close(self) -> None:
        self._conn.close()
