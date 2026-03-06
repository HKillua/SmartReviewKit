"""Skill memory — stores successful question->tool-chain patterns (async via aiosqlite)."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolUsageRecord(BaseModel):
    question_pattern: str = ""
    tool_chain: list[str] = Field(default_factory=list)
    tool_args: dict = Field(default_factory=dict)
    quality_score: float = 0.0


class SkillMemory:
    """aiosqlite-backed store for tool-usage patterns."""

    def __init__(self, db_dir: str = "data/memory") -> None:
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = str(Path(db_dir) / "skill_memory.db")
        self._conn: Optional[aiosqlite.Connection] = None
        self._conn_lock = asyncio.Lock()
        self._init_db_sync()

    def _init_db_sync(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    question_pattern TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tu_user ON tool_usage(user_id)")

    async def _get_conn(self) -> aiosqlite.Connection:
        async with self._conn_lock:
            if self._conn is None:
                self._conn = await aiosqlite.connect(self._db_path)
            return self._conn

    async def save_usage(self, user_id: str, record: ToolUsageRecord) -> None:
        db = await self._get_conn()
        await db.execute(
            "INSERT INTO tool_usage (user_id, question_pattern, data, created_at) VALUES (?, ?, ?, ?)",
            (user_id, record.question_pattern, record.model_dump_json(), datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()

    async def search_similar(self, user_id: str, question: str, limit: int = 3) -> list[ToolUsageRecord]:
        """Simple keyword overlap search — can be upgraded to semantic later."""
        keywords = set(question.lower().split())
        db = await self._get_conn()
        async with db.execute(
            "SELECT question_pattern, data FROM tool_usage WHERE user_id = ? ORDER BY created_at DESC LIMIT 100",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        scored: list[tuple[float, ToolUsageRecord]] = []
        for pattern, data in rows:
            pattern_words = set(pattern.lower().split())
            overlap = len(keywords & pattern_words)
            if overlap > 0:
                try:
                    scored.append((overlap, ToolUsageRecord.model_validate_json(data)))
                except Exception:
                    logger.warning("Skipping corrupted skill memory record")

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    async def close(self) -> None:
        async with self._conn_lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None
