"""Error memory — stores incorrect answers for spaced-repetition review."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ErrorRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_id: str = ""
    question: str = ""
    question_type: str = ""
    topic: str = ""
    concepts: list[str] = Field(default_factory=list)
    user_answer: str = ""
    correct_answer: str = ""
    explanation: str = ""
    error_type: str = "conceptual"
    difficulty: int = 3
    mastered: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    mastered_at: Optional[datetime] = None


_CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS error_records (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        data TEXT NOT NULL,
        mastered INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    )
"""


class ErrorMemory:
    """SQLite-backed error record store (async via aiosqlite)."""

    def __init__(self, db_dir: str = "data/memory") -> None:
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = str(Path(db_dir) / "errors.db")
        self._init_db_sync()

    def _init_db_sync(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_CREATE_SQL)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_err_user ON error_records(user_id)")

    async def add_error(self, user_id: str, record: ErrorRecord) -> None:
        record.user_id = user_id
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO error_records (id, user_id, data, mastered, created_at) VALUES (?, ?, ?, ?, ?)",
                (record.id, user_id, record.model_dump_json(), int(record.mastered), record.created_at.isoformat()),
            )
            await db.commit()

    async def get_errors(
        self,
        user_id: str,
        topic: Optional[str] = None,
        mastered: Optional[bool] = None,
        limit: int = 50,
    ) -> list[ErrorRecord]:
        query = "SELECT data FROM error_records WHERE user_id = ?"
        params: list = [user_id]
        if mastered is not None:
            query += " AND mastered = ?"
            params.append(int(mastered))
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

        records = [ErrorRecord.model_validate_json(r[0]) for r in rows]
        if topic:
            records = [r for r in records if topic.lower() in r.topic.lower()]
        return records

    async def mark_mastered(self, error_id: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("SELECT data FROM error_records WHERE id = ?", (error_id,)) as cursor:
                row = await cursor.fetchone()
            if row:
                record = ErrorRecord.model_validate_json(row[0])
                record.mastered = True
                record.mastered_at = datetime.now()
                await db.execute(
                    "UPDATE error_records SET data = ?, mastered = 1 WHERE id = ?",
                    (record.model_dump_json(), error_id),
                )
                await db.commit()

    async def get_weak_concepts(self, user_id: str) -> list[str]:
        errors = await self.get_errors(user_id, mastered=False, limit=200)
        counter: Counter = Counter()
        for e in errors:
            for c in e.concepts:
                counter[c] += 1
        return [concept for concept, _ in counter.most_common(20)]

    async def close(self) -> None:
        pass
