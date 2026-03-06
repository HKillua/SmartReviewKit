"""Student profile memory — tracks per-user learning state."""

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


def _default_preferences() -> dict:
    return {
        "detail_level": "normal",    # concise / normal / detailed
        "style": "default",          # default / exam_focused / example_heavy
        "quiz_difficulty": "medium", # easy / medium / hard
    }


class StudentProfile(BaseModel):
    user_id: str
    preferences: dict = Field(default_factory=_default_preferences)
    weak_topics: list[str] = Field(default_factory=list)
    strong_topics: list[str] = Field(default_factory=list)
    learning_pace: str = "medium"
    total_sessions: int = 0
    total_quizzes: int = 0
    overall_accuracy: float = 0.0
    last_active: Optional[datetime] = None
    notes: str = ""


_CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS student_profiles (
        user_id TEXT PRIMARY KEY,
        data TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
"""


class StudentProfileMemory:
    """SQLite-backed student profile store (async via aiosqlite).

    Maintains a persistent connection to avoid per-query open/close overhead.
    """

    def __init__(self, db_dir: str = "data/memory") -> None:
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = str(Path(db_dir) / "profiles.db")
        self._conn: Optional[aiosqlite.Connection] = None
        self._conn_lock = asyncio.Lock()
        self._init_db_sync()

    def _init_db_sync(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_CREATE_SQL)

    async def _get_conn(self) -> aiosqlite.Connection:
        async with self._conn_lock:
            if self._conn is None:
                self._conn = await aiosqlite.connect(self._db_path)
            return self._conn

    async def get_profile(self, user_id: str) -> StudentProfile:
        db = await self._get_conn()
        async with db.execute(
            "SELECT data FROM student_profiles WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            try:
                return StudentProfile.model_validate_json(row[0])
            except Exception:
                logger.warning("Corrupted profile data for %s, returning default", user_id)
        return StudentProfile(user_id=user_id)

    _ALLOWED_UPDATE_FIELDS = frozenset({
        "preferences", "weak_topics", "strong_topics", "learning_pace",
        "total_sessions", "total_quizzes", "overall_accuracy", "last_active", "notes",
    })

    async def update_profile(self, user_id: str, updates: dict) -> None:
        db = await self._get_conn()
        async with db.execute(
            "SELECT data FROM student_profiles WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if row:
            try:
                profile = StudentProfile.model_validate_json(row[0])
            except Exception:
                profile = StudentProfile(user_id=user_id)
        else:
            profile = StudentProfile(user_id=user_id)

        for key, value in updates.items():
            if key in self._ALLOWED_UPDATE_FIELDS:
                setattr(profile, key, value)
        profile.last_active = datetime.now(timezone.utc)

        await db.execute(
            """INSERT OR REPLACE INTO student_profiles (user_id, data, updated_at)
               VALUES (?, ?, ?)""",
            (user_id, profile.model_dump_json(), datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
