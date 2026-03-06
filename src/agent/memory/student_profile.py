"""Student profile memory — tracks per-user learning state."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
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
    """SQLite-backed student profile store (async via aiosqlite)."""

    def __init__(self, db_dir: str = "data/memory") -> None:
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = str(Path(db_dir) / "profiles.db")
        self._init_db_sync()

    def _init_db_sync(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_CREATE_SQL)

    async def get_profile(self, user_id: str) -> StudentProfile:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT data FROM student_profiles WHERE user_id = ?", (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
        if row:
            return StudentProfile.model_validate_json(row[0])
        return StudentProfile(user_id=user_id)

    async def update_profile(self, user_id: str, updates: dict) -> None:
        profile = await self.get_profile(user_id)
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        profile.last_active = datetime.now()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO student_profiles (user_id, data, updated_at)
                   VALUES (?, ?, ?)""",
                (user_id, profile.model_dump_json(), datetime.now().isoformat()),
            )
            await db.commit()

    async def close(self) -> None:
        pass
