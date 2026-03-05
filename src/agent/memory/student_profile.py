"""Student profile memory — tracks per-user learning state."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StudentProfile(BaseModel):
    user_id: str
    preferences: dict = Field(default_factory=dict)
    weak_topics: list[str] = Field(default_factory=list)
    strong_topics: list[str] = Field(default_factory=list)
    learning_pace: str = "medium"
    total_sessions: int = 0
    total_quizzes: int = 0
    overall_accuracy: float = 0.0
    last_active: Optional[datetime] = None
    notes: str = ""


class StudentProfileMemory:
    """SQLite-backed student profile store."""

    def __init__(self, db_dir: str = "data/memory") -> None:
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = str(Path(db_dir) / "profiles.db")
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS student_profiles (
                    user_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    async def get_profile(self, user_id: str) -> StudentProfile:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT data FROM student_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row:
            return StudentProfile.model_validate_json(row[0])
        return StudentProfile(user_id=user_id)

    async def update_profile(self, user_id: str, updates: dict) -> None:
        profile = await self.get_profile(user_id)
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        profile.last_active = datetime.now()

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO student_profiles (user_id, data, updated_at)
                   VALUES (?, ?, ?)""",
                (user_id, profile.model_dump_json(), datetime.now().isoformat()),
            )
