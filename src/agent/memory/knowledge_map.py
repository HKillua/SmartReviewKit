"""Knowledge map memory — tracks per-concept mastery with Ebbinghaus decay."""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class KnowledgeNode(BaseModel):
    concept: str
    chapter: str = ""
    mastery_level: float = Field(default=0.0, ge=0.0, le=1.0)
    quiz_count: int = 0
    correct_count: int = 0
    last_reviewed: Optional[datetime] = None
    review_interval_days: float = 1.0


_CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS knowledge_nodes (
        user_id TEXT NOT NULL,
        concept TEXT NOT NULL,
        data TEXT NOT NULL,
        PRIMARY KEY (user_id, concept)
    )
"""


class KnowledgeMapMemory:
    """SQLite-backed knowledge map with Ebbinghaus forgetting curve (async via aiosqlite)."""

    def __init__(self, db_dir: str = "data/memory") -> None:
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = str(Path(db_dir) / "knowledge_map.db")
        self._init_db_sync()

    def _init_db_sync(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_CREATE_SQL)

    async def get_node(self, user_id: str, concept: str) -> Optional[KnowledgeNode]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT data FROM knowledge_nodes WHERE user_id = ? AND concept = ?",
                (user_id, concept),
            ) as cursor:
                row = await cursor.fetchone()
        if row:
            return KnowledgeNode.model_validate_json(row[0])
        return None

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

        node.last_reviewed = datetime.now()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO knowledge_nodes (user_id, concept, data) VALUES (?, ?, ?)",
                (user_id, concept, node.model_dump_json()),
            )
            await db.commit()

    async def _get_all_nodes(self, user_id: str) -> list[KnowledgeNode]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT data FROM knowledge_nodes WHERE user_id = ?", (user_id,)
            ) as cursor:
                rows = await cursor.fetchall()
        return [KnowledgeNode.model_validate_json(r[0]) for r in rows]

    async def get_weak_nodes(self, user_id: str, threshold: float = 0.5) -> list[KnowledgeNode]:
        nodes = await self._get_all_nodes(user_id)
        return [n for n in nodes if n.mastery_level < threshold]

    async def get_due_for_review(self, user_id: str) -> list[KnowledgeNode]:
        now = datetime.now()
        nodes = await self._get_all_nodes(user_id)
        result = []
        for node in nodes:
            if node.last_reviewed is None:
                result.append(node)
            elif (now - node.last_reviewed).total_seconds() > node.review_interval_days * 86400:
                result.append(node)
        return result

    async def apply_decay(self, user_id: str) -> int:
        """Apply Ebbinghaus forgetting curve decay to all nodes. Returns count of decayed nodes."""
        now = datetime.now()
        decayed = 0
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT data FROM knowledge_nodes WHERE user_id = ?", (user_id,)
            ) as cursor:
                rows = await cursor.fetchall()

            for r in rows:
                node = KnowledgeNode.model_validate_json(r[0])
                if node.last_reviewed is None:
                    continue
                days_since = (now - node.last_reviewed).total_seconds() / 86400
                if days_since < 0.5:
                    continue
                retention = math.exp(-days_since / (node.review_interval_days * 2))
                new_mastery = max(0.0, node.mastery_level * retention)
                if abs(new_mastery - node.mastery_level) > 0.01:
                    node.mastery_level = round(new_mastery, 3)
                    await db.execute(
                        "UPDATE knowledge_nodes SET data = ? WHERE user_id = ? AND concept = ?",
                        (node.model_dump_json(), user_id, node.concept),
                    )
                    decayed += 1
            await db.commit()
        return decayed

    async def close(self) -> None:
        pass
