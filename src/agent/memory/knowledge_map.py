"""Knowledge map memory — tracks per-concept mastery with Ebbinghaus decay."""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timezone
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
    """SQLite-backed knowledge map with Ebbinghaus forgetting curve (async via aiosqlite).

    Maintains a persistent connection to avoid per-query open/close overhead.
    """

    def __init__(self, db_dir: str = "data/memory") -> None:
        import asyncio
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = str(Path(db_dir) / "knowledge_map.db")
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

    async def get_node(self, user_id: str, concept: str) -> Optional[KnowledgeNode]:
        db = await self._get_conn()
        async with db.execute(
            "SELECT data FROM knowledge_nodes WHERE user_id = ? AND concept = ?",
            (user_id, concept),
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            try:
                return KnowledgeNode.model_validate_json(row[0])
            except Exception:
                logger.warning("Corrupted knowledge node data for %s/%s", user_id, concept)
        return None

    async def update_mastery(self, user_id: str, concept: str, correct: bool) -> None:
        db = await self._get_conn()
        async with db.execute(
            "SELECT data FROM knowledge_nodes WHERE user_id = ? AND concept = ?",
            (user_id, concept),
        ) as cursor:
            row = await cursor.fetchone()

        if row:
            try:
                node = KnowledgeNode.model_validate_json(row[0])
            except Exception:
                node = KnowledgeNode(concept=concept)
        else:
            node = KnowledgeNode(concept=concept)

        node.quiz_count += 1
        if correct:
            node.correct_count += 1
            node.mastery_level = min(1.0, node.mastery_level + 0.1)
            node.review_interval_days = min(30.0, node.review_interval_days * 2.0)
        else:
            node.mastery_level = max(0.0, node.mastery_level - 0.15)
            node.review_interval_days = max(1.0, node.review_interval_days / 2.0)

        node.last_reviewed = datetime.now(timezone.utc)

        await db.execute(
            "INSERT OR REPLACE INTO knowledge_nodes (user_id, concept, data) VALUES (?, ?, ?)",
            (user_id, concept, node.model_dump_json()),
        )
        await db.commit()

    async def _get_all_nodes(self, user_id: str) -> list[KnowledgeNode]:
        db = await self._get_conn()
        async with db.execute(
            "SELECT data FROM knowledge_nodes WHERE user_id = ?", (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()
        nodes: list[KnowledgeNode] = []
        for r in rows:
            try:
                nodes.append(KnowledgeNode.model_validate_json(r[0]))
            except Exception:
                logger.warning("Skipping corrupted knowledge node for user %s", user_id)
        return nodes

    async def get_weak_nodes(self, user_id: str, threshold: float = 0.5) -> list[KnowledgeNode]:
        nodes = await self._get_all_nodes(user_id)
        return [n for n in nodes if n.mastery_level < threshold]

    async def get_decayed_nodes(
        self,
        user_id: str,
        *,
        threshold: float = 0.45,
        limit: int = 5,
    ) -> list[KnowledgeNode]:
        """Return low-mastery nodes after applying decay, weakest first."""
        await self.apply_decay(user_id)
        nodes = await self._get_all_nodes(user_id)
        filtered = [node for node in nodes if node.mastery_level < threshold]
        filtered.sort(key=lambda node: (node.mastery_level, node.concept))
        return filtered[: max(limit, 0)]

    async def get_due_for_review(self, user_id: str) -> list[KnowledgeNode]:
        now = datetime.now(timezone.utc)
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
        now = datetime.now(timezone.utc)
        db = await self._get_conn()
        async with db.execute(
            "SELECT data FROM knowledge_nodes WHERE user_id = ?", (user_id,)
        ) as cursor:
            rows = await cursor.fetchall()

        batch_updates: list[tuple[str, str, str]] = []
        for r in rows:
            try:
                node = KnowledgeNode.model_validate_json(r[0])
            except Exception:
                continue
            if node.last_reviewed is None:
                continue
            last_reviewed = node.last_reviewed
            if last_reviewed.tzinfo is None:
                last_reviewed = last_reviewed.replace(tzinfo=timezone.utc)
            days_since = (now - last_reviewed).total_seconds() / 86400
            if days_since < 0.5:
                continue
            retention = math.exp(-days_since / (node.review_interval_days * 2))
            new_mastery = max(0.0, node.mastery_level * retention)
            if abs(new_mastery - node.mastery_level) > 0.01:
                node.mastery_level = round(new_mastery, 3)
                batch_updates.append((node.model_dump_json(), user_id, node.concept))

        if batch_updates:
            await db.executemany(
                "UPDATE knowledge_nodes SET data = ? WHERE user_id = ? AND concept = ?",
                batch_updates,
            )
            await db.commit()
        return len(batch_updates)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
