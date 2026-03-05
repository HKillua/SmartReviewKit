"""Knowledge map memory — tracks per-concept mastery with Ebbinghaus decay."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

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


class KnowledgeMapMemory:
    """SQLite-backed knowledge map with Ebbinghaus forgetting curve."""

    def __init__(self, db_dir: str = "data/memory") -> None:
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = str(Path(db_dir) / "knowledge_map.db")
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_nodes (
                    user_id TEXT NOT NULL,
                    concept TEXT NOT NULL,
                    data TEXT NOT NULL,
                    PRIMARY KEY (user_id, concept)
                )
            """)

    async def get_node(self, user_id: str, concept: str) -> Optional[KnowledgeNode]:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT data FROM knowledge_nodes WHERE user_id = ? AND concept = ?",
                (user_id, concept),
            ).fetchone()
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

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO knowledge_nodes (user_id, concept, data) VALUES (?, ?, ?)",
                (user_id, concept, node.model_dump_json()),
            )

    async def get_weak_nodes(self, user_id: str, threshold: float = 0.5) -> list[KnowledgeNode]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT data FROM knowledge_nodes WHERE user_id = ?", (user_id,)
            ).fetchall()
        nodes = [KnowledgeNode.model_validate_json(r[0]) for r in rows]
        return [n for n in nodes if n.mastery_level < threshold]

    async def get_due_for_review(self, user_id: str) -> list[KnowledgeNode]:
        now = datetime.now()
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT data FROM knowledge_nodes WHERE user_id = ?", (user_id,)
            ).fetchall()
        result = []
        for r in rows:
            node = KnowledgeNode.model_validate_json(r[0])
            if node.last_reviewed is None:
                result.append(node)
            elif (now - node.last_reviewed).total_seconds() > node.review_interval_days * 86400:
                result.append(node)
        return result

    async def apply_decay(self, user_id: str) -> int:
        """Apply Ebbinghaus forgetting curve decay to all nodes. Returns count of decayed nodes."""
        now = datetime.now()
        decayed = 0
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT data FROM knowledge_nodes WHERE user_id = ?", (user_id,)
            ).fetchall()

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
                    conn.execute(
                        "UPDATE knowledge_nodes SET data = ? WHERE user_id = ? AND concept = ?",
                        (node.model_dump_json(), user_id, node.concept),
                    )
                    decayed += 1
        return decayed
