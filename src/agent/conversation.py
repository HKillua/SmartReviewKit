"""Conversation persistence — file-based and in-memory stores."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.agent.types import Conversation, Message

logger = logging.getLogger(__name__)


class ConversationStore(ABC):
    """Abstract interface for conversation persistence."""

    @abstractmethod
    async def create(self, user_id: str) -> Conversation:
        ...

    @abstractmethod
    async def get(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        ...

    @abstractmethod
    async def update(self, conversation: Conversation) -> None:
        ...

    @abstractmethod
    async def list_conversations(self, user_id: str, limit: int = 20) -> list[Conversation]:
        ...

    @abstractmethod
    async def delete(self, conversation_id: str, user_id: str) -> bool:
        ...


class FileConversationStore(ConversationStore):
    """JSON-file-based conversation store with per-user directory isolation."""

    def __init__(self, base_dir: str = "data/conversations") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._write_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    def _user_dir(self, user_id: str) -> Path:
        hashed = hashlib.sha256(user_id.encode()).hexdigest()[:12]
        d = self._base_dir / hashed
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _conv_path(self, user_id: str, conversation_id: str) -> Path:
        return self._user_dir(user_id) / f"{conversation_id}.json"

    async def create(self, user_id: str) -> Conversation:
        now = datetime.now(timezone.utc)
        conv = Conversation(
            id=uuid.uuid4().hex[:16],
            user_id=user_id,
            messages=[],
            created_at=now,
            updated_at=now,
        )
        await self.update(conv)
        return conv

    _CURRENT_SCHEMA_VERSION = 1

    async def get(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        path = self._conv_path(user_id, conversation_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data = self._migrate_schema(data)
            conv = Conversation.model_validate(data)
            if conv.user_id != user_id:
                return None
            return conv
        except Exception:
            logger.exception("Failed to load conversation %s", conversation_id)
            return None

    @classmethod
    def _migrate_schema(cls, data: dict) -> dict:
        """Apply forward-compatible migrations to conversation data."""
        version = data.get("schema_version", 0)
        if version < 1:
            data.setdefault("schema_version", 1)
            data.setdefault("title", "")
        return data

    async def _get_write_lock(self, conversation_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            if conversation_id not in self._write_locks:
                self._write_locks[conversation_id] = asyncio.Lock()
            return self._write_locks[conversation_id]

    async def update(self, conversation: Conversation) -> None:
        lock = await self._get_write_lock(conversation.id)
        async with lock:
            conversation.updated_at = datetime.now(timezone.utc)
            path = self._conv_path(conversation.user_id, conversation.id)
            data = conversation.model_dump_json(indent=2)
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            fd_closed = False
            try:
                os.write(fd, data.encode("utf-8"))
                os.close(fd)
                fd_closed = True
                os.replace(tmp, str(path))
            except Exception:
                if not fd_closed:
                    os.close(fd)
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise

    async def list_conversations(self, user_id: str, limit: int = 20) -> list[Conversation]:
        d = self._user_dir(user_id)
        convs: list[Conversation] = []
        for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                conv = Conversation.model_validate(data)
                if conv.user_id == user_id:
                    convs.append(conv)
                    if len(convs) >= limit:
                        break
            except Exception:
                continue
        return convs

    async def delete(self, conversation_id: str, user_id: str) -> bool:
        path = self._conv_path(user_id, conversation_id)
        if path.exists():
            path.unlink()
            return True
        return False


class MemoryConversationStore(ConversationStore):
    """In-memory store for testing."""

    def __init__(self) -> None:
        self._store: dict[str, Conversation] = {}
        self._store_lock = asyncio.Lock()

    async def create(self, user_id: str) -> Conversation:
        now = datetime.now(timezone.utc)
        conv = Conversation(
            id=uuid.uuid4().hex[:16],
            user_id=user_id,
            messages=[],
            created_at=now,
            updated_at=now,
        )
        async with self._store_lock:
            self._store[conv.id] = conv
        return conv

    async def get(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        async with self._store_lock:
            conv = self._store.get(conversation_id)
        if conv and conv.user_id == user_id:
            return conv
        return None

    async def update(self, conversation: Conversation) -> None:
        conversation.updated_at = datetime.now(timezone.utc)
        async with self._store_lock:
            self._store[conversation.id] = conversation

    async def list_conversations(self, user_id: str, limit: int = 20) -> list[Conversation]:
        async with self._store_lock:
            items = list(self._store.values())
        return sorted(
            [c for c in items if c.user_id == user_id],
            key=lambda c: c.updated_at,
            reverse=True,
        )[:limit]

    async def delete(self, conversation_id: str, user_id: str) -> bool:
        async with self._store_lock:
            conv = self._store.get(conversation_id)
            if conv and conv.user_id == user_id:
                del self._store[conversation_id]
                return True
        return False
