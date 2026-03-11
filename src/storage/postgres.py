"""PostgreSQL helpers for compatibility-mode storage migration."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, TypeVar

T = TypeVar("T")

try:
    import psycopg

    PSYCOPG_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency in dev
    psycopg = None
    PSYCOPG_AVAILABLE = False


class PostgresUnavailableError(ImportError):
    """Raised when PostgreSQL backends are requested without psycopg installed."""


def ensure_psycopg() -> None:
    if not PSYCOPG_AVAILABLE:
        raise PostgresUnavailableError(
            "psycopg is required for PostgreSQL-backed storage. "
            "Install with: pip install 'psycopg[binary]'"
        )


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PostgresExecutor:
    """Thin sync psycopg wrapper with async helpers for repository stores."""

    def __init__(self, dsn: str) -> None:
        ensure_psycopg()
        if not dsn:
            raise ValueError("PostgreSQL DSN cannot be empty")
        self._dsn = dsn

    @contextmanager
    def connect(self) -> Iterator[Any]:
        conn = psycopg.connect(self._dsn, autocommit=True)
        try:
            yield conn
        finally:
            conn.close()

    async def run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)

    def execute_ddl(self, statements: list[str]) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
