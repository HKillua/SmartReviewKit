"""Trace collector with pluggable sinks for file and PostgreSQL persistence."""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from src.core.settings import resolve_path
from src.core.trace.trace_context import TraceContext
from src.storage.postgres import PostgresExecutor, utcnow

logger = logging.getLogger(__name__)

_DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceSink(ABC):
    @abstractmethod
    def collect(self, trace_dict: dict[str, Any]) -> None:
        raise NotImplementedError


class FileTraceSink(TraceSink):
    def __init__(self, traces_path: str | Path = _DEFAULT_TRACES_PATH) -> None:
        self._path = Path(traces_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def collect(self, trace_dict: dict[str, Any]) -> None:
        line = json.dumps(trace_dict, ensure_ascii=False)
        with self._write_lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    @property
    def path(self) -> Path:
        return self._path


class PostgresTraceSink(TraceSink):
    _DDL = [
        """
        CREATE TABLE IF NOT EXISTS observability_traces (
            trace_id TEXT PRIMARY KEY,
            trace_type TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            finished_at TIMESTAMPTZ,
            duration_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            stages_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'ok',
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_pg_observability_traces_type ON observability_traces(trace_type)",
        "CREATE INDEX IF NOT EXISTS idx_pg_observability_traces_started_at ON observability_traces(started_at DESC)",
    ]

    def __init__(self, dsn: str) -> None:
        self._db = PostgresExecutor(dsn)
        self._db.execute_ddl(self._DDL)

    def collect(self, trace_dict: dict[str, Any]) -> None:
        status = "error" if any(stage.get("stage") == "error" for stage in trace_dict.get("stages", [])) else "ok"
        started_at = trace_dict.get("started_at") or utcnow()
        finished_at = trace_dict.get("finished_at")
        duration_ms = float(trace_dict.get("duration_ms") or 0.0)
        metadata = json.dumps(trace_dict.get("metadata", {}), ensure_ascii=False, default=str)
        stages = json.dumps(trace_dict.get("stages", []), ensure_ascii=False, default=str)
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO observability_traces
                (trace_id, trace_type, started_at, finished_at, duration_ms, metadata_json, stages_json, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (trace_id) DO UPDATE
                SET trace_type = EXCLUDED.trace_type,
                    started_at = EXCLUDED.started_at,
                    finished_at = EXCLUDED.finished_at,
                    duration_ms = EXCLUDED.duration_ms,
                    metadata_json = EXCLUDED.metadata_json,
                    stages_json = EXCLUDED.stages_json,
                    status = EXCLUDED.status
                """,
                (
                    trace_dict.get("trace_id"),
                    trace_dict.get("trace_type", "query"),
                    started_at,
                    finished_at,
                    duration_ms,
                    metadata,
                    stages,
                    status,
                    utcnow(),
                ),
            )


class CompositeTraceSink(TraceSink):
    def __init__(self, sinks: list[TraceSink]) -> None:
        self._sinks = list(sinks)

    def collect(self, trace_dict: dict[str, Any]) -> None:
        for sink in self._sinks:
            sink.collect(trace_dict)


class TraceCollector:
    """Collect finished traces and persist them via the configured sink."""

    def __init__(
        self,
        traces_path: str | Path = _DEFAULT_TRACES_PATH,
        *,
        sink: Optional[TraceSink] = None,
        postgres_dsn: str = "",
        sink_mode: str = "file",
    ) -> None:
        if sink is not None:
            self._sink = sink
        else:
            self._sink = self._build_sink(
                traces_path=traces_path,
                postgres_dsn=postgres_dsn,
                sink_mode=sink_mode,
            )
        self._path = Path(traces_path)
        # Backward-compatible lock surface for tests and legacy callers.
        self._write_lock = getattr(self._sink, "_write_lock", threading.Lock())

    @staticmethod
    def _build_sink(*, traces_path: str | Path, postgres_dsn: str, sink_mode: str) -> TraceSink:
        normalized = (sink_mode or "file").strip().lower()
        file_sink = FileTraceSink(traces_path)
        if normalized == "postgres":
            if not postgres_dsn:
                raise ValueError("postgres_dsn is required when trace sink is 'postgres'")
            return PostgresTraceSink(postgres_dsn)
        if normalized == "composite":
            if not postgres_dsn:
                raise ValueError("postgres_dsn is required when trace sink is 'composite'")
            return CompositeTraceSink([file_sink, PostgresTraceSink(postgres_dsn)])
        return file_sink

    @classmethod
    def from_settings(cls, settings: Any, traces_path: str | Path | None = None) -> "TraceCollector":
        postgres_dsn = getattr(getattr(settings, "postgres", None), "dsn", "") if getattr(getattr(settings, "postgres", None), "enabled", False) else ""
        configured_path = traces_path or getattr(getattr(settings, "observability", None), "trace_file", _DEFAULT_TRACES_PATH)
        trace_path = resolve_path(configured_path)
        sink_mode = getattr(getattr(settings, "observability", None), "trace_sink", "file")
        return cls(trace_path, postgres_dsn=postgres_dsn, sink_mode=sink_mode)

    def collect(self, trace: TraceContext) -> None:
        if trace.finished_at is None:
            trace.finish()
        try:
            self._sink.collect(trace.to_dict())
        except OSError:
            logger.exception("Failed to write trace %s", trace.trace_id)
        except Exception:
            logger.exception("Unexpected trace sink failure for %s", trace.trace_id)

    @property
    def path(self) -> Path:
        return self._path
