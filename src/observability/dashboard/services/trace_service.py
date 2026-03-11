"""TraceService – read traces from file or PostgreSQL."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import load_settings, resolve_path
from src.storage.postgres import PostgresExecutor

logger = logging.getLogger(__name__)

DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")


class TraceService:
    """Read-only service for querying recorded traces from configured sources."""

    def __init__(
        self,
        traces_path: Optional[str | Path] = None,
        *,
        source: str | None = None,
        postgres_dsn: str = "",
    ) -> None:
        self.traces_path = Path(traces_path) if traces_path else DEFAULT_TRACES_PATH
        self._source = self._resolve_source(source, postgres_dsn, explicit_path=traces_path is not None)
        self._postgres_dsn = postgres_dsn or self._resolve_postgres_dsn()
        self._db = PostgresExecutor(self._postgres_dsn) if self._source == "postgres" and self._postgres_dsn else None

    def _resolve_source(self, source: Optional[str], postgres_dsn: str, *, explicit_path: bool) -> str:
        if explicit_path and source is None:
            return "file"
        if source in {"file", "postgres"}:
            return source
        try:
            settings = load_settings()
        except Exception:
            return "postgres" if postgres_dsn else "file"
        configured = getattr(settings.observability, "trace_source", "auto")
        if configured == "postgres" and settings.postgres.enabled and settings.postgres.dsn:
            return "postgres"
        if configured == "file":
            return "file"
        if settings.postgres.enabled and settings.postgres.dsn and getattr(settings.observability, "trace_sink", "file") == "postgres":
            return "postgres"
        return "file"

    def _resolve_postgres_dsn(self) -> str:
        try:
            settings = load_settings()
        except Exception:
            return ""
        if settings.postgres.enabled:
            return settings.postgres.dsn
        return ""

    def list_traces(self, trace_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        traces = self._load_all()
        if trace_type:
            traces = [t for t in traces if t.get("trace_type") == trace_type]
        traces.sort(key=lambda t: t.get("started_at", ""), reverse=True)
        return traces[:limit]

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        for trace in self._load_all():
            if trace.get("trace_id") == trace_id:
                return trace
        return None

    def get_stage_timings(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        stages = trace.get("stages", [])
        timings: List[Dict[str, Any]] = []
        for s in stages:
            stage_data = s.get("data", {})
            if isinstance(stage_data, dict) and stage_data:
                normalized_data = stage_data
            else:
                normalized_data = {
                    key: value
                    for key, value in s.items()
                    if key not in {"stage", "timestamp", "elapsed_ms", "data"}
                }
            timings.append(
                {
                    "stage_name": s.get("stage"),
                    "elapsed_ms": s.get("elapsed_ms", 0),
                    "data": normalized_data,
                }
            )
        return timings

    def _load_all(self) -> List[Dict[str, Any]]:
        return self._load_from_postgres() if self._source == "postgres" and self._db is not None else self._load_from_file()

    def _load_from_file(self) -> List[Dict[str, Any]]:
        if not self.traces_path.exists():
            return []
        traces: List[Dict[str, Any]] = []
        with self.traces_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed trace line: %s", line[:80])
        return traces

    def _load_from_postgres(self) -> List[Dict[str, Any]]:
        assert self._db is not None
        with self._db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT trace_id, trace_type, started_at, finished_at, duration_ms, metadata_json, stages_json
                FROM observability_traces
                ORDER BY started_at DESC
                LIMIT 500
                """
            )
            rows = cur.fetchall()
        traces: List[Dict[str, Any]] = []
        for row in rows:
            traces.append(
                {
                    "trace_id": row[0],
                    "trace_type": row[1],
                    "started_at": row[2].isoformat() if hasattr(row[2], "isoformat") else str(row[2]),
                    "finished_at": row[3].isoformat() if row[3] and hasattr(row[3], "isoformat") else (str(row[3]) if row[3] else None),
                    "duration_ms": float(row[4] or 0.0),
                    "metadata": json.loads(row[5] or "{}"),
                    "stages": json.loads(row[6] or "[]"),
                }
            )
        return traces
