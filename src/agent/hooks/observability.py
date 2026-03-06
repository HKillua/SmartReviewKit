"""Observability — lightweight metrics collection and span tracing."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Span(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    start_time: float = Field(default_factory=time.monotonic)
    end_time: Optional[float] = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    parent_id: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class MetricsCollector:
    """Collects counters, histograms, and spans, writing to JSONL periodically."""

    def __init__(self, metrics_path: str = "logs/metrics.jsonl") -> None:
        self._path = Path(metrics_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._counters: dict[str, float] = {}
        self._spans: list[Span] = []

    def record_counter(self, name: str, value: float = 1.0, tags: dict[str, str] | None = None) -> None:
        key = name
        self._counters[key] = self._counters.get(key, 0.0) + value
        self._write_metric("counter", name, value, tags)

    def record_histogram(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        self._write_metric("histogram", name, value, tags)

    def create_span(self, name: str, parent_id: str | None = None) -> Span:
        span = Span(name=name, parent_id=parent_id)
        return span

    def end_span(self, span: Span) -> None:
        span.end_time = time.monotonic()
        self._spans.append(span)
        self._write_metric(
            "span", span.name, span.duration_ms,
            {"span_id": span.id, "parent_id": span.parent_id or ""},
        )

    def _write_metric(self, metric_type: str, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "type": metric_type,
            "name": name,
            "value": round(value, 3),
            "tags": tags or {},
        }
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Failed to write metric: %s/%s", metric_type, name)
