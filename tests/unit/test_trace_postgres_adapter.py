from __future__ import annotations

from contextlib import contextmanager

from src.core.trace.trace_collector import PostgresTraceSink, TraceCollector
from src.core.trace.trace_context import TraceContext
from src.observability.dashboard.services.trace_service import TraceService


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self._result = []

    def execute(self, query, params=None):
        sql = " ".join(str(query).split()).lower()
        params = params or ()
        if "insert into observability_traces" in sql:
            payload = {
                "trace_id": params[0],
                "trace_type": params[1],
                "started_at": params[2],
                "finished_at": params[3],
                "duration_ms": params[4],
                "metadata_json": params[5],
                "stages_json": params[6],
            }
            self._rows[payload["trace_id"]] = payload
            self._result = []
            return
        if "select trace_id, trace_type, started_at, finished_at, duration_ms, metadata_json, stages_json" in sql:
            ordered = sorted(self._rows.values(), key=lambda item: item["started_at"], reverse=True)
            self._result = [
                (
                    row["trace_id"],
                    row["trace_type"],
                    row["started_at"],
                    row["finished_at"],
                    row["duration_ms"],
                    row["metadata_json"],
                    row["stages_json"],
                )
                for row in ordered
            ]
            return
        self._result = []

    def fetchall(self):
        return self._result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)


class _FakeExecutor:
    _rows = {}

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    @contextmanager
    def connect(self):
        yield _FakeConn(self._rows)

    def execute_ddl(self, statements):
        return None


def test_postgres_trace_sink_and_service(monkeypatch) -> None:
    monkeypatch.setattr("src.core.trace.trace_collector.PostgresExecutor", _FakeExecutor)
    monkeypatch.setattr("src.observability.dashboard.services.trace_service.PostgresExecutor", _FakeExecutor)

    collector = TraceCollector(postgres_dsn="postgresql://example/test", sink_mode="postgres")
    trace = TraceContext(trace_type="agent")
    trace.metadata["message_preview"] = "hello"
    trace.record_stage("prompt_build", {"messages": 3}, elapsed_ms=12.0)
    collector.collect(trace)

    service = TraceService(source="postgres", postgres_dsn="postgresql://example/test")
    traces = service.list_traces(trace_type="agent")

    assert len(traces) == 1
    assert traces[0]["trace_type"] == "agent"
    assert traces[0]["metadata"]["message_preview"] == "hello"
    assert service.get_stage_timings(traces[0])[0]["stage_name"] == "prompt_build"


def test_postgres_trace_sink_direct(monkeypatch) -> None:
    monkeypatch.setattr("src.core.trace.trace_collector.PostgresExecutor", _FakeExecutor)
    monkeypatch.setattr("src.observability.dashboard.services.trace_service.PostgresExecutor", _FakeExecutor)
    sink = PostgresTraceSink("postgresql://example/test")
    trace = TraceContext(trace_type="query")
    trace.record_stage("retrieve", {"top_k": 5}, elapsed_ms=3.0)
    sink.collect(trace.to_dict())
    service = TraceService(source="postgres", postgres_dsn="postgresql://example/test")
    assert service.get_trace(trace.trace_id)["trace_type"] == "query"
