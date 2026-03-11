"""Backend-only API smoke test for upload -> worker -> chat."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.settings import load_settings
from src.libs.vector_store import VectorStoreFactory
from src.storage.postgres import PostgresExecutor
from src.storage.runtime import create_ingestion_backends, create_sparse_index


def _wait_for_health(base_url: str, timeout_seconds: float = 240.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/api/health", timeout=2)
            if response.ok:
                return
            last_error = response.text
        except Exception as exc:  # pragma: no cover - smoke script
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"API server failed health check: {last_error}")


def _poll_task(base_url: str, task_id: str, timeout_seconds: float = 120.0) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_payload: dict[str, Any] = {}
    while time.time() < deadline:
        response = requests.get(f"{base_url}/api/ingestion/tasks/{task_id}", timeout=30)
        response.raise_for_status()
        payload = response.json()
        last_payload = payload
        if payload.get("status") in {"succeeded", "failed", "skipped_existing"}:
            return payload
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for task {task_id}: {last_payload}")


def _wait_for_storage_stack(settings_path: Path, timeout_seconds: float = 180.0) -> None:
    settings = load_settings(str(settings_path))
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            executor = PostgresExecutor(settings.postgres.dsn)
            with executor.connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            import redis

            redis.Redis.from_url(settings.redis.url, decode_responses=True).ping()
            create_ingestion_backends(settings, collection="api-smoke")
            create_sparse_index(settings, collection="api-smoke").ensure_collection_ready("api-smoke")
            VectorStoreFactory.create(settings, collection_name="api_smoke_health").list_collections()
            return
        except Exception as exc:  # pragma: no cover - smoke script
            last_error = str(exc)
            time.sleep(2)
    raise RuntimeError(f"Storage stack not ready: {last_error}")


def _stream_chat(base_url: str, message: str, user_id: str) -> list[dict[str, Any]]:
    response = requests.post(
        f"{base_url}/api/chat",
        json={"message": message, "user_id": user_id},
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=120,
    )
    response.raise_for_status()
    chunks: list[dict[str, Any]] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        chunks.append(payload)
        if payload.get("type") == "done":
            break
    return chunks


def _write_temp_settings(
    settings_path: Path,
    *,
    port: int,
    default_collection: str,
) -> Path:
    with settings_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}

    settings.setdefault("server", {})
    settings["server"]["port"] = port
    settings["server"]["host"] = "127.0.0.1"

    settings.setdefault("agent", {})
    settings["agent"]["default_collection"] = default_collection
    settings["agent"]["auto_ingest_dir"] = ""

    settings.setdefault("vector_store", {})
    settings["vector_store"]["collection_name"] = default_collection

    temp_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    with temp_file as handle:
        yaml.safe_dump(settings, handle, allow_unicode=True, sort_keys=False)
    return Path(temp_file.name)


def _get_postgres_dsn(settings_path: Path) -> str:
    with settings_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    return str(((settings.get("postgres") or {}).get("dsn")) or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backend-only API smoke against the storage stack.")
    parser.add_argument("--settings", default="config/settings.storage_stack.yaml")
    parser.add_argument("--sample", default="tests/fixtures/sample_documents/blogger_intro.pdf")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--user-id", default="api_e2e_user")
    args = parser.parse_args()

    sample_path = (REPO_ROOT / args.sample).resolve()
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    suffix = uuid.uuid4().hex[:8]
    collection_a = f"api_e2e_a_{suffix}"
    collection_b = f"api_e2e_b_{suffix}"
    temp_settings = _write_temp_settings(
        (REPO_ROOT / args.settings).resolve(),
        port=args.port,
        default_collection=collection_b,
    )
    env = os.environ.copy()
    env["MODULAR_RAG_SETTINGS_PATH"] = str(temp_settings)
    env["PYTHONUNBUFFERED"] = "1"
    postgres_dsn = _get_postgres_dsn(temp_settings)

    server_log = tempfile.NamedTemporaryFile("w+", suffix=".server.log", delete=False)
    worker_log = tempfile.NamedTemporaryFile("w+", suffix=".worker.log", delete=False)
    server_proc = subprocess.Popen(
        [sys.executable, "run_server.py"],
        cwd=REPO_ROOT,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    worker_proc = None
    base_url = f"http://127.0.0.1:{args.port}"

    try:
        _wait_for_storage_stack(temp_settings)
        _wait_for_health(base_url)
        worker_proc = subprocess.Popen(
            [sys.executable, "scripts/run_ingestion_worker.py", "--settings", str(temp_settings)],
            cwd=REPO_ROOT,
            env=env,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
        )

        def upload(collection: str) -> dict[str, Any]:
            with sample_path.open("rb") as handle:
                response = requests.post(
                    f"{base_url}/api/upload",
                    params={"user_id": args.user_id, "collection": collection},
                    files={"file": (sample_path.name, handle, "application/pdf")},
                    timeout=30,
                )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("task_id"):
                raise RuntimeError(f"Upload did not return task_id: {payload}")
            return payload

        upload_a = upload(collection_a)
        task_a = _poll_task(base_url, upload_a["task_id"])
        if task_a.get("status") != "succeeded" or (task_a.get("result") or {}).get("chunk_count", 0) <= 0:
            raise RuntimeError(f"Collection A ingestion failed: {task_a}")

        upload_b = upload(collection_b)
        task_b = _poll_task(base_url, upload_b["task_id"])
        if task_b.get("status") != "succeeded" or (task_b.get("result") or {}).get("chunk_count", 0) <= 0:
            raise RuntimeError(f"Collection B ingestion failed: {task_b}")

        chunks = _stream_chat(base_url, "博主的笔记价格是多少？", args.user_id)
        tool_starts = [chunk for chunk in chunks if chunk.get("type") == "tool_start"]
        if not any(chunk.get("tool_name") == "knowledge_query" for chunk in tool_starts):
            raise RuntimeError(f"Expected knowledge_query tool_start, got: {tool_starts}")

        final_text = "".join(chunk.get("content") or "" for chunk in chunks if chunk.get("type") == "text_delta")
        done_event = next((chunk for chunk in chunks if chunk.get("type") == "done"), None)
        if done_event is None:
            raise RuntimeError(f"Missing done event: {chunks}")
        done_metadata = done_event.get("metadata") or {}
        if "199" not in final_text:
            raise RuntimeError(f"Answer missing expected fact: {final_text}")
        if not done_metadata.get("trace_id"):
            raise RuntimeError(f"Missing trace_id in done metadata: {done_metadata}")
        if not done_metadata.get("query_trace_ids"):
            raise RuntimeError(f"Missing query_trace_ids in done metadata: {done_metadata}")
        if done_metadata.get("grounding_policy_action") not in {"normal", "conservative_rewrite"}:
            raise RuntimeError(f"Expected grounding_policy_action=normal, got: {done_metadata}")
        if done_metadata.get("low_evidence"):
            raise RuntimeError(f"Expected grounded answer without low_evidence warning, got: {done_metadata}")
        citations = done_metadata.get("citations") or []
        if not citations:
            raise RuntimeError(f"Expected citations in done metadata: {done_metadata}")
        sources = [str(citation.get("source", "")) for citation in citations]
        if any("/tmp/" in source or "/private/var/folders/" in source for source in sources):
            raise RuntimeError(f"Sources leaked worker temp paths: {sources}")

        from src.observability.dashboard.services.trace_service import TraceService

        trace_service = TraceService(source="postgres", postgres_dsn=postgres_dsn) if postgres_dsn else TraceService()
        query_trace_ids = [str(value) for value in done_metadata.get("query_trace_ids", []) if str(value)]
        if not query_trace_ids:
            raise RuntimeError(f"Expected non-empty query trace ids: {done_metadata}")
        query_trace = trace_service.get_trace(query_trace_ids[0])
        if query_trace is None:
            raise RuntimeError(f"Unable to load query trace {query_trace_ids[0]}")
        stage_names = [str(stage.get("stage", "")) for stage in query_trace.get("stages", [])]
        if "rerank" not in stage_names:
            raise RuntimeError(f"Expected rerank stage in query trace, got: {stage_names}")

        print(
            json.dumps(
                {
                    "ok": True,
                    "collections": [collection_a, collection_b],
                    "task_a": task_a,
                    "task_b": task_b,
                    "final_text": final_text,
                    "done_metadata": done_metadata,
                    "query_trace_stage_names": stage_names,
                    "server_log": server_log.name,
                    "worker_log": worker_log.name,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception:
        print(
            json.dumps(
                {
                    "ok": False,
                    "server_log": server_log.name,
                    "worker_log": worker_log.name,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise
    finally:
        if worker_proc is not None:
            worker_proc.terminate()
            try:
                worker_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - smoke script
                worker_proc.kill()
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - smoke script
            server_proc.kill()
        try:
            temp_settings.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
