"""Short soak test for upload/worker/chat backend flow."""

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
        except Exception as exc:  # pragma: no cover - script
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"API server failed health check: {last_error}")


def _poll_task(base_url: str, task_id: str, timeout_seconds: float = 180.0) -> dict[str, Any]:
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
            create_ingestion_backends(settings, collection="soak-smoke")
            create_sparse_index(settings, collection="soak-smoke").ensure_collection_ready("soak-smoke")
            VectorStoreFactory.create(settings, collection_name="soak_health").list_collections()
            return
        except Exception as exc:  # pragma: no cover - script
            last_error = str(exc)
            time.sleep(2)
    raise RuntimeError(f"Storage stack not ready: {last_error}")


def _stream_chat(base_url: str, message: str, user_id: str) -> tuple[str, dict[str, Any]]:
    response = requests.post(
        f"{base_url}/api/chat",
        json={"message": message, "user_id": user_id},
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=180,
    )
    response.raise_for_status()
    text_parts: list[str] = []
    done_metadata: dict[str, Any] = {}
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = json.loads(line[6:])
        if payload.get("type") == "text_delta":
            text_parts.append(payload.get("content") or "")
        if payload.get("type") == "done":
            done_metadata = payload.get("metadata") or {}
            break
    return "".join(text_parts), done_metadata


def _write_temp_settings(settings_path: Path, *, port: int, default_collection: str) -> Path:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short backend soak test against the storage stack.")
    parser.add_argument("--settings", default="config/settings.storage_stack.yaml")
    parser.add_argument("--sample", default="tests/fixtures/sample_documents/blogger_intro.pdf")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--user-id-prefix", default="soak_user")
    args = parser.parse_args()

    sample_path = (REPO_ROOT / args.sample).resolve()
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    suffix = uuid.uuid4().hex[:8]
    default_collection = f"soak_default_{suffix}"
    settings_path = _write_temp_settings((REPO_ROOT / args.settings).resolve(), port=args.port, default_collection=default_collection)
    env = os.environ.copy()
    env["MODULAR_RAG_SETTINGS_PATH"] = str(settings_path)
    env["PYTHONUNBUFFERED"] = "1"

    logs: list[tempfile.NamedTemporaryFile] = [
        tempfile.NamedTemporaryFile("w+", suffix=".server.log", delete=False),
        tempfile.NamedTemporaryFile("w+", suffix=".worker.log", delete=False),
    ]
    server = subprocess.Popen([sys.executable, "run_server.py"], cwd=REPO_ROOT, env=env, stdout=logs[0], stderr=subprocess.STDOUT)
    worker = None
    base_url = f"http://127.0.0.1:{args.port}"
    rounds: list[dict[str, Any]] = []
    try:
        _wait_for_storage_stack(settings_path)
        _wait_for_health(base_url)
        worker = subprocess.Popen(
            [sys.executable, "scripts/run_ingestion_worker.py", "--settings", str(settings_path)],
            cwd=REPO_ROOT,
            env=env,
            stdout=logs[1],
            stderr=subprocess.STDOUT,
        )

        for idx in range(args.rounds):
            collection = default_collection
            user_id = f"{args.user_id_prefix}_{idx}"
            started = time.time()
            with sample_path.open("rb") as handle:
                upload = requests.post(
                    f"{base_url}/api/upload",
                    params={"user_id": user_id, "collection": collection},
                    files={"file": (sample_path.name, handle, "application/pdf")},
                    timeout=30,
                )
            upload.raise_for_status()
            upload_payload = upload.json()
            task = _poll_task(base_url, upload_payload["task_id"])
            if task.get("status") not in {"succeeded", "skipped_existing"}:
                raise RuntimeError(f"Round {idx} ingestion failed: {task}")
            answer, done_metadata = _stream_chat(base_url, "博主的笔记价格是多少？", user_id)
            if "199" not in answer:
                raise RuntimeError(f"Round {idx} answer missing fact: {answer}")
            rounds.append(
                {
                    "round": idx,
                    "collection": collection,
                    "task_id": upload_payload["task_id"],
                    "task_status": task.get("status"),
                    "answer": answer,
                    "trace_id": done_metadata.get("trace_id"),
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )

        print(
            json.dumps(
                {
                    "ok": True,
                    "round_count": len(rounds),
                    "rounds": rounds,
                    "logs": [log.name for log in logs],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if worker is not None:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - script
                worker.kill()
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - script
            server.kill()
        try:
            settings_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
