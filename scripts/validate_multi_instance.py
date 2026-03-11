"""Validate two API instances sharing the same production storage stack."""

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
            create_ingestion_backends(settings, collection="multi-instance-smoke")
            create_sparse_index(settings, collection="multi-instance-smoke").ensure_collection_ready("multi-instance-smoke")
            VectorStoreFactory.create(settings, collection_name="multi_instance_health").list_collections()
            return
        except Exception as exc:  # pragma: no cover - script
            last_error = str(exc)
            time.sleep(2)
    raise RuntimeError(f"Storage stack not ready: {last_error}")


def _stream_chat(base_url: str, message: str, user_id: str, conversation_id: str = "") -> list[dict[str, Any]]:
    payload: dict[str, Any] = {"message": message, "user_id": user_id}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    response = requests.post(
        f"{base_url}/api/chat",
        json=payload,
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=180,
    )
    response.raise_for_status()
    events: list[dict[str, Any]] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        events.append(event)
        if event.get("type") == "done":
            break
    return events


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
    parser = argparse.ArgumentParser(description="Validate two API instances sharing the same storage stack.")
    parser.add_argument("--settings", default="config/settings.storage_stack.yaml")
    parser.add_argument("--sample", default="tests/fixtures/sample_documents/blogger_intro.pdf")
    parser.add_argument("--port-a", type=int, default=8011)
    parser.add_argument("--port-b", type=int, default=8012)
    parser.add_argument("--user-id", default="multi_instance_user")
    args = parser.parse_args()

    sample_path = (REPO_ROOT / args.sample).resolve()
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    suffix = uuid.uuid4().hex[:8]
    collection = f"multi_instance_{suffix}"
    settings_a = _write_temp_settings((REPO_ROOT / args.settings).resolve(), port=args.port_a, default_collection=collection)
    settings_b = _write_temp_settings((REPO_ROOT / args.settings).resolve(), port=args.port_b, default_collection=collection)
    env_a = os.environ.copy()
    env_b = os.environ.copy()
    env_a["MODULAR_RAG_SETTINGS_PATH"] = str(settings_a)
    env_b["MODULAR_RAG_SETTINGS_PATH"] = str(settings_b)
    env_a["PYTHONUNBUFFERED"] = "1"
    env_b["PYTHONUNBUFFERED"] = "1"

    logs: list[tempfile.NamedTemporaryFile] = [
        tempfile.NamedTemporaryFile("w+", suffix=".server-a.log", delete=False),
        tempfile.NamedTemporaryFile("w+", suffix=".server-b.log", delete=False),
        tempfile.NamedTemporaryFile("w+", suffix=".worker.log", delete=False),
    ]
    server_a = subprocess.Popen([sys.executable, "run_server.py"], cwd=REPO_ROOT, env=env_a, stdout=logs[0], stderr=subprocess.STDOUT)
    server_b = subprocess.Popen([sys.executable, "run_server.py"], cwd=REPO_ROOT, env=env_b, stdout=logs[1], stderr=subprocess.STDOUT)
    worker = None
    try:
        _wait_for_storage_stack(settings_a)
        base_a = f"http://127.0.0.1:{args.port_a}"
        base_b = f"http://127.0.0.1:{args.port_b}"
        _wait_for_health(base_a)
        _wait_for_health(base_b)

        worker = subprocess.Popen(
            [sys.executable, "scripts/run_ingestion_worker.py", "--settings", str(settings_a)],
            cwd=REPO_ROOT,
            env=env_a,
            stdout=logs[2],
            stderr=subprocess.STDOUT,
        )

        with sample_path.open("rb") as handle:
            upload = requests.post(
                f"{base_a}/api/upload",
                params={"user_id": args.user_id, "collection": collection},
                files={"file": (sample_path.name, handle, "application/pdf")},
                timeout=30,
            )
        upload.raise_for_status()
        upload_payload = upload.json()
        task = _poll_task(base_a, upload_payload["task_id"])
        if task.get("status") != "succeeded":
            raise RuntimeError(f"Upload did not succeed: {task}")

        first_chat = _stream_chat(base_a, "博主的笔记价格是多少？", args.user_id)
        first_done = next(event for event in first_chat if event.get("type") == "done")
        conversation_id = str((first_done.get("metadata") or {}).get("conversation_id") or "")
        first_text = "".join(event.get("content") or "" for event in first_chat if event.get("type") == "text_delta")
        if "199" not in first_text:
            raise RuntimeError(f"Server A answer missing fact: {first_text}")

        second_chat = _stream_chat(base_b, "这个笔记的价格再确认一下。", args.user_id, conversation_id=conversation_id)
        second_done = next(event for event in second_chat if event.get("type") == "done")
        second_text = "".join(event.get("content") or "" for event in second_chat if event.get("type") == "text_delta")
        if "199" not in second_text:
            raise RuntimeError(f"Server B answer missing fact: {second_text}")

        print(
            json.dumps(
                {
                    "ok": True,
                    "collection": collection,
                    "task": task,
                    "conversation_id": conversation_id,
                    "first_answer": first_text,
                    "second_answer": second_text,
                    "trace_ids": [
                        (first_done.get("metadata") or {}).get("trace_id"),
                        (second_done.get("metadata") or {}).get("trace_id"),
                    ],
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
        for proc in (server_a, server_b):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - script
                proc.kill()
        for temp_file in (settings_a, settings_b):
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
