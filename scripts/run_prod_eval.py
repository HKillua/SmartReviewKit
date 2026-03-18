#!/usr/bin/env python3
"""Production-stack overnight evaluation runner.

This script brings up the storage stack, ingests course materials into an
isolated evaluation collection, runs production-path RAG + Agent evaluation,
and writes structured artifacts plus a Markdown report.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import textwrap
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.tools.knowledge_query import KnowledgeQueryArgs
from src.agent.types import LlmMessage, LlmRequest, ToolContext
from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline
from src.observability.dashboard.services.trace_service import TraceService
from src.observability.evaluation.agent_eval_runner import AgentEvalRunner
from src.observability.evaluation.eval_runner import load_test_set
from src.observability.evaluation.ragas_evaluator import RagasEvaluator
from src.server.app import create_app
from src.storage.sparse_index import OpenSearchSparseIndex, create_sparse_index


DATE_STAMP = "2026-03-18-prod"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs" / "eval" / DATE_STAMP
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "PROD_EVAL_2026-03-18.md"
DEFAULT_COLLECTION = "computer_network_eval_20260318"
DEFAULT_SETTINGS = REPO_ROOT / "config" / "settings.storage_stack.yaml"
DEFAULT_COMPOSE = REPO_ROOT / "docker-compose.storage.yml"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs" / "computer_internet"
DEFAULT_GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"
DEFAULT_AGENT_GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "agent_golden_test_set.json"
DEFAULT_PRODUCT_SOURCE_MAP = REPO_ROOT / "config" / "eval" / "computer_network_product_sources.yaml"

ALLOWED_DOC_EXTS = {".pdf", ".pptx", ".docx"}
ANSWER_MAX_CHARS = 1500
RETRIEVAL_TOP_K = 10
EVAL_PROFILES = ("balanced_fast", "quality_first")


def _configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


logger = logging.getLogger("run_prod_eval")


def _run_command(cmd: list[str], *, cwd: Path = REPO_ROOT, capture: bool = False) -> subprocess.CompletedProcess[str]:
    logger.info("Running command: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )


def _wait_until(name: str, check_fn: Any, *, timeout_s: float = 300.0, interval_s: float = 3.0) -> dict[str, Any]:
    started = time.monotonic()
    last_error = ""
    while time.monotonic() - started < timeout_s:
        try:
            extra = check_fn()
            return {
                "name": name,
                "ready": True,
                "elapsed_s": round(time.monotonic() - started, 2),
                "detail": extra or "",
            }
        except Exception as exc:  # pragma: no cover - runtime probing
            last_error = str(exc)
            time.sleep(interval_s)
    raise RuntimeError(f"{name} did not become ready within {timeout_s:.0f}s: {last_error}")


def _check_postgres(dsn: str) -> str:
    import psycopg

    with psycopg.connect(dsn, connect_timeout=3) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    return "postgres ok"


def _check_redis(redis_url: str) -> str:
    parsed = urlparse(redis_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6379
    with socket.create_connection((host, port), timeout=3.0) as sock:
        sock.sendall(b"*1\r\n$4\r\nPING\r\n")
        payload = sock.recv(16)
    if b"PONG" not in payload:
        raise RuntimeError(f"unexpected redis ping response: {payload!r}")
    return "redis ok"


def _check_http(url: str, *, expect_substring: str = "") -> str:
    with urlopen(url, timeout=5.0) as resp:  # noqa: S310 - trusted local endpoints
        body = resp.read(4096).decode("utf-8", errors="ignore")
    if expect_substring and expect_substring not in body:
        raise RuntimeError(f"{url} missing expected payload fragment '{expect_substring}'")
    return f"http {url} ok"


def _check_milvus(_uri: str) -> str:
    return _check_http("http://127.0.0.1:9091/healthz", expect_substring="OK")


def _check_milvus_grpc(uri: str) -> str:
    parsed = urlparse(uri)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 19530
    with socket.create_connection((host, port), timeout=3.0):
        pass
    return f"grpc {host}:{port} ok"


def _check_etcd_running(compose_file: Path) -> str:
    proc = _run_command(
        ["docker", "compose", "-f", str(compose_file), "ps", "etcd"],
        capture=True,
    )
    output = proc.stdout.lower()
    if "running" not in output and " up " not in output:
        raise RuntimeError(f"etcd not reported as running: {output.strip()}")
    return "etcd container running"


def _normalize_source(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).name.casefold()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    expected_set = set(expected)
    for rank, value in enumerate(retrieved, start=1):
        if value in expected_set:
            return 1.0 / rank
    return 0.0


def _hit_rate(retrieved: list[str], expected: list[str]) -> float:
    if not expected:
        return 0.0
    return 1.0 if set(retrieved) & set(expected) else 0.0


def _ndcg(retrieved: list[str], expected: list[str], k: int) -> float:
    import math

    expected_set = set(expected)
    dcg = 0.0
    seen_hits: set[str] = set()
    for index, value in enumerate(retrieved[:k]):
        if value in expected_set and value not in seen_hits:
            dcg += 1.0 / math.log2(index + 2)
            seen_hits.add(value)
    ideal_hits = min(len(expected_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(index + 2) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else 0.0


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_product_source_map(path: Path) -> dict[str, list[str]]:
    payload = _load_yaml(path)
    groups = payload.get("source_groups", {})
    normalized: dict[str, list[str]] = {}
    if isinstance(groups, dict):
        for source, aliases in groups.items():
            key = _normalize_source(source)
            if not key:
                continue
            values = [_normalize_source(source)]
            if isinstance(aliases, list):
                values.extend(_normalize_source(alias) for alias in aliases)
            normalized[key] = _dedupe_preserve_order([value for value in values if value])
    return normalized


def _expand_product_sources(expected_sources: list[str], source_map: dict[str, list[str]]) -> list[str]:
    expanded: list[str] = []
    for source in expected_sources:
        normalized = _normalize_source(source)
        if not normalized:
            continue
        expanded.extend(source_map.get(normalized, [normalized]))
    return _dedupe_preserve_order(expanded)


def _build_runtime_config(
    base_path: Path,
    runtime_path: Path,
    collection: str,
    *,
    response_profile: str,
) -> Path:
    data = _load_yaml(base_path)
    data.setdefault("agent", {})
    data["agent"]["default_collection"] = collection
    data["agent"]["auto_ingest_dir"] = ""
    data["agent"]["response_profile"] = response_profile
    data.setdefault("vision_llm", {})
    data["vision_llm"]["enabled"] = False
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    return runtime_path


def _build_runtime_agent_golden(base_path: Path, runtime_path: Path, import_file: Path) -> Path:
    payload = json.loads(base_path.read_text(encoding="utf-8"))
    for case in payload.get("test_cases", []):
        if str(case.get("id", "")) == "document_ingest_upload":
            case["message"] = f"请把 {import_file} 导入知识库。"
    _write_json(runtime_path, payload)
    return runtime_path


def _discover_docs(docs_dir: Path) -> list[Path]:
    return sorted(
        [path for path in docs_dir.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_DOC_EXTS],
        key=lambda path: path.name,
    )


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _start_storage_stack(compose_file: Path) -> None:
    _run_command(["docker", "compose", "-f", str(compose_file), "up", "-d"])


def _restart_milvus_service(compose_file: Path) -> None:
    _run_command(["docker", "compose", "-f", str(compose_file), "restart", "milvus"])


def _wait_for_stack(settings_path: Path, compose_file: Path) -> list[dict[str, Any]]:
    settings = load_settings(settings_path)
    checks = [
        ("postgres", lambda: _check_postgres(settings.postgres.dsn)),
        ("redis", lambda: _check_redis(settings.redis.url)),
        ("minio", lambda: _check_http("http://127.0.0.1:9000/minio/health/live")),
        ("milvus", lambda: _check_milvus(settings.milvus.uri)),
        ("opensearch", lambda: _check_http("http://127.0.0.1:9200/_cluster/health", expect_substring="status")),
        ("etcd", lambda: _check_etcd_running(compose_file)),
    ]
    results: list[dict[str, Any]] = []
    for name, fn in checks:
        logger.info("Waiting for %s...", name)
        results.append(_wait_until(name, fn))
    return results


def _recover_milvus_load_state(settings_path: Path, compose_file: Path, collection: str) -> dict[str, Any]:
    from pymilvus import MilvusClient

    settings = load_settings(settings_path)
    uri = getattr(settings.milvus, "uri", "") or f"http://{settings.milvus.host}:{settings.milvus.port}"
    client = MilvusClient(uri=uri)
    collections = client.list_collections()
    loading_before = {}
    for name in collections:
        try:
            state = client.get_load_state(name)
        except Exception:
            continue
        if not str(state.get("state", "")).endswith("NotLoad"):
            loading_before[name] = state

    released = 0
    for name in collections:
        if name == collection:
            continue
        try:
            client.release_collection(name, timeout=5)
            released += 1
        except Exception:
            logger.warning("Failed to release non-target Milvus collection %s", name, exc_info=True)

    try:
        client.release_collection(collection, timeout=5)
    except Exception:
        logger.warning("Failed to release target Milvus collection %s before app init", collection, exc_info=True)

    try:
        target_state = client.get_load_state(collection)
    except Exception:
        target_state = {"state": "unknown"}

    restarted = False
    if not str(target_state.get("state", "")).endswith("NotLoad"):
        logger.warning(
            "Target Milvus collection %s remained in state %s after release; restarting milvus service",
            collection,
            target_state,
        )
        _restart_milvus_service(compose_file)
        _wait_until("milvus-http-restart", lambda: _check_milvus(uri), timeout_s=180.0, interval_s=2.0)
        _wait_until("milvus-grpc-restart", lambda: _check_milvus_grpc(uri), timeout_s=180.0, interval_s=2.0)
        client = MilvusClient(uri=uri)
        try:
            client.release_collection(collection, timeout=5)
        except Exception:
            logger.warning("Failed to release target Milvus collection %s after restart", collection, exc_info=True)
        target_state = client.get_load_state(collection)
        restarted = True

    diagnostics = {
        "collection": collection,
        "released_non_target_count": released,
        "loading_before_count": len(loading_before),
        "loading_before": {name: str(state.get("state", "")) for name, state in loading_before.items()},
        "restarted_milvus": restarted,
        "target_state_after_recovery": str(target_state.get("state", "")),
    }
    logger.info("Milvus recovery diagnostics: %s", diagnostics)
    return diagnostics


def _reset_eval_backends(settings_path: Path, collection: str) -> dict[str, Any]:
    from pymilvus import MilvusClient

    settings = load_settings(settings_path)
    uri = getattr(settings.milvus, "uri", "") or f"http://{settings.milvus.host}:{settings.milvus.port}"
    client = MilvusClient(uri=uri)

    dropped_collection = False
    if client.has_collection(collection):
        client.drop_collection(collection)
        dropped_collection = True

    sparse_index = create_sparse_index(settings, collection=collection)
    deleted_sparse_index = False
    sparse_index_name = ""
    if isinstance(sparse_index, OpenSearchSparseIndex):
        sparse_index_name = sparse_index._index_name(collection)
        if sparse_index._client.indices.exists(index=sparse_index_name):
            sparse_index._client.indices.delete(index=sparse_index_name)
            deleted_sparse_index = True

    diagnostics = {
        "collection": collection,
        "dropped_milvus_collection": dropped_collection,
        "deleted_sparse_index": deleted_sparse_index,
        "sparse_index_name": sparse_index_name,
    }
    logger.info("Reset eval backends diagnostics: %s", diagnostics)
    return diagnostics


def _ingest_documents(settings_path: Path, docs_dir: Path, collection: str, output_dir: Path) -> dict[str, Any]:
    settings = load_settings(settings_path)
    files = _discover_docs(docs_dir)
    pipeline = IngestionPipeline(settings, collection=collection, force=True)
    entries: list[dict[str, Any]] = []
    try:
        for path in files:
            started = time.monotonic()
            source_type = IngestionPipeline.infer_source_type(path)
            logger.info("Ingesting %s (source_type=%s)", path.name, source_type)
            try:
                result = pipeline.run(
                    str(path),
                    source_type=source_type,
                    metadata_overrides={
                        "source_path": str(path),
                        "source_label": path.name,
                        "original_filename": path.name,
                    },
                    record_file_path=str(path),
                )
                entry = {
                    "filename": path.name,
                    "path": str(path),
                    "ext": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                    "source_type": source_type,
                    "success": bool(result.success),
                    "doc_id": result.doc_id or "",
                    "chunk_count": int(result.chunk_count),
                    "image_count": int(result.image_count),
                    "failed_chunk_count": int(result.failed_chunk_count),
                    "integrity_skipped": bool(result.stages.get("integrity", {}).get("skipped", False)),
                    "error": result.error or "",
                    "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
                }
            except Exception as exc:  # pragma: no cover - runtime pipeline failures
                entry = {
                    "filename": path.name,
                    "path": str(path),
                    "ext": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                    "source_type": source_type,
                    "success": False,
                    "doc_id": "",
                    "chunk_count": 0,
                    "image_count": 0,
                    "failed_chunk_count": 0,
                    "integrity_skipped": False,
                    "error": str(exc),
                    "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
                }
            entries.append(entry)
    finally:
        pipeline.close()

    summary = {
        "collection": collection,
        "docs_dir": str(docs_dir),
        "file_count": len(entries),
        "success_count": sum(1 for entry in entries if entry["success"]),
        "failure_count": sum(1 for entry in entries if not entry["success"]),
        "integrity_skipped_count": sum(1 for entry in entries if entry["integrity_skipped"]),
        "total_chunks": sum(int(entry["chunk_count"]) for entry in entries),
        "total_images": sum(int(entry["image_count"]) for entry in entries),
        "entries": entries,
    }

    _write_json(output_dir / "ingestion_summary.json", summary)
    rows = [
        [
            entry["filename"],
            entry["source_type"],
            "yes" if entry["success"] else "no",
            "yes" if entry["integrity_skipped"] else "no",
            str(entry["chunk_count"]),
            str(entry["image_count"]),
            f"{entry['elapsed_ms']:.1f}",
            entry["error"] or "—",
        ]
        for entry in entries
    ]
    md = "# Ingestion Summary\n\n" + _format_table(
        ["File", "Source Type", "Success", "Skipped", "Chunks", "Images", "Elapsed(ms)", "Error"],
        rows,
    )
    _write_text(output_dir / "ingestion_summary.md", md)
    return summary


def _maybe_reuse_ingestion_summary(
    output_dir: Path,
    docs_dir: Path,
    collection: str,
) -> dict[str, Any] | None:
    summary_path = output_dir / "ingestion_summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if str(payload.get("collection", "")) != collection:
        return None
    if Path(str(payload.get("docs_dir", ""))).resolve() != docs_dir.resolve():
        return None
    entries = payload.get("entries", [])
    if not isinstance(entries, list) or not entries:
        return None
    expected_files = {path.name for path in _discover_docs(docs_dir)}
    actual_files = {str(entry.get("filename", "")) for entry in entries}
    if expected_files != actual_files:
        return None
    if int(payload.get("success_count", 0)) != len(expected_files):
        return None
    if int(payload.get("failure_count", 0)) != 0:
        return None
    logger.info(
        "Reusing existing ingestion summary for collection %s from %s",
        collection,
        summary_path,
    )
    return payload


def _create_runtime_app(settings_path: Path) -> Any:
    os.environ["MODULAR_RAG_SETTINGS_PATH"] = str(settings_path)
    return create_app(str(settings_path))


def _invalidate_eval_semantic_cache(app: Any, collection: str) -> dict[str, Any]:
    agent = getattr(getattr(app, "state", None), "agent", None)
    tools = getattr(agent, "tools", None)
    tool = tools.get_tool("knowledge_query") if tools is not None else None
    semantic_cache = getattr(tool, "_semantic_cache", None)
    invalidate = getattr(semantic_cache, "invalidate_by_collection", None)
    if not callable(invalidate):
        return {
            "collection": collection,
            "cache_present": bool(semantic_cache),
            "invalidated": 0,
        }
    try:
        invalidated = int(invalidate(collection) or 0)
    except Exception:
        logger.warning("Failed to invalidate semantic cache for collection %s", collection, exc_info=True)
        invalidated = 0
    return {
        "collection": collection,
        "cache_present": True,
        "invalidated": invalidated,
    }


def _close_runtime_app(app: Any) -> None:
    if app is None:
        return
    agent = getattr(getattr(app, "state", None), "agent", None)
    llm = getattr(agent, "llm", None)
    close = getattr(llm, "close", None)
    if callable(close):
        logger.info("Closing runtime LLM client")
        try:
            asyncio.run(close())
        except RuntimeError as exc:
            if "Event loop is closed" not in str(exc):
                raise
            logger.debug("Runtime LLM client close skipped after loop shutdown")


def _build_tool_context(query: str, collection: str, index: int) -> ToolContext:
    return ToolContext(
        user_id="rag_eval_user",
        conversation_id=f"rag_eval_conv_{index:03d}",
        metadata={
            "default_collection": collection,
            "planner_task_intent": "knowledge_query",
            "source": "eval",
        },
        recent_messages=[{"role": "user", "content": query}],
    )


def _build_evidence_block(citations: list[dict[str, Any]], evidence_texts: list[str], *, max_items: int = 3) -> str:
    lines: list[str] = []
    for index, text in enumerate(evidence_texts[:max_items], start=1):
        if not text:
            continue
        citation = citations[index - 1] if index - 1 < len(citations) else {}
        source = str(citation.get("source", "") or "未知来源").strip()
        lines.append(f"[{index}] {source}")
        lines.append(str(text).strip()[:420])
        lines.append("")
    return "\n".join(lines).strip()


async def _generate_rag_answer(
    app: Any,
    *,
    query: str,
    citations: list[dict[str, Any]],
    evidence_texts: list[str],
    response_profile: str,
) -> dict[str, Any]:
    agent = app.state.agent
    evidence_block = _build_evidence_block(citations, evidence_texts)
    if not evidence_block:
        return {"error": "no evidence block", "answer": "", "elapsed_ms": 0.0}
    max_tokens = 420 if response_profile == "balanced_fast" else 720
    request = LlmRequest(
        messages=[
            LlmMessage(
                role="system",
                content=(
                    "你是一名计算机网络课程助教。"
                    "请基于给定证据直接回答问题，回答要准确、简洁，并在关键结论后保留 `[1]`、`[2]` 这类引用编号。"
                ),
            ),
            LlmMessage(
                role="user",
                content=(
                    f"问题：{query}\n\n"
                    f"课程证据：\n{evidence_block}\n\n"
                    "请直接给出最终回答。若证据不足，请明确指出缺口。"
                ),
            ),
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        stream=False,
        metadata={
            "course_task": True,
            "skip_reflection_warning": True,
            "generation_mode": f"rag_eval_{response_profile}",
        },
    )
    started = time.monotonic()
    for middleware in agent.middlewares:
        try:
            request = await middleware.before_llm_request(request)
        except Exception:
            logger.exception("RAG eval before_llm_request middleware failed")
    response = await agent.llm.send_request(request)
    for middleware in reversed(agent.middlewares):
        try:
            response = await middleware.after_llm_response(request, response)
        except Exception:
            logger.exception("RAG eval after_llm_response middleware failed")
    elapsed_ms = round((time.monotonic() - started) * 1000.0, 1)
    if response.error:
        return {"error": response.error, "answer": "", "elapsed_ms": elapsed_ms}
    answer = (response.content or "").strip()
    if len(answer) > ANSWER_MAX_CHARS:
        answer = answer[:ANSWER_MAX_CHARS]
    return {"answer": answer, "elapsed_ms": elapsed_ms}


def _run_rag_eval(
    app: Any,
    settings_path: Path,
    golden_path: Path,
    collection: str,
    output_dir: Path,
    *,
    product_source_map: dict[str, list[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    tool = app.state.agent.tools.get_tool("knowledge_query")
    test_cases = load_test_set(golden_path)

    retrieval_cases: list[dict[str, Any]] = []
    ragas_cases_by_profile: dict[str, list[dict[str, Any]]] = {
        profile: [] for profile in EVAL_PROFILES
    }

    ragas_error = ""
    ragas_evaluator: RagasEvaluator | None = None
    try:
        ragas_evaluator = RagasEvaluator(
            settings=load_settings(settings_path),
            metrics=["faithfulness", "answer_relevancy", "context_precision"],
        )
    except Exception as exc:  # pragma: no cover - dependency/runtime guarded in report
        ragas_error = str(exc)
        logger.warning("RAGAS evaluator unavailable: %s", exc)

    for index, test_case in enumerate(test_cases, start=1):
        started = time.monotonic()
        tool_result = asyncio.run(
            tool.execute(
                _build_tool_context(test_case.query, collection, index),
                KnowledgeQueryArgs(query=test_case.query, top_k=RETRIEVAL_TOP_K, collection=collection),
            )
        )
        elapsed_ms = round((time.monotonic() - started) * 1000.0, 1)

        metadata = dict(tool_result.metadata or {})
        citations = metadata.get("citations", []) if isinstance(metadata.get("citations", []), list) else []
        evidence_texts = metadata.get("evidence_texts", []) if isinstance(metadata.get("evidence_texts", []), list) else []
        retrieved_sources = _dedupe_preserve_order(
            [_normalize_source(citation.get("source", "")) for citation in citations]
        )
        expected_sources = [_normalize_source(value) for value in test_case.expected_sources]
        product_sources = _expand_product_sources(expected_sources, product_source_map)

        retrieval_case = {
            "query": test_case.query,
            "expected_sources": expected_sources,
            "product_sources": product_sources,
            "retrieved_sources": retrieved_sources,
            "success": bool(tool_result.success),
            "error": tool_result.error or "",
            "elapsed_ms": elapsed_ms,
            "citation_count": len(citations),
            "fast_mode": bool(metadata.get("fast_mode", False)),
            "cache_hit": bool(metadata.get("cache_hit", False)),
            "query_trace_ids": [str(value) for value in metadata.get("query_trace_ids", []) if str(value)],
            "strict_hit_rate": _hit_rate(retrieved_sources, expected_sources),
            "strict_mrr": _reciprocal_rank(retrieved_sources, expected_sources),
            f"strict_ndcg@{RETRIEVAL_TOP_K}": _ndcg(retrieved_sources, expected_sources, RETRIEVAL_TOP_K),
            "product_hit_rate": _hit_rate(retrieved_sources, product_sources),
            "product_mrr": _reciprocal_rank(retrieved_sources, product_sources),
            f"product_ndcg@{RETRIEVAL_TOP_K}": _ndcg(retrieved_sources, product_sources, RETRIEVAL_TOP_K),
        }
        retrieval_cases.append(retrieval_case)

        retrieved_chunks = []
        chunk_count = max(len(citations), len(evidence_texts))
        for chunk_index in range(chunk_count):
            citation = citations[chunk_index] if chunk_index < len(citations) else {}
            text = evidence_texts[chunk_index] if chunk_index < len(evidence_texts) else ""
            if not text:
                continue
            retrieved_chunks.append(
                {
                    "chunk_id": citation.get("chunk_id", f"eval_chunk_{chunk_index + 1}"),
                    "text": str(text),
                    "source": citation.get("source", ""),
                }
            )
        for profile in EVAL_PROFILES:
            ragas_case = {
                "query": test_case.query,
                "retrieved_sources": retrieved_sources,
                "answer_snippet": "",
                "generation_elapsed_ms": 0.0,
                "judge_elapsed_ms": 0.0,
            }
            if ragas_evaluator is None:
                ragas_case["error"] = ragas_error or "RAGAS evaluator unavailable"
            elif not retrieved_chunks:
                ragas_case["error"] = "no retrieved chunks"
            else:
                generated = asyncio.run(
                    _generate_rag_answer(
                        app,
                        query=test_case.query,
                        citations=citations,
                        evidence_texts=evidence_texts,
                        response_profile=profile,
                    )
                )
                ragas_case["generation_elapsed_ms"] = float(generated.get("elapsed_ms", 0.0) or 0.0)
                answer = str(generated.get("answer", "") or "").strip()
                ragas_case["answer_snippet"] = answer[:240]
                if generated.get("error"):
                    ragas_case["error"] = str(generated["error"])
                elif not answer:
                    ragas_case["error"] = "no generated answer"
                else:
                    judge_started = time.monotonic()
                    try:
                        ragas_case["metrics"] = ragas_evaluator.evaluate(
                            query=test_case.query,
                            retrieved_chunks=retrieved_chunks,
                            generated_answer=answer,
                        )
                    except Exception as exc:  # pragma: no cover - live API/judge path
                        ragas_case["error"] = str(exc)
                    ragas_case["judge_elapsed_ms"] = round((time.monotonic() - judge_started) * 1000.0, 1)
            ragas_cases_by_profile[profile].append(ragas_case)

    retrieval_report = {
        "collection": collection,
        "settings_path": str(settings_path),
        "golden_test_set": str(golden_path),
        "top_k": RETRIEVAL_TOP_K,
        "strict_source_metrics": {
            "hit_rate": round(_mean_metric(retrieval_cases, "strict_hit_rate"), 4),
            "mrr": round(_mean_metric(retrieval_cases, "strict_mrr"), 4),
            f"ndcg@{RETRIEVAL_TOP_K}": round(_mean_metric(retrieval_cases, f"strict_ndcg@{RETRIEVAL_TOP_K}"), 4),
            "avg_latency_ms": round(_mean_metric(retrieval_cases, "elapsed_ms"), 1),
        },
        "product_concept_metrics": {
            "hit_rate": round(_mean_metric(retrieval_cases, "product_hit_rate"), 4),
            "mrr": round(_mean_metric(retrieval_cases, "product_mrr"), 4),
            f"ndcg@{RETRIEVAL_TOP_K}": round(_mean_metric(retrieval_cases, f"product_ndcg@{RETRIEVAL_TOP_K}"), 4),
            "avg_latency_ms": round(_mean_metric(retrieval_cases, "elapsed_ms"), 1),
        },
        "query_results": retrieval_cases,
    }
    ragas_report = {
        "collection": collection,
        "settings_path": str(settings_path),
        "golden_test_set": str(golden_path),
        "profiles": {},
        "blocked_reason": ragas_error,
    }
    for profile, ragas_cases in ragas_cases_by_profile.items():
        ragas_metric_keys = sorted(
            {
                metric_name
                for case in ragas_cases
                for metric_name in case.get("metrics", {}).keys()
            }
        )
        ragas_report["profiles"][profile] = {
            "aggregate_metrics": {
                key: round(
                    sum(float(case["metrics"][key]) for case in ragas_cases if key in case.get("metrics", {}))
                    / max(1, sum(1 for case in ragas_cases if key in case.get("metrics", {}))),
                    4,
                )
                for key in ragas_metric_keys
            },
            "query_results": ragas_cases,
        }

    _write_json(output_dir / "rag_retrieval_eval.json", retrieval_report)
    retrieval_lines = ["RAG Retrieval Evaluation", ""]
    for case in retrieval_cases:
        retrieval_lines.append(
            f"- strict={'HIT' if case['strict_hit_rate'] > 0 else 'MISS'} "
            f"product={'HIT' if case['product_hit_rate'] > 0 else 'MISS'} | {case['query']} | "
            f"sources={case['retrieved_sources']} | strict_mrr={case['strict_mrr']:.3f} | "
            f"product_mrr={case['product_mrr']:.3f} | {case['elapsed_ms']:.1f}ms"
        )
    retrieval_lines.extend(
        [
            "",
            f"Strict hit_rate={retrieval_report['strict_source_metrics']['hit_rate']:.4f}",
            f"Strict mrr={retrieval_report['strict_source_metrics']['mrr']:.4f}",
            f"Strict ndcg@{RETRIEVAL_TOP_K}={retrieval_report['strict_source_metrics'][f'ndcg@{RETRIEVAL_TOP_K}']:.4f}",
            f"Product hit_rate={retrieval_report['product_concept_metrics']['hit_rate']:.4f}",
            f"Product mrr={retrieval_report['product_concept_metrics']['mrr']:.4f}",
            f"Product ndcg@{RETRIEVAL_TOP_K}={retrieval_report['product_concept_metrics'][f'ndcg@{RETRIEVAL_TOP_K}']:.4f}",
            f"Avg latency_ms={retrieval_report['product_concept_metrics']['avg_latency_ms']:.1f}",
        ]
    )
    _write_text(output_dir / "rag_retrieval_eval.log", "\n".join(retrieval_lines))

    _write_json(output_dir / "rag_ragas_eval.json", ragas_report)
    ragas_lines = ["RAG RAGAS Evaluation", ""]
    if ragas_error:
        ragas_lines.append(f"Evaluator init blocked: {ragas_error}")
    for profile in EVAL_PROFILES:
        ragas_lines.append(f"[{profile}]")
        for case in ragas_cases_by_profile[profile]:
            if "metrics" in case:
                metric_preview = ", ".join(
                    f"{name}={float(value):.4f}" for name, value in sorted(case["metrics"].items())
                )
                ragas_lines.append(
                    f"- OK | {case['query']} | {metric_preview} | gen={case['generation_elapsed_ms']:.1f}ms judge={case['judge_elapsed_ms']:.1f}ms"
                )
            else:
                ragas_lines.append(f"- ERR | {case['query']} | {case.get('error', 'unknown error')}")
        ragas_lines.append("")
    _write_text(output_dir / "rag_ragas_eval.log", "\n".join(ragas_lines))

    return retrieval_report, ragas_report


def _start_worker(settings_path: Path, output_dir: Path) -> subprocess.Popen[str]:
    log_path = output_dir / "ingestion_worker.log"
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(  # noqa: S603 - local trusted command
        [sys.executable, str(REPO_ROOT / "scripts" / "run_ingestion_worker.py"), "--settings", str(settings_path)],
        cwd=REPO_ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    logger.info("Started ingestion worker pid=%s", proc.pid)
    return proc


def _stop_worker(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:  # pragma: no cover - runtime cleanup
        proc.kill()
        proc.wait(timeout=5)


def _run_agent_eval(app: Any, settings_path: Path, agent_golden_path: Path) -> dict[str, Any]:
    settings = load_settings(settings_path)
    trace_service = TraceService(source="postgres", postgres_dsn=settings.postgres.dsn)
    runner = AgentEvalRunner(agent=app.state.agent, trace_service=trace_service)
    return runner.run(agent_golden_path).to_dict()


def _build_parse_assessment(ingestion_summary: dict[str, Any], retrieval_report: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    entries = ingestion_summary.get("entries", [])
    for entry in entries:
        if not entry["success"]:
            issues.append(f"{entry['filename']}: ingestion failed ({entry['error'] or 'unknown error'})")
        elif entry["integrity_skipped"]:
            continue
        elif entry["chunk_count"] <= 0:
            issues.append(f"{entry['filename']}: zero chunks produced")
        elif entry["size_bytes"] >= 500_000 and entry["chunk_count"] <= 2:
            issues.append(f"{entry['filename']}: chunk count suspiciously low ({entry['chunk_count']}) for file size {entry['size_bytes']}")

    misses_by_ext: dict[str, list[str]] = defaultdict(list)
    expected_by_ext: dict[str, int] = defaultdict(int)
    source_to_ext = {
        entry["filename"].casefold(): entry["ext"]
        for entry in entries
    }
    for case in retrieval_report.get("query_results", []):
        expected_sources = case.get("expected_sources", [])
        if not expected_sources:
            continue
        ext = source_to_ext.get(expected_sources[0], "")
        if ext:
            expected_by_ext[ext] += 1
            if case.get("strict_hit_rate", 0.0) <= 0:
                misses_by_ext[ext].append(case["query"])

    for ext, total in expected_by_ext.items():
        misses = misses_by_ext.get(ext, [])
        if total > 0 and len(misses) == total:
            issues.append(f"all retrieval eval queries for {ext} sources missed ({len(misses)}/{total})")

    return {
        "needs_loader_hardening": bool(issues),
        "issues": issues,
        "note": (
            "No direct parser hardening trigger found."
            if not issues
            else "Loader hardening recommended before trusting production retrieval quality."
        ),
    }


def _build_markdown_report(
    *,
    settings_path: Path,
    runtime_settings_paths: dict[str, Path],
    output_dir: Path,
    collection: str,
    services: list[dict[str, Any]],
    ingestion_summary: dict[str, Any],
    retrieval_report: dict[str, Any],
    ragas_report: dict[str, Any],
    agent_report: dict[str, Any],
    parse_assessment: dict[str, Any],
) -> str:
    service_rows = [
        [
            item["name"],
            "ready" if item["ready"] else "not-ready",
            f"{item['elapsed_s']:.2f}",
            str(item.get("detail", "") or "—"),
        ]
        for item in services
    ]
    ingestion_rows = [
        [
            entry["filename"],
            entry["ext"],
            entry["source_type"],
            "yes" if entry["success"] else "no",
            str(entry["chunk_count"]),
            str(entry["image_count"]),
            "yes" if entry["integrity_skipped"] else "no",
            entry["error"] or "—",
        ]
        for entry in ingestion_summary.get("entries", [])
    ]
    retrieval_misses = [
        f"- {case['query']} -> expected={case['expected_sources']} retrieved={case['retrieved_sources']}"
        for case in retrieval_report.get("query_results", [])
        if case.get("strict_hit_rate", 0.0) <= 0
    ]
    product_misses = [
        f"- {case['query']} -> acceptable={case['product_sources']} retrieved={case['retrieved_sources']}"
        for case in retrieval_report.get("query_results", [])
        if case.get("product_hit_rate", 0.0) <= 0
    ]
    ragas_sections: list[str] = []
    for profile in EVAL_PROFILES:
        profile_report = ragas_report.get("profiles", {}).get(profile, {})
        metrics = profile_report.get("aggregate_metrics", {})
        ragas_sections.append(f"### {profile}")
        ragas_sections.append(f"- Faithfulness：{metrics.get('faithfulness', 0.0):.4f}")
        ragas_sections.append(f"- Answer Relevancy：{metrics.get('answer_relevancy', 0.0):.4f}")
        ragas_sections.append(f"- Context Precision：{metrics.get('context_precision', 0.0):.4f}")
        errors = [
            f"- {case['query']} -> {case.get('error', 'unknown error')}"
            for case in profile_report.get("query_results", [])
            if "metrics" not in case
        ]
        ragas_sections.append("")
        ragas_sections.append("失败项：")
        ragas_sections.append("\n".join(errors) if errors else "- 无")
        ragas_sections.append("")

    agent_profile_rows = []
    for profile in EVAL_PROFILES:
        profile_report = agent_report.get("profiles", {}).get(profile, {})
        metrics = profile_report.get("aggregate_metrics", {})
        agent_profile_rows.append(
            [
                profile,
                f"{metrics.get('success_rate', 0.0):.4f}",
                f"{metrics.get('expected_tool_recall', 0.0):.4f}",
                f"{metrics.get('citation_presence_rate', 0.0):.4f}",
                f"{metrics.get('avg_grounding_score', 0.0):.4f}",
                f"{metrics.get('avg_latency_ms', 0.0):.1f}",
                f"{metrics.get('knowledge_query_latency_ms', 0.0):.1f}",
                f"{metrics.get('heavy_task_latency_ms', 0.0):.1f}",
            ]
        )
    balanced_cases = agent_report.get("profiles", {}).get("balanced_fast", {}).get("case_results", [])
    agent_case_rows = [
        [
            case.get("id", ""),
            ",".join(case.get("actual_tool_chain", []) or ["—"]),
            case.get("actual_planner_intent", "") or "—",
            case.get("actual_control_mode", "") or "—",
            f"{float(case.get('grounding_score', 0.0)):.3f}",
            f"{float(case.get('elapsed_ms', 0.0)):.1f}",
            case.get("error", "") or "—",
        ]
        for case in balanced_cases
    ]
    runtime_config_lines = [
        f"- `{profile}`: `{path}`"
        for profile, path in runtime_settings_paths.items()
    ]

    report = f"""\
        # 生产栈评测报告（{DATE_STAMP}）

        ## 1. 环境与配置

        - 生产配置：`{settings_path}`
        - 运行时配置副本：
        {chr(10).join(runtime_config_lines)}
        - 评测产物目录：`{output_dir}`
        - 独立评测 collection：`{collection}`
        - 评测路线：Postgres + Redis + MinIO + Milvus + OpenSearch

        ### 存储栈健康检查

        {_format_table(["Service", "Status", "Ready In(s)", "Detail"], service_rows)}

        ## 2. 实际入库文件列表

        - docs 目录：`{ingestion_summary.get('docs_dir', '')}`
        - 文件总数：{ingestion_summary.get('file_count', 0)}
        - 成功：{ingestion_summary.get('success_count', 0)}
        - 失败：{ingestion_summary.get('failure_count', 0)}
        - integrity skip：{ingestion_summary.get('integrity_skipped_count', 0)}
        - 总 chunks：{ingestion_summary.get('total_chunks', 0)}
        - 总 images：{ingestion_summary.get('total_images', 0)}

        {_format_table(["File", "Ext", "Source Type", "Success", "Chunks", "Images", "Skipped", "Error"], ingestion_rows)}

        ## 3. RAG 检索评测

        ### 严格口径（strict source）

        - Hit Rate：{retrieval_report.get('strict_source_metrics', {}).get('hit_rate', 0.0):.4f}
        - MRR：{retrieval_report.get('strict_source_metrics', {}).get('mrr', 0.0):.4f}
        - NDCG@{RETRIEVAL_TOP_K}：{retrieval_report.get('strict_source_metrics', {}).get(f'ndcg@{RETRIEVAL_TOP_K}', 0.0):.4f}
        - 平均耗时：{retrieval_report.get('strict_source_metrics', {}).get('avg_latency_ms', 0.0):.1f} ms

        ### 产品口径（product concept）

        - Hit Rate：{retrieval_report.get('product_concept_metrics', {}).get('hit_rate', 0.0):.4f}
        - MRR：{retrieval_report.get('product_concept_metrics', {}).get('mrr', 0.0):.4f}
        - NDCG@{RETRIEVAL_TOP_K}：{retrieval_report.get('product_concept_metrics', {}).get(f'ndcg@{RETRIEVAL_TOP_K}', 0.0):.4f}
        - 平均耗时：{retrieval_report.get('product_concept_metrics', {}).get('avg_latency_ms', 0.0):.1f} ms

        ### 严格口径失配项

        {chr(10).join(retrieval_misses) if retrieval_misses else "- 无"}

        ### 产品口径失配项

        {chr(10).join(product_misses) if product_misses else "- 无"}

        ## 4. RAG 回答质量评测（RAGAS）

        {chr(10).join(ragas_sections)}

        ## 5. Agent 全链路评测

        {_format_table(["Profile", "Success", "Tool Recall", "Citation", "Grounding", "Avg Latency(ms)", "Knowledge(ms)", "Heavy(ms)"], agent_profile_rows)}

        ### balanced_fast Case 明细

        {_format_table(["Case", "Actual Tools", "Planner Intent", "Control Mode", "Grounding", "Elapsed(ms)", "Error"], agent_case_rows)}

        ## 6. PDF / DOCX / PPTX 解析结论

        - 是否需要加强解析链路：{"是" if parse_assessment.get("needs_loader_hardening") else "否"}
        - 结论：{parse_assessment.get("note", "")}

        ### 触发证据

        {chr(10).join(f"- {item}" for item in parse_assessment.get("issues", [])) if parse_assessment.get("issues") else "- 未发现明确触发项"}

        ## 7. 下一步建议

        - 若要把本次指标作为简历或汇报口径，建议在固定版本课程语料和固定 golden set 下再跑一轮复测，避免受缓存和供应商抖动影响。
        - 针对检索 miss 的问题，优先回查对应源文档 chunk 质量、标题边界和术语覆盖情况，再决定是否需要对 PDF / DOCX / PPTX loader 做局部加固。
        - 若后续要长期跑生产评测，建议把本脚本接入定时任务，并把 `logs/eval` 结果沉淀到历史目录做回归对比。
        """
    return "\n".join(
        line[8:] if line.startswith("        ") else line
        for line in report.splitlines()
    ).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production-stack overnight evaluation.")
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS), help="Base production settings file")
    parser.add_argument("--compose-file", default=str(DEFAULT_COMPOSE), help="docker-compose file for storage stack")
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR), help="Directory containing course docs")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN_PATH), help="RAG golden test set path")
    parser.add_argument("--agent-golden", default=str(DEFAULT_AGENT_GOLDEN_PATH), help="Agent golden test set path")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Isolated evaluation collection")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for logs/artifacts")
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH), help="Markdown report output path")
    parser.add_argument(
        "--reuse-ingested",
        action="store_true",
        help="Reuse an existing successful ingestion summary for the target collection instead of resetting backends and re-ingesting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    _configure_logging(output_dir)

    settings_path = Path(args.settings).resolve()
    compose_file = Path(args.compose_file).resolve()
    docs_dir = Path(args.docs_dir).resolve()
    golden_path = Path(args.golden).resolve()
    agent_golden_path = Path(args.agent_golden).resolve()
    report_path = Path(args.report_path).resolve()
    collection = str(args.collection)
    product_source_map = _load_product_source_map(DEFAULT_PRODUCT_SOURCE_MAP)

    runtime_settings_paths = {
        profile: output_dir / f"settings.runtime.{profile}.yaml"
        for profile in EVAL_PROFILES
    }
    runtime_agent_golden_path = output_dir / "agent_golden_test_set.runtime.json"
    import_file = docs_dir / "计算机网络（第8版）_谢希仁_上半.pdf"

    logger.info("Building runtime config and runtime agent golden set...")
    for profile, runtime_settings_path in runtime_settings_paths.items():
        _build_runtime_config(
            settings_path,
            runtime_settings_path,
            collection,
            response_profile=profile,
        )
    _build_runtime_agent_golden(agent_golden_path, runtime_agent_golden_path, import_file)
    os.environ["MODULAR_RAG_SETTINGS_PATH"] = str(runtime_settings_paths["balanced_fast"])

    logger.info("Starting production storage stack...")
    _start_storage_stack(compose_file)
    services = _wait_for_stack(runtime_settings_paths["balanced_fast"], compose_file)
    milvus_recovery = _recover_milvus_load_state(
        runtime_settings_paths["balanced_fast"],
        compose_file,
        collection,
    )
    services.append(
        {
            "name": "milvus_recovery",
            "ready": True,
            "elapsed_s": 0.0,
            "detail": json.dumps(milvus_recovery, ensure_ascii=False),
        }
    )

    ingestion_summary: dict[str, Any] | None = None
    if args.reuse_ingested:
        ingestion_summary = _maybe_reuse_ingestion_summary(output_dir, docs_dir, collection)
        if ingestion_summary is not None:
            services.append(
                {
                    "name": "eval_backend_reset",
                    "ready": True,
                    "elapsed_s": 0.0,
                    "detail": json.dumps(
                        {
                            "collection": collection,
                            "skipped": True,
                            "reason": "reuse_ingested",
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            services.append(
                {
                    "name": "ingestion_reuse",
                    "ready": True,
                    "elapsed_s": 0.0,
                    "detail": json.dumps(
                        {
                            "collection": collection,
                            "reused": True,
                            "file_count": ingestion_summary.get("file_count", 0),
                            "total_chunks": ingestion_summary.get("total_chunks", 0),
                        },
                        ensure_ascii=False,
                    ),
                }
            )

    if ingestion_summary is None:
        reset_diagnostics = _reset_eval_backends(
            runtime_settings_paths["balanced_fast"],
            collection,
        )
        services.append(
            {
                "name": "eval_backend_reset",
                "ready": True,
                "elapsed_s": 0.0,
                "detail": json.dumps(reset_diagnostics, ensure_ascii=False),
            }
        )

        logger.info("Running explicit ingestion into collection %s", collection)
        ingestion_summary = _ingest_documents(
            runtime_settings_paths["balanced_fast"],
            docs_dir,
            collection,
            output_dir,
        )

    logger.info("Creating balanced_fast runtime app for production-path evaluation")
    app = None
    app = _create_runtime_app(runtime_settings_paths["balanced_fast"])
    cache_invalidation = _invalidate_eval_semantic_cache(app, collection)
    services.append(
        {
            "name": "semantic_cache_invalidation",
            "ready": True,
            "elapsed_s": 0.0,
            "detail": json.dumps(cache_invalidation, ensure_ascii=False),
        }
    )

    logger.info("Running RAG retrieval and RAGAS evaluation")
    retrieval_report, ragas_report = _run_rag_eval(
        app,
        runtime_settings_paths["balanced_fast"],
        golden_path,
        collection,
        output_dir,
        product_source_map=product_source_map,
    )
    agent_cache_invalidation = _invalidate_eval_semantic_cache(app, collection)
    services.append(
        {
            "name": "semantic_cache_invalidation_agent",
            "ready": True,
            "elapsed_s": 0.0,
            "detail": json.dumps(agent_cache_invalidation, ensure_ascii=False),
        }
    )

    worker_proc: subprocess.Popen[str] | None = None
    try:
        logger.info("Starting ingestion worker for full agent runtime parity")
        worker_proc = _start_worker(runtime_settings_paths["balanced_fast"], output_dir)
        time.sleep(3.0)

        logger.info("Running end-to-end agent evaluation across profiles")
        agent_report = {"profiles": {}}
        for profile in EVAL_PROFILES:
            app.state.agent.config.response_profile = profile
            agent_report["profiles"][profile] = _run_agent_eval(
                app,
                runtime_settings_paths[profile],
                runtime_agent_golden_path,
            )
        _write_json(output_dir / "agent_eval.json", agent_report)
        lines = ["Agent Evaluation", ""]
        for profile in EVAL_PROFILES:
            report = agent_report["profiles"][profile]
            lines.append(f"[{profile}]")
            lines.append(
                f"case_count={report.get('case_count', 0)} total_elapsed_ms={report.get('total_elapsed_ms', 0):.1f}"
            )
            for key, value in sorted((report.get("aggregate_metrics") or {}).items()):
                lines.append(f"{key}={value}")
            lines.append("")
        _write_text(output_dir / "agent_eval.log", "\n".join(lines))
    finally:
        _stop_worker(worker_proc)

    try:
        parse_assessment = _build_parse_assessment(ingestion_summary, retrieval_report)
        report_text = _build_markdown_report(
            settings_path=settings_path,
            runtime_settings_paths=runtime_settings_paths,
            output_dir=output_dir,
            collection=collection,
            services=services,
            ingestion_summary=ingestion_summary,
            retrieval_report=retrieval_report,
            ragas_report=ragas_report,
            agent_report=agent_report,
            parse_assessment=parse_assessment,
        )
        _write_text(report_path, report_text)
        logger.info("Report written to %s", report_path)
    finally:
        _close_runtime_app(app)


if __name__ == "__main__":
    main()
