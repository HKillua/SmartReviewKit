"""Post-action artifact exports for learning-agent workflows."""

from __future__ import annotations

import asyncio
import csv
import io
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class ArtifactRecord:
    artifact_id: str
    artifact_type: str
    filename: str
    content_type: str
    object_key: str
    download_url: str

    def to_metadata(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "filename": self.filename,
            "content_type": self.content_type,
            "object_key": self.object_key,
            "download_url": self.download_url,
        }


class ArtifactPostActionAdapter:
    """MCP-shaped post-action adapter backed by object storage artifacts."""

    def __init__(self, *, object_store: Any | None = None) -> None:
        self._object_store = object_store

    async def run(
        self,
        *,
        user_id: str,
        conversation_id: str,
        matched_skill: str,
        post_actions: list[str],
        final_text: str,
        tool_path: list[str],
    ) -> list[dict[str, str]]:
        if self._object_store is None or not post_actions or not final_text.strip():
            return []
        artifacts: list[dict[str, str]] = []
        payload = {
            "matched_skill": matched_skill,
            "final_text": final_text,
            "tool_path": list(tool_path),
            "conversation_id": conversation_id,
            "user_id": user_id,
        }
        for action in post_actions:
            built = self._build_artifact(
                action=action,
                conversation_id=conversation_id,
                payload=payload,
            )
            if built is None:
                continue
            record = built
            await asyncio.to_thread(
                self._object_store.put_bytes,
                record.object_key,
                built_content(record),
                record.content_type,
            )
            artifacts.append(record.to_metadata())
        return artifacts

    def _build_artifact(
        self,
        *,
        action: str,
        conversation_id: str,
        payload: dict[str, Any],
    ) -> ArtifactRecord | None:
        normalized = str(action or "").strip().lower()
        suffix = uuid.uuid4().hex[:10]
        if normalized == "notes_export":
            filename = f"notes_{suffix}.md"
            object_key = f"artifacts/{conversation_id}/{filename}"
            return _set_artifact_payload(ArtifactRecord(
                artifact_id=filename,
                artifact_type="notes_export",
                filename=filename,
                content_type="text/markdown; charset=utf-8",
                object_key=object_key,
                download_url=f"/api/artifacts/{filename}?conversation_id={conversation_id}&user_id={payload.get('user_id', '')}",
            ), payload)
        if normalized == "flashcard_export":
            filename = f"flashcards_{suffix}.tsv"
            object_key = f"artifacts/{conversation_id}/{filename}"
            return _set_artifact_payload(ArtifactRecord(
                artifact_id=filename,
                artifact_type="flashcard_export",
                filename=filename,
                content_type="text/tab-separated-values; charset=utf-8",
                object_key=object_key,
                download_url=f"/api/artifacts/{filename}?conversation_id={conversation_id}&user_id={payload.get('user_id', '')}",
            ), payload)
        if normalized == "schedule_export":
            filename = f"schedule_{suffix}.md"
            object_key = f"artifacts/{conversation_id}/{filename}"
            return _set_artifact_payload(ArtifactRecord(
                artifact_id=filename,
                artifact_type="schedule_export",
                filename=filename,
                content_type="text/markdown; charset=utf-8",
                object_key=object_key,
                download_url=f"/api/artifacts/{filename}?conversation_id={conversation_id}&user_id={payload.get('user_id', '')}",
            ), payload)
        return None


def built_content(record: ArtifactRecord) -> bytes:
    """Content is reconstructed lazily from filename convention by helper builders."""
    artifact_type = record.artifact_type
    if artifact_type == "notes_export":
        return _build_notes_markdown(record).encode("utf-8")
    if artifact_type == "flashcard_export":
        return _build_flashcards_tsv(record).encode("utf-8")
    if artifact_type == "schedule_export":
        return _build_schedule_markdown(record).encode("utf-8")
    return b""


def _artifact_payload(record: ArtifactRecord) -> dict[str, Any]:
    return getattr(record, "_payload", {})


def _set_artifact_payload(record: ArtifactRecord, payload: dict[str, Any]) -> ArtifactRecord:
    setattr(record, "_payload", payload)
    return record


def _build_notes_markdown(record: ArtifactRecord) -> str:
    payload = _artifact_payload(record)
    lines = [
        f"# 学习笔记：{payload.get('matched_skill') or '学习任务'}",
        "",
        "## 本轮工具路径",
        f"- {' -> '.join(payload.get('tool_path') or ['无'])}",
        "",
        "## 结论摘要",
        payload.get("final_text", "").strip(),
        "",
        "## 复习建议",
        "- 先回看上述结构化讲解，再结合原课件核对关键流程和定义。",
    ]
    return "\n".join(lines).strip() + "\n"


def _build_flashcards_tsv(record: ArtifactRecord) -> str:
    payload = _artifact_payload(record)
    final_text = str(payload.get("final_text", "") or "").strip()
    sentences = [line.strip("- ").strip() for line in final_text.splitlines() if line.strip()]
    rows = [["Front", "Back", "Tags"]]
    for index, sentence in enumerate(sentences[:5], start=1):
        rows.append([f"卡片 {index}", sentence[:160], payload.get("matched_skill") or "learning_agent"])
    output = io.StringIO()
    writer = csv.writer(output, delimiter="\t")
    writer.writerows(rows)
    return output.getvalue()


def _build_schedule_markdown(record: ArtifactRecord) -> str:
    payload = _artifact_payload(record)
    lines = [
        "# 复习计划",
        "",
        "## 第 1 天",
        "- 通读本轮总结，标出仍不理解的概念",
        "- 回看原课件中的核心定义与流程图",
        "",
        "## 第 2 天",
        "- 针对薄弱点做 2~3 道练习",
        "- 对计算题或状态机题重新手推一遍",
        "",
        "## 第 3 天",
        "- 做一次限时回忆，口头复述关键流程",
        "- 对照本轮结果补齐遗漏概念",
        "",
        "## 本轮依据",
        payload.get("final_text", "").strip(),
    ]
    return "\n".join(lines).strip() + "\n"
