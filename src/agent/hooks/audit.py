"""Audit logging — JSONL file-based audit trail for all tool calls."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.agent.hooks.lifecycle import LifecycleHook
from src.agent.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = {"password", "secret", "token", "api_key", "apikey", "authorization"}


class AuditEvent(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    user_id: str = ""
    action: str = ""
    tool_name: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


def _sanitize(params: dict[str, Any]) -> dict[str, Any]:
    sanitized = {}
    for k, v in params.items():
        if k.lower() in _SENSITIVE_KEYS:
            sanitized[k] = "***REDACTED***"
        elif isinstance(v, dict):
            sanitized[k] = _sanitize(v)
        else:
            sanitized[k] = v
    return sanitized


class FileAuditLogger:
    """Appends AuditEvent records as JSONL to a file."""

    def __init__(self, log_path: str = "logs/audit.jsonl") -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: AuditEvent) -> None:
        event.parameters = _sanitize(event.parameters)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")


class AuditHook(LifecycleHook):
    """Records audit events for tool calls via before_tool / after_tool."""

    def __init__(self, audit_logger: FileAuditLogger | None = None) -> None:
        self._logger = audit_logger or FileAuditLogger()
        self._start_times: dict[str, float] = {}

    async def before_tool(self, tool_name: str, context: ToolContext) -> None:
        self._start_times[context.request_id] = time.monotonic()

    async def after_tool(
        self, tool_name: str, result: ToolResult, context: ToolContext | None = None,
    ) -> Optional[ToolResult]:
        key = context.request_id if context else ""
        start = self._start_times.pop(key, time.monotonic())
        duration = (time.monotonic() - start) * 1000

        event = AuditEvent(
            action="tool_call",
            tool_name=tool_name,
            user_id=context.user_id if context else "",
            result_summary=result.result_for_llm[:200] if result.success else (result.error or "")[:200],
            duration_ms=round(duration, 2),
            metadata={"success": result.success},
        )
        try:
            self._logger.log_event(event)
        except Exception:
            logger.warning("Failed to write audit event")
        return None
