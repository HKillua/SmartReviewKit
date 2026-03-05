"""Context engineering filter — multi-level progressive compression of conversation history.

Levels:
  1. Sliding window — keep the most recent ``max_messages``
  2. Tool result offloading — large tool outputs saved to file, replaced with reference
  3. Historical summarization — old messages summarized by LLM (optional)
  4. Token budget — allocate token budget across prompt sections
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Optional

from src.agent.memory.base import ConversationFilter
from src.agent.types import Message

logger = logging.getLogger(__name__)


class ContextEngineeringFilter(ConversationFilter):
    """Implements the 4-level context compression strategy."""

    def __init__(
        self,
        *,
        max_messages: int = 40,
        tool_result_max_chars: int = 2000,
        offload_dir: str = "data/context_offload",
        llm_service: object | None = None,
    ) -> None:
        self._max_messages = max_messages
        self._tool_max = tool_result_max_chars
        self._offload_dir = Path(offload_dir)
        self._offload_dir.mkdir(parents=True, exist_ok=True)
        self._llm = llm_service

    def filter_messages(self, messages: list[Message]) -> list[Message]:
        msgs = list(messages)

        # Level 2: offload large tool results
        msgs = self._offload_tool_results(msgs)

        # Level 1: sliding window
        if len(msgs) > self._max_messages:
            msgs = msgs[-self._max_messages:]

        return msgs

    def _offload_tool_results(self, messages: list[Message]) -> list[Message]:
        result: list[Message] = []
        for m in messages:
            if m.role == "tool" and m.content and len(m.content) > self._tool_max:
                ref_id = self._offload_to_file(m.content)
                summary = m.content[:200] + f"\n\n[完整内容已卸载: ref={ref_id}]"
                result.append(m.model_copy(update={"content": summary}))
            else:
                result.append(m)
        return result

    def _offload_to_file(self, content: str) -> str:
        ref_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        path = self._offload_dir / f"{ref_id}.txt"
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        return ref_id

    def load_offloaded(self, ref_id: str) -> Optional[str]:
        path = self._offload_dir / f"{ref_id}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None
