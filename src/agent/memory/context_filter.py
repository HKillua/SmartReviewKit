"""Context engineering filter — multi-level progressive compression.

Levels:
  1. Sliding window — keep the most recent ``max_messages``
  2. Tool result offloading — large tool outputs saved to file, replaced with ref
  3. Historical summarization — old messages summarized by LLM (CoPaw-style compaction)
  4. Token budget — allocate token budget across prompt sections (future)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Optional

from src.agent.memory.base import ConversationFilter
from src.agent.types import Message

logger = logging.getLogger(__name__)

_COMPACTION_PROMPT = """请将以下旧对话历史压缩为一段简洁摘要，保留所有关键信息。

要求保留：
1. 用户的学习目标和已提出的问题
2. 已解决的问题和关键结论
3. 用户表现出的薄弱点和掌握较好的知识
4. 重要的文件路径、函数名、术语
5. 下一步计划

对话历史：
{history}

请直接输出压缩摘要（不要加任何前缀），控制在300字以内："""


class ContextEngineeringFilter(ConversationFilter):
    """4-level context compression strategy.

    When ``compaction_threshold`` is set and an LLM service is provided,
    messages exceeding the threshold are summarised into a single
    ``[COMPACTED_SUMMARY]`` message (Level 3, inspired by CoPaw Compaction).
    """

    def __init__(
        self,
        *,
        max_messages: int = 40,
        tool_result_max_chars: int = 2000,
        offload_dir: str = "data/context_offload",
        llm_service: object | None = None,
        compaction_threshold: int = 30,
        compaction_keep_recent: int = 10,
    ) -> None:
        self._max_messages = max_messages
        self._tool_max = tool_result_max_chars
        self._offload_dir = Path(offload_dir)
        self._offload_dir.mkdir(parents=True, exist_ok=True)
        self._llm = llm_service
        self._compact_threshold = compaction_threshold
        self._compact_keep = compaction_keep_recent
        self._cached_compactions: dict[str, str] = {}

    def filter_messages(self, messages: list[Message]) -> list[Message]:
        msgs = list(messages)

        # Level 2: offload large tool results
        msgs = self._offload_tool_results(msgs)

        # Level 3: LLM compaction (sync wrapper — Agent can also call filter_messages_async)
        if self._llm and len(msgs) > self._compact_threshold:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Inside an async context — schedule but can't await here,
                # fall through to Level 1 sliding window
                pass
            else:
                msgs = asyncio.run(self._compact_messages(msgs))

        # Level 1: sliding window (fallback if compaction didn't run)
        if len(msgs) > self._max_messages:
            msgs = msgs[-self._max_messages:]

        return msgs

    async def filter_messages_async(self, messages: list[Message]) -> list[Message]:
        """Async version that supports Level 3 LLM compaction."""
        msgs = list(messages)

        # Level 2
        msgs = self._offload_tool_results(msgs)

        # Level 3
        if self._llm and len(msgs) > self._compact_threshold:
            msgs = await self._compact_messages(msgs)

        # Level 1 fallback
        if len(msgs) > self._max_messages:
            msgs = msgs[-self._max_messages:]

        return msgs

    async def _compact_messages(self, messages: list[Message]) -> list[Message]:
        """Compress old messages into a single summary, keeping recent ones intact."""
        if len(messages) <= self._compact_keep:
            return messages

        old_msgs = messages[: -self._compact_keep]
        recent_msgs = messages[-self._compact_keep:]

        cache_key = hashlib.sha256(
            "".join(m.content or "" for m in old_msgs).encode()
        ).hexdigest()[:16]

        if cache_key in self._cached_compactions:
            summary_text = self._cached_compactions[cache_key]
        else:
            summary_text = await self._summarize_via_llm(old_msgs)
            if not summary_text:
                return messages  # LLM failed → fall through to Level 1
            self._cached_compactions[cache_key] = summary_text

        compacted_msg = Message(
            role="system",
            content=f"[对话历史摘要]\n{summary_text}",
        )
        return [compacted_msg] + recent_msgs

    async def _summarize_via_llm(self, messages: list[Message]) -> str | None:
        """Call LLM to compress messages into a summary."""
        try:
            from src.agent.types import LlmMessage, LlmRequest

            history_lines: list[str] = []
            for m in messages:
                if m.role in ("user", "assistant") and m.content:
                    role_label = "学生" if m.role == "user" else "助手"
                    history_lines.append(f"{role_label}: {m.content[:300]}")

            if not history_lines:
                return None

            history_text = "\n".join(history_lines[-40:])
            prompt = _COMPACTION_PROMPT.format(history=history_text)

            request = LlmRequest(
                messages=[LlmMessage(role="user", content=prompt)],
                temperature=0.2,
                max_tokens=500,
            )
            response = await self._llm.send_request(request)
            return response.content.strip() if response.content else None
        except Exception:
            logger.warning("LLM compaction failed", exc_info=True)
            return None

    # ── Level 2 helpers ──

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
