"""Memory-Agent integration — enhancer and lifecycle hook."""

from __future__ import annotations

import logging
from typing import Optional

from src.agent.hooks.lifecycle import LifecycleHook
from src.agent.types import Conversation

logger = logging.getLogger(__name__)


class MemoryContextEnhancer:
    """Retrieves memory state and formats it for injection into the system prompt."""

    def __init__(
        self,
        *,
        student_profile: object | None = None,
        error_memory: object | None = None,
        knowledge_map: object | None = None,
        skill_memory: object | None = None,
    ) -> None:
        self._profile = student_profile
        self._errors = error_memory
        self._kmap = knowledge_map
        self._skills = skill_memory

    async def get_memory_summary(self, user_id: str) -> str:
        """Build a concise text summary of all memory for system prompt injection."""
        sections: list[str] = []

        if self._profile:
            try:
                profile = await self._profile.get_profile(user_id)
                if profile.total_sessions > 0:
                    sections.append(
                        f"- 学习节奏: {profile.learning_pace}, "
                        f"总会话: {profile.total_sessions}, "
                        f"总测验: {profile.total_quizzes}, "
                        f"正确率: {profile.overall_accuracy:.0%}"
                    )
                if profile.weak_topics:
                    sections.append(f"- 薄弱主题: {', '.join(profile.weak_topics[:5])}")
                if profile.strong_topics:
                    sections.append(f"- 擅长主题: {', '.join(profile.strong_topics[:5])}")
            except Exception:
                logger.warning("Failed to retrieve student profile")

        if self._errors:
            try:
                errors = await self._errors.get_errors(user_id, mastered=False, limit=5)
                if errors:
                    sections.append(f"- 待复习错题: {len(errors)} 道")
                    concepts = await self._errors.get_weak_concepts(user_id)
                    if concepts:
                        sections.append(f"- 易错知识点: {', '.join(concepts[:8])}")
            except Exception:
                logger.warning("Failed to retrieve error memory")

        if self._kmap:
            try:
                due = await self._kmap.get_due_for_review(user_id)
                if due:
                    sections.append(f"- 需复习知识点: {', '.join(n.concept for n in due[:5])}")
                weak = await self._kmap.get_weak_nodes(user_id)
                if weak:
                    sections.append(
                        f"- 低掌握度知识: "
                        + ", ".join(f"{n.concept}({n.mastery_level:.0%})" for n in weak[:5])
                    )
            except Exception:
                logger.warning("Failed to retrieve knowledge map")

        if not sections:
            return ""
        return "\n".join(sections)

    async def enhance_system_prompt(self, base_prompt: str, user_id: str) -> str:
        summary = await self.get_memory_summary(user_id)
        if not summary:
            return base_prompt
        return base_prompt + f"\n\n## 学生记忆上下文\n{summary}"


class MemoryRecordHook(LifecycleHook):
    """Records learning signals from completed conversations into memory stores."""

    def __init__(
        self,
        *,
        student_profile: object | None = None,
        skill_memory: object | None = None,
    ) -> None:
        self._profile = student_profile
        self._skills = skill_memory

    async def after_message(self, conversation: Conversation) -> None:
        user_id = conversation.user_id

        # Update session count
        if self._profile:
            try:
                await self._profile.update_profile(user_id, {"total_sessions": 1})
            except Exception:
                logger.warning("Failed to update student profile after message")

        # Extract successful tool chains for skill memory
        if self._skills:
            try:
                tool_chain = []
                for m in conversation.messages:
                    if m.tool_calls:
                        for tc in m.tool_calls:
                            tool_chain.append(tc.name)
                if tool_chain and len(conversation.messages) >= 2:
                    user_msg = next(
                        (m.content for m in conversation.messages if m.role == "user" and m.content),
                        None,
                    )
                    if user_msg:
                        from src.agent.memory.skill_memory import ToolUsageRecord
                        record = ToolUsageRecord(
                            question_pattern=user_msg[:200],
                            tool_chain=tool_chain,
                            quality_score=0.8,
                        )
                        await self._skills.save_usage(user_id, record)
            except Exception:
                logger.warning("Failed to save tool usage to skill memory")
