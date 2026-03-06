"""ReviewScheduleHook — proactive review recommendations at session start.

On the first message of a new conversation, this hook:
  1. Triggers Ebbinghaus decay on the knowledge map
  2. Checks for concepts due for review
  3. Checks for unmastered error records
  4. Checks recent session topics for continuity
  5. Injects a review recommendation into the agent context

The hook does NOT modify the user message; instead it stores the
recommendation text in ``self.last_recommendation`` for the agent to
read and inject into the system prompt (via MemoryContextEnhancer or
directly in Agent.chat).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.agent.hooks.lifecycle import LifecycleHook

logger = logging.getLogger(__name__)


class ReviewScheduleHook(LifecycleHook):
    """Generates proactive review suggestions at the start of each conversation."""

    def __init__(
        self,
        *,
        knowledge_map: object | None = None,
        error_memory: object | None = None,
        session_memory: object | None = None,
        enable_decay: bool = True,
        decay_cooldown_hours: int = 12,
    ) -> None:
        self._kmap = knowledge_map
        self._errors = error_memory
        self._sessions = session_memory
        self._enable_decay = enable_decay
        self._decay_cooldown = timedelta(hours=decay_cooldown_hours)
        self._last_decay: dict[str, datetime] = {}
        self.last_recommendation: str = ""
        self._triggered_conversations: set[str] = set()

    async def before_message(self, user_id: str, message: str) -> Optional[str]:
        """Generate review recommendation for new conversations.

        This is called for every message, but we only generate a
        recommendation once per conversation (tracked by a flag).
        The Agent should call this before building the system prompt.
        """
        try:
            recommendation = await self._generate_recommendation(user_id)
            self.last_recommendation = recommendation
        except Exception:
            logger.warning("ReviewScheduleHook failed", exc_info=True)
            self.last_recommendation = ""

        return None  # never modify the user message

    async def get_review_context(self, user_id: str) -> str:
        """Public API for Agent to call directly when building system prompt."""
        try:
            return await self._generate_recommendation(user_id)
        except Exception:
            logger.warning("Failed to generate review context", exc_info=True)
            return ""

    async def _generate_recommendation(self, user_id: str) -> str:
        lines: list[str] = []

        # ── Step 1: Ebbinghaus decay ──
        if self._enable_decay and self._kmap:
            now = datetime.now(timezone.utc)
            last = self._last_decay.get(user_id)
            if last is None or (now - last) > self._decay_cooldown:
                try:
                    decayed = await self._kmap.apply_decay(user_id)
                    self._last_decay[user_id] = now
                    if decayed > 0:
                        logger.info("Applied decay to %d knowledge nodes for %s", decayed, user_id)
                except Exception:
                    logger.debug("Decay failed for %s", user_id)

        # ── Step 2: Due for review ──
        if self._kmap:
            try:
                due_nodes = await self._kmap.get_due_for_review(user_id)
                for n in due_nodes[:3]:
                    days = ""
                    if n.last_reviewed:
                        d = (datetime.now() - n.last_reviewed).days
                        days = f"，已{d}天未复习"
                    lines.append(
                        f"- 知识点「{n.concept}」掌握度 {n.mastery_level:.0%}{days}，建议今天复习"
                    )
            except Exception:
                logger.debug("Failed to check due-for-review nodes")

        # ── Step 3: Unmastered errors ──
        if self._errors:
            try:
                errors = await self._errors.get_errors(user_id, mastered=False, limit=3)
                for e in errors[:2]:
                    lines.append(
                        f"- 错题提醒：「{e.topic}」— {e.question[:50]}…"
                    )
            except Exception:
                logger.debug("Failed to check error memory")

        # ── Step 4: Last session continuity ──
        if self._sessions:
            try:
                recent = await self._sessions.get_recent_sessions(user_id, limit=1)
                if recent:
                    last_sess = recent[0]
                    if last_sess.topics:
                        topics_str = "、".join(last_sess.topics[:3])
                        lines.append(f"- 上次学习了「{topics_str}」")
                        weak_obs = [
                            k for k, v in last_sess.mastery_observations.items()
                            if v == "weak"
                        ]
                        if weak_obs:
                            lines.append(f"  其中「{'、'.join(weak_obs[:2])}」掌握较弱，建议继续巩固")
            except Exception:
                logger.debug("Failed to check session history")

        if not lines:
            return ""

        return "### 主动复习建议\n" + "\n".join(lines)
