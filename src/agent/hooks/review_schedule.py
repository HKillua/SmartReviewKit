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
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.agent.hooks.lifecycle import LifecycleHook

logger = logging.getLogger(__name__)

_LOW_INFO_RE = re.compile(r"^(你好|您好|hi|hello|hey|在吗|在不在|有人吗|嗨)$", re.IGNORECASE)
_COURSE_KEYWORD_RE = re.compile(
    r"(tcp|udp|dns|ip|http|osi|三次握手|四次挥手|拥塞控制|流量控制|子网|crc|滑动窗口|路由|网络层|传输层|应用层)",
    re.IGNORECASE,
)


def _empty_proactive_metadata() -> dict[str, Any]:
    return {
        "proactive_triggered": False,
        "proactive_reason": "",
        "proactive_signals": {},
    }


class ReviewScheduleHook(LifecycleHook):
    """Generates proactive review suggestions at the start of each conversation."""

    def __init__(
        self,
        *,
        knowledge_map: object | None = None,
        error_memory: object | None = None,
        student_profile: object | None = None,
        session_memory: object | None = None,
        enable_decay: bool = True,
        decay_cooldown_hours: int = 12,
    ) -> None:
        self._kmap = knowledge_map
        self._errors = error_memory
        self._profile = student_profile
        self._sessions = session_memory
        self._enable_decay = enable_decay
        self._decay_cooldown = timedelta(hours=decay_cooldown_hours)
        self._last_decay: dict[str, datetime] = {}
        self.last_recommendation: str = ""
        self.last_metadata: dict[str, Any] = _empty_proactive_metadata()
        self._last_cache_key: tuple[str, str] | None = None
        self._triggered_conversations: set[str] = set()

    async def before_message(self, user_id: str, message: str) -> Optional[str]:
        """Generate review recommendation for new conversations.

        This is called for every message, but we only generate a
        recommendation once per conversation (tracked by a flag).
        The Agent should call this before building the system prompt.
        """
        try:
            recommendation, metadata = await self._generate_recommendation(user_id, message=message)
            self.last_recommendation = recommendation
            self.last_metadata = metadata
            self._last_cache_key = (user_id, message or "")
        except Exception:
            logger.warning("ReviewScheduleHook failed", exc_info=True)
            self.last_recommendation = ""
            self.last_metadata = _empty_proactive_metadata()
            self._last_cache_key = None

        return None  # never modify the user message

    async def get_review_context(self, user_id: str, message: str = "") -> tuple[str, dict[str, Any]]:
        """Public API for Agent to call directly when building system prompt."""
        cache_key = (user_id, message or "")
        if self._last_cache_key == cache_key:
            return self.last_recommendation, dict(self.last_metadata)
        try:
            recommendation, metadata = await self._generate_recommendation(user_id, message=message)
            self.last_recommendation = recommendation
            self.last_metadata = metadata
            self._last_cache_key = cache_key
            return recommendation, metadata
        except Exception:
            logger.warning("Failed to generate review context", exc_info=True)
            self.last_metadata = _empty_proactive_metadata()
            return "", dict(self.last_metadata)

    def _is_low_information_message(self, message: str) -> bool:
        text = (message or "").strip()
        if not text:
            return True
        if _LOW_INFO_RE.match(text):
            return True
        compact = re.sub(r"\s+", "", text)
        return len(compact) < 5 and _COURSE_KEYWORD_RE.search(text) is None

    @staticmethod
    def _days_since_last_active(value: object | None) -> int | None:
        if value is None or not isinstance(value, datetime):
            return None
        current = value
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - current
        return max(delta.days, 0)

    async def _generate_recommendation(
        self,
        user_id: str,
        *,
        message: str = "",
    ) -> tuple[str, dict[str, Any]]:
        if not self._is_low_information_message(message):
            return (
                "",
                {
                    "proactive_triggered": False,
                    "proactive_reason": "message_not_low_information",
                    "proactive_signals": {},
                },
            )

        lines: list[str] = []
        signals: dict[str, Any] = {}

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

        # ── Step 2: Decayed weak nodes ──
        if self._kmap:
            try:
                decayed_nodes = await self._kmap.get_decayed_nodes(
                    user_id,
                    threshold=0.45,
                    limit=5,
                )
                if decayed_nodes:
                    signals["decayed"] = [
                        {
                            "concept": node.concept,
                            "mastery_level": round(float(node.mastery_level), 3),
                        }
                        for node in decayed_nodes
                    ]
                    lines.append("以下知识点掌握度已明显衰减：")
                    for node in decayed_nodes[:3]:
                        lines.append(
                            f"- {node.concept}：当前掌握度 {node.mastery_level:.0%}，建议快速复习"
                        )
            except Exception:
                logger.debug("Failed to check decayed nodes")

        # ── Step 3: Weak concepts from error memory ──
        if self._errors:
            try:
                weak_concepts = await self._errors.get_weak_concepts(user_id)
                if weak_concepts:
                    signals["weak_concepts"] = list(weak_concepts[:5])
                    lines.append(
                        f"高频薄弱知识点：{'、'.join(weak_concepts[:3])}。"
                    )
            except Exception:
                logger.debug("Failed to check error memory")

        # ── Step 4: Inactivity signal ──
        if self._profile:
            try:
                profile = await self._profile.get_profile(user_id)
                inactivity_days = self._days_since_last_active(getattr(profile, "last_active", None))
                if inactivity_days is not None and inactivity_days > 0:
                    signals["inactivity_days"] = inactivity_days
                    lines.insert(0, f"该学生已 {inactivity_days} 天未活跃。")
            except Exception:
                logger.debug("Failed to check student profile")

        if not lines:
            return (
                "",
                {
                    "proactive_triggered": False,
                    "proactive_reason": "no_decay_or_weak_signals",
                    "proactive_signals": signals,
                },
            )

        return (
            "[Proactive Insight]\n" + "\n".join(lines) + "\n建议主动询问用户是否需要快速复习这些内容。",
            {
                "proactive_triggered": True,
                "proactive_reason": "low_information_message_with_learning_signals",
                "proactive_signals": signals,
            },
        )
