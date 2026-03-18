"""Memory-Agent integration — enhancer and lifecycle hook.

Provides two core components:
  MemoryContextEnhancer  — reads memory stores and formats context for injection
  MemoryRecordHook       — extracts learning signals after each conversation
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from src.agent.hooks.lifecycle import LifecycleHook
from src.agent.pacing import compute_pacing_from_conversation
from src.agent.types import Conversation
from src.core.trace.trace_collector import TraceCollector
from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# ── preference detection patterns ──
_PREF_CONCISE_RE = re.compile(r"简洁|简单|简短|brief|concise|不用太详细", re.I)
_PREF_DETAILED_RE = re.compile(r"详细|详尽|展开|细致|detailed|elaborate|深入", re.I)
_PREF_EXAM_RE = re.compile(r"考点|考试|重点|要点|exam|key.?point|应试", re.I)
_PREF_EXAMPLE_RE = re.compile(r"举例|例子|示例|example|举个", re.I)

_VALID_DETAIL_LEVELS = {"concise", "normal", "detailed"}
_VALID_STYLE_VALUES = {"default", "exam_focused", "example_heavy"}


def _hash_user_id(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]

# ── topic extraction patterns ──
_TOPIC_PATTERNS = [
    re.compile(r"(TCP|UDP|IP|HTTP|HTTPS|DNS|DHCP|ARP|ICMP|FTP|SMTP|POP3|IMAP|BGP|OSPF|RIP)", re.I),
    re.compile(r"(三次握手|四次挥手|拥塞控制|流量控制|滑动窗口|慢启动|快速重传|快速恢复)"),
    re.compile(r"(子网掩码|路由|交换机|防火墙|NAT|VLAN|以太网|无线网络|WiFi)"),
    re.compile(r"(OSI|TCP/IP|网络层|运输层|应用层|链路层|物理层|数据报|分组)"),
    re.compile(r"(第[一二三四五六七八九十\d]+章)"),
]

_LLM_EXTRACT_PROMPT = """分析以下课程学习对话，提取结构化学习数据。只返回 JSON，不要其他内容。

对话内容：
{conversation_text}

返回格式：
{{
  "topics_discussed": ["讨论到的知识点名称"],
  "weak_points_observed": ["用户表现出不理解的知识点"],
  "strong_points_observed": ["用户掌握较好的知识点"],
  "user_preference": {{"detail_level": "concise/normal/detailed", "style": "default/exam_focused/example_heavy"}},
  "key_questions": ["用户提出的关键问题，最多3个"],
  "summary": "一句话总结本次学习内容"
}}"""


# =====================================================================
# MemoryContextEnhancer — K6: structured, personalized injection
# =====================================================================

class MemoryContextEnhancer:
    """Reads memory stores and formats context for system prompt injection."""

    def __init__(
        self,
        *,
        student_profile: object | None = None,
        error_memory: object | None = None,
        knowledge_map: object | None = None,
        skill_memory: object | None = None,
        session_memory: object | None = None,
    ) -> None:
        self._profile = student_profile
        self._errors = error_memory
        self._kmap = knowledge_map
        self._skills = skill_memory
        self._sessions = session_memory

    async def get_memory_summary(self, user_id: str) -> str:
        """Build a structured text summary for system prompt injection.

        P2: all independent DB reads run in parallel via asyncio.gather().
        """

        async def _load_profile():
            if not self._profile:
                return None
            try:
                return await self._profile.get_profile(user_id)
            except Exception:
                logger.warning("Failed to retrieve student profile")
                return None

        async def _load_sessions():
            if not self._sessions:
                return []
            try:
                return await self._sessions.get_recent_sessions(user_id, limit=2)
            except Exception:
                logger.warning("Failed to retrieve session memory")
                return []

        async def _load_kmap():
            if not self._kmap:
                return [], []
            try:
                due, weak = await asyncio.gather(
                    self._kmap.get_due_for_review(user_id),
                    self._kmap.get_weak_nodes(user_id),
                )
                return due or [], weak or []
            except Exception:
                logger.warning("Failed to retrieve knowledge map")
                return [], []

        async def _load_errors():
            if not self._errors:
                return []
            try:
                return await self._errors.get_errors(user_id, mastered=False, limit=5)
            except Exception:
                logger.warning("Failed to retrieve error memory")
                return []

        profile, recent_sessions, (due, weak), errors = await asyncio.gather(
            _load_profile(), _load_sessions(), _load_kmap(), _load_errors(),
        )

        sections: list[str] = []

        if profile is not None:
            prefs = profile.preferences or {}
            detail = prefs.get("detail_level", "normal")
            style = prefs.get("style", "default")
            pref_lines: list[str] = []
            if detail != "normal":
                label = {"concise": "简洁", "detailed": "详细"}.get(detail, detail)
                pref_lines.append(f"回答风格: {label}")
            if style != "default":
                label = {"exam_focused": "考点版", "example_heavy": "多举例"}.get(style, style)
                pref_lines.append(f"内容偏好: {label}")
            diff = prefs.get("quiz_difficulty")
            if diff and diff != "medium":
                pref_lines.append(f"测验难度: {diff}")
            if pref_lines:
                sections.append("### 学习偏好\n" + "\n".join(f"- {l}" for l in pref_lines))
            if profile.total_sessions > 0:
                stats = (
                    f"- 总会话: {profile.total_sessions}, "
                    f"总测验: {profile.total_quizzes}, "
                    f"正确率: {profile.overall_accuracy:.0%}"
                )
                sections.append("### 学习统计\n" + stats)

        if recent_sessions:
            last = recent_sessions[0]
            lines: list[str] = []
            if last.topics:
                lines.append(f"- 话题: {', '.join(last.topics[:5])}")
            if last.key_questions:
                lines.append(f"- 关键问题: {last.key_questions[0]}")
            obs = last.mastery_observations
            if obs:
                w = [k for k, v in obs.items() if v == "weak"]
                s = [k for k, v in obs.items() if v == "strong"]
                if w:
                    lines.append(f"- 薄弱: {', '.join(w[:3])}")
                if s:
                    lines.append(f"- 掌握: {', '.join(s[:3])}")
            if lines:
                sections.append("### 上次学习\n" + "\n".join(lines))

        if due:
            due_lines = []
            for n in due[:5]:
                days = ""
                if n.last_reviewed:
                    lr = n.last_reviewed
                    if lr.tzinfo is None:
                        lr = lr.replace(tzinfo=timezone.utc)
                    d = (datetime.now(timezone.utc) - lr).days
                    days = f", {d}天未复习"
                due_lines.append(f"- {n.concept} (掌握度: {n.mastery_level:.0%}{days})")
            sections.append("### 需要复习的知识点\n" + "\n".join(due_lines))

        if weak:
            weak_items = ", ".join(f"{n.concept}({n.mastery_level:.0%})" for n in weak[:5])
            sections.append(f"### 低掌握度知识\n- {weak_items}")

        if errors:
            from collections import Counter
            topic_counts = Counter(e.topic for e in errors if e.topic)
            err_lines = []
            for e in errors[:3]:
                count = topic_counts.get(e.topic, 1)
                err_lines.append(f"- {e.topic}: {e.question[:40]}… (错{count}次)")
            sections.append("### 错题提醒\n" + "\n".join(err_lines))

        if not sections:
            return ""
        return "\n\n".join(sections)

    async def enhance_system_prompt(self, base_prompt: str, user_id: str) -> str:
        summary = await self.get_memory_summary(user_id)
        if not summary:
            return base_prompt
        return base_prompt + f"\n\n## 学生记忆上下文\n{summary}"


# =====================================================================
# MemoryRecordHook — K2+K3+K5: learning data extraction & sync
# =====================================================================

class MemoryRecordHook(LifecycleHook):
    """Extracts learning signals from completed conversations and distributes
    updates to all memory stores.

    Supports two extraction modes (configurable via extraction_mode):
      - "llm": send conversation to LLM for structured JSON extraction
      - "rule": regex/heuristic extraction (no LLM cost)
      - "both": try LLM first, fallback to rule on failure
    """

    def __init__(
        self,
        *,
        student_profile: object | None = None,
        skill_memory: object | None = None,
        error_memory: object | None = None,
        knowledge_map: object | None = None,
        session_memory: object | None = None,
        llm_service: object | None = None,
        extraction_mode: str = "both",
        write_gating_enabled: bool = True,
        session_write_min_confidence: float = 0.2,
        preference_write_min_confidence: float = 0.65,
        preference_conflict_guard: bool = True,
        trace_enabled: bool = False,
        trace_collector: TraceCollector | None = None,
    ) -> None:
        self._profile = student_profile
        self._skills = skill_memory
        self._errors = error_memory
        self._kmap = knowledge_map
        self._sessions = session_memory
        self._llm = llm_service
        self._mode = extraction_mode
        self._write_gating_enabled = write_gating_enabled
        self._session_write_min_confidence = session_write_min_confidence
        self._preference_write_min_confidence = preference_write_min_confidence
        self._preference_conflict_guard = preference_conflict_guard
        self._trace_enabled = trace_enabled
        self._trace_collector = trace_collector or (TraceCollector() if trace_enabled else None)

    async def after_message(self, conversation: Conversation) -> None:
        user_id = conversation.user_id
        memory_trace: TraceContext | None = None
        if self._trace_enabled and self._trace_collector is not None:
            memory_trace = TraceContext(trace_type="memory")
            memory_trace.metadata.update(
                {
                    "conversation_id": conversation.id,
                    "user_id_hash": _hash_user_id(user_id),
                    "configured_extraction_mode": self._mode,
                    "write_gating_enabled": self._write_gating_enabled,
                    "session_write_min_confidence": self._session_write_min_confidence,
                    "preference_write_min_confidence": self._preference_write_min_confidence,
                }
            )
            memory_trace.record_stage(
                "conversation_scan",
                {
                    "message_count": len(conversation.messages),
                    "user_message_count": sum(1 for msg in conversation.messages if msg.role == "user"),
                    "assistant_message_count": sum(
                        1 for msg in conversation.messages if msg.role == "assistant"
                    ),
                    "user_text_length": len(self._format_user_text(conversation, max_chars=4000)),
                },
            )

        try:
            # Step 1: Extract learning data from conversation
            extracted = await self._extract_learning_data(conversation)
            extraction_meta = dict(extracted.get("extraction_metadata", {}))
            preference_updates = dict(extracted.get("user_preference", {}))
            should_save_session, session_reason = self._should_save_session(extracted)
            should_update_preferences, preference_reason = self._should_update_preferences(extracted)
            write_decisions = {
                "session_saved": should_save_session,
                "session_reason": session_reason,
                "profile_preferences_updated": bool(preference_updates) and should_update_preferences,
                "profile_preferences_reason": preference_reason,
            }
            extraction_meta["write_decisions"] = write_decisions
            extracted["extraction_metadata"] = extraction_meta

            if memory_trace is not None:
                memory_trace.metadata["extraction_mode_used"] = extraction_meta.get("mode", "none")
                memory_trace.metadata["write_decisions"] = write_decisions
                memory_trace.metadata["preference_conflicts"] = extraction_meta.get(
                    "preference_conflicts", []
                )
                memory_trace.metadata["confidence"] = float(
                    extraction_meta.get("confidence", 0.0) or 0.0
                )
                memory_trace.record_stage(
                    "memory_extraction",
                    {
                        "confidence": float(extraction_meta.get("confidence", 0.0) or 0.0),
                        "signal_counts": extraction_meta.get("signal_counts", {}),
                        "preference_conflicts": extraction_meta.get("preference_conflicts", []),
                        "topics_count": len(extracted.get("topics_discussed", [])),
                        "summary_present": bool(extracted.get("summary", "")),
                    },
                )
                memory_trace.record_stage(
                    "memory_quality_gate",
                    {
                        "write_decisions": write_decisions,
                        "session_reason": session_reason,
                        "profile_preferences_reason": preference_reason,
                    },
                )

            # Step 2: Save session summary (K1)
            if self._sessions and should_save_session:
                try:
                    from src.agent.memory.session_memory import SessionSummary

                    summary = SessionSummary(
                        session_id=conversation.id,
                        topics=extracted.get("topics_discussed", []),
                        key_questions=extracted.get("key_questions", []),
                        mastery_observations=self._build_mastery_obs(extracted),
                        preference_snapshot=preference_updates,
                        summary_text=extracted.get("summary", ""),
                        extraction_metadata=extraction_meta,
                    )
                    await self._sessions.save_session(user_id, summary)
                    if memory_trace is not None:
                        memory_trace.record_stage(
                            "session_memory_write",
                            {
                                "status": "saved",
                                "reason": session_reason,
                                "topics_count": len(summary.topics),
                                "question_count": len(summary.key_questions),
                            },
                        )
                except Exception as exc:
                    logger.warning("Failed to save session summary", exc_info=True)
                    if memory_trace is not None:
                        memory_trace.record_stage(
                            "session_memory_write",
                            {
                                "status": "failed",
                                "reason": "exception",
                                "error": str(exc)[:300],
                            },
                        )
            elif self._sessions:
                logger.info(
                    "Skipped session memory write for %s (%s, confidence=%.2f)",
                    conversation.id,
                    session_reason,
                    float(extraction_meta.get("confidence", 0.0) or 0.0),
                )
                if memory_trace is not None:
                    memory_trace.record_stage(
                        "session_memory_write",
                        {
                            "status": "skipped",
                            "reason": session_reason,
                        },
                    )
            elif memory_trace is not None:
                memory_trace.record_stage(
                    "session_memory_write",
                    {
                        "status": "skipped",
                        "reason": "session_memory_disabled",
                    },
                )

            # Step 3: Update student profile (K5 — fix total_sessions, sync topics)
            if self._profile:
                try:
                    profile = await self._profile.get_profile(user_id)
                    updates: dict[str, Any] = {
                        "total_sessions": profile.total_sessions + 1,
                        "last_active": datetime.now(timezone.utc),
                    }
                    pacing_level, pacing_reason = compute_pacing_from_conversation(conversation)
                    updates["learning_pace"] = pacing_level

                    # Merge preference updates
                    if preference_updates and should_update_preferences:
                        merged_prefs = dict(profile.preferences or {})
                        merged_prefs.update(
                            {
                                k: v
                                for k, v in preference_updates.items()
                                if v and v != "default" and v != "normal"
                            }
                        )
                        updates["preferences"] = merged_prefs
                    elif preference_updates:
                        logger.info(
                            "Skipped preference update for %s (%s, confidence=%.2f)",
                            conversation.id,
                            preference_reason,
                            float(extraction_meta.get("confidence", 0.0) or 0.0),
                        )

                    # Sync weak/strong from knowledge map
                    if self._kmap:
                        try:
                            weak_nodes = await self._kmap.get_weak_nodes(user_id)
                            weak_from_kmap = [n.concept for n in weak_nodes[:10]]
                            if self._errors:
                                weak_from_err = await self._errors.get_weak_concepts(user_id)
                                weak_merged = list(
                                    dict.fromkeys(weak_from_kmap + weak_from_err[:10])
                                )
                            else:
                                weak_merged = weak_from_kmap
                            if weak_merged:
                                updates["weak_topics"] = weak_merged[:15]

                            all_rows = await self._kmap._get_all_nodes(user_id)
                            strong = [n.concept for n in all_rows if n.mastery_level >= 0.8]
                            if strong:
                                updates["strong_topics"] = strong[:15]

                            total_q = sum(n.quiz_count for n in all_rows)
                            total_c = sum(n.correct_count for n in all_rows)
                            if total_q > 0:
                                updates["overall_accuracy"] = round(total_c / total_q, 3)
                                updates["total_quizzes"] = total_q
                        except Exception:
                            logger.debug("Could not sync knowledge map to profile")

                    await self._profile.update_profile(user_id, updates)
                    if memory_trace is not None:
                        profile_status = (
                            "updated" if bool(preference_updates) and should_update_preferences else "skipped"
                        )
                        profile_reason = "stats_only"
                        if preference_reason != "no_preference_signal":
                            profile_reason = preference_reason
                        memory_trace.record_stage(
                            "profile_update",
                            {
                                "status": profile_status,
                                "reason": profile_reason,
                                "updated_fields": sorted(updates.keys()),
                                "learning_pace": pacing_level,
                                "pacing_reason": pacing_reason,
                            },
                        )
                except Exception as exc:
                    logger.warning("Failed to update student profile", exc_info=True)
                    if memory_trace is not None:
                        memory_trace.record_stage(
                            "profile_update",
                            {
                                "status": "failed",
                                "reason": "exception",
                                "error": str(exc)[:300],
                            },
                        )
            elif memory_trace is not None:
                memory_trace.record_stage(
                    "profile_update",
                    {
                        "status": "skipped",
                        "reason": "student_profile_disabled",
                    },
                )

            # Step 4: Save tool chains to skill memory (existing logic)
            if self._skills:
                try:
                    saved = await self._save_tool_chains(conversation)
                    if memory_trace is not None:
                        memory_trace.record_stage(
                            "skill_memory_write",
                            {
                                "status": "saved" if saved else "skipped",
                                "reason": "tool_chain_present" if saved else "no_tool_chain",
                            },
                        )
                except Exception as exc:
                    logger.warning("Failed to save tool chains")
                    if memory_trace is not None:
                        memory_trace.record_stage(
                            "skill_memory_write",
                            {
                                "status": "failed",
                                "reason": "exception",
                                "error": str(exc)[:300],
                            },
                        )
            elif memory_trace is not None:
                memory_trace.record_stage(
                    "skill_memory_write",
                    {
                        "status": "skipped",
                        "reason": "skill_memory_disabled",
                    },
                )
        except Exception as exc:
            if memory_trace is not None:
                memory_trace.record_stage(
                    "error",
                    {
                        "phase": "after_message",
                        "error": str(exc)[:300],
                    },
                )
            raise
        finally:
            if memory_trace is not None and self._trace_collector is not None:
                self._trace_collector.collect(memory_trace)

    # ── Extraction engines ──

    async def _extract_learning_data(self, conversation: Conversation) -> dict:
        """Extract structured learning data from conversation."""
        if self._mode in ("llm", "both") and self._llm:
            try:
                result = await self._extract_via_llm(conversation)
                if result:
                    return self._postprocess_extraction(result, conversation, mode_used="llm")
            except Exception:
                logger.info("LLM extraction failed, falling back to rule-based")

        if self._mode in ("rule", "both"):
            return self._postprocess_extraction(
                self._extract_via_rules(conversation),
                conversation,
                mode_used="rule",
            )

        return self._postprocess_extraction({}, conversation, mode_used="none")

    async def _extract_via_llm(self, conversation: Conversation) -> dict | None:
        """Use LLM to extract structured learning data."""
        conv_text = self._format_conversation_text(conversation, max_chars=3000)
        if not conv_text.strip():
            return None

        from src.agent.types import LlmMessage, LlmRequest

        prompt = _LLM_EXTRACT_PROMPT.format(conversation_text=conv_text)
        request = LlmRequest(
            messages=[LlmMessage(role="user", content=prompt)],
            temperature=0.1,
            max_tokens=800,
        )
        response = await self._llm.send_request(request)
        if not response.content:
            return None

        from src.agent.utils.json_helpers import safe_parse_json
        result = safe_parse_json(response.content)
        return result if isinstance(result, dict) else None

    def _extract_via_rules(self, conversation: Conversation) -> dict:
        """Heuristic rule-based extraction from conversation messages."""
        user_texts: list[str] = []
        assistant_texts: list[str] = []
        for m in conversation.messages:
            if m.role == "user" and m.content:
                user_texts.append(m.content)
            elif m.role == "assistant" and m.content:
                assistant_texts.append(m.content)

        all_text = " ".join(user_texts + assistant_texts)
        user_all = " ".join(user_texts)

        # Extract topics
        topics: list[str] = []
        for pat in _TOPIC_PATTERNS:
            for match in pat.findall(all_text):
                if match and match not in topics:
                    topics.append(match)

        # Detect preferences from user messages
        pref: dict[str, str] = {}
        if _PREF_CONCISE_RE.search(user_all):
            pref["detail_level"] = "concise"
        elif _PREF_DETAILED_RE.search(user_all):
            pref["detail_level"] = "detailed"
        if _PREF_EXAM_RE.search(user_all):
            pref["style"] = "exam_focused"
        elif _PREF_EXAMPLE_RE.search(user_all):
            pref["style"] = "example_heavy"

        # Extract key questions (user messages that end with ？or ?)
        questions: list[str] = []
        for t in user_texts:
            if t.strip().endswith(("？", "?")) and len(t) > 5:
                questions.append(t.strip()[:100])
        questions = questions[:3]

        # Build summary
        summary = ""
        if topics:
            summary = f"讨论了{', '.join(topics[:3])}"

        return {
            "topics_discussed": topics[:10],
            "weak_points_observed": [],
            "strong_points_observed": [],
            "user_preference": pref,
            "key_questions": questions,
            "summary": summary,
        }

    def _postprocess_extraction(
        self,
        extracted: dict[str, Any] | None,
        conversation: Conversation,
        *,
        mode_used: str,
    ) -> dict[str, Any]:
        payload = extracted if isinstance(extracted, dict) else {}
        user_text = self._format_user_text(conversation, max_chars=2000)

        topics = self._normalize_list(payload.get("topics_discussed"), limit=10, max_len=48)
        weak_points = self._normalize_list(payload.get("weak_points_observed"), limit=10, max_len=48)
        strong_points = self._normalize_list(payload.get("strong_points_observed"), limit=10, max_len=48)
        key_questions = self._normalize_list(payload.get("key_questions"), limit=3, max_len=100)
        summary = self._normalize_text(payload.get("summary"), max_len=220)
        preferences = self._normalize_preferences(payload.get("user_preference"))
        conflicts = self._detect_preference_conflicts(user_text)
        if self._preference_conflict_guard:
            for key in conflicts:
                preferences.pop(key, None)

        signal_counts = {
            "topics": len(topics),
            "weak_points": len(weak_points),
            "strong_points": len(strong_points),
            "key_questions": len(key_questions),
            "has_summary": int(bool(summary)),
            "preference_fields": len(preferences),
        }
        confidence = self._compute_extraction_confidence(
            signal_counts,
            mode_used=mode_used,
            conflict_count=len(conflicts),
        )
        return {
            "topics_discussed": topics,
            "weak_points_observed": weak_points,
            "strong_points_observed": strong_points,
            "user_preference": preferences,
            "key_questions": key_questions,
            "summary": summary,
            "extraction_metadata": {
                "mode": mode_used,
                "confidence": confidence,
                "signal_counts": signal_counts,
                "preference_conflicts": conflicts,
                "write_decisions": {},
            },
        }

    # ── Helpers ──

    def _should_save_session(self, extracted: dict[str, Any]) -> tuple[bool, str]:
        meta = extracted.get("extraction_metadata", {})
        signal_counts = meta.get("signal_counts", {})
        has_signal = any(int(signal_counts.get(key, 0) or 0) > 0 for key in signal_counts)
        confidence = float(meta.get("confidence", 0.0) or 0.0)
        if not has_signal:
            return False, "no_meaningful_signal"
        if self._write_gating_enabled and confidence < self._session_write_min_confidence:
            return False, "low_confidence"
        return True, "saved"

    def _should_update_preferences(self, extracted: dict[str, Any]) -> tuple[bool, str]:
        meta = extracted.get("extraction_metadata", {})
        conflicts = meta.get("preference_conflicts", [])
        if self._preference_conflict_guard and conflicts:
            return False, "preference_conflict"
        preferences = extracted.get("user_preference", {})
        if not preferences:
            return False, "no_preference_signal"
        confidence = float(meta.get("confidence", 0.0) or 0.0)
        if self._write_gating_enabled and confidence < self._preference_write_min_confidence:
            return False, "low_confidence"
        return True, "updated"

    @staticmethod
    def _normalize_list(value: Any, *, limit: int, max_len: int) -> list[str]:
        items = value if isinstance(value, list) else []
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in items:
            text = MemoryRecordHook._normalize_text(raw, max_len=max_len)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
            if len(normalized) >= limit:
                break
        return normalized

    @staticmethod
    def _normalize_text(value: Any, *, max_len: int) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text[:max_len]

    @staticmethod
    def _normalize_preferences(value: Any) -> dict[str, str]:
        payload = value if isinstance(value, dict) else {}
        normalized: dict[str, str] = {}
        detail = str(payload.get("detail_level", "") or "").strip().lower()
        style = str(payload.get("style", "") or "").strip().lower()
        if detail in _VALID_DETAIL_LEVELS and detail != "normal":
            normalized["detail_level"] = detail
        if style in _VALID_STYLE_VALUES and style != "default":
            normalized["style"] = style
        return normalized

    @staticmethod
    def _detect_preference_conflicts(user_text: str) -> list[str]:
        conflicts: list[str] = []
        if _PREF_CONCISE_RE.search(user_text) and _PREF_DETAILED_RE.search(user_text):
            conflicts.append("detail_level")
        if _PREF_EXAM_RE.search(user_text) and _PREF_EXAMPLE_RE.search(user_text):
            conflicts.append("style")
        return conflicts

    @staticmethod
    def _compute_extraction_confidence(
        signal_counts: dict[str, int],
        *,
        mode_used: str,
        conflict_count: int,
    ) -> float:
        total_signals = sum(int(signal_counts.get(key, 0) or 0) for key in signal_counts)
        if total_signals == 0:
            return 0.0
        score = 0.15
        score += min(signal_counts.get("topics", 0), 3) * 0.18
        score += min(signal_counts.get("key_questions", 0), 2) * 0.10
        score += min(signal_counts.get("weak_points", 0) + signal_counts.get("strong_points", 0), 2) * 0.08
        score += signal_counts.get("has_summary", 0) * 0.12
        score += min(signal_counts.get("preference_fields", 0), 2) * 0.24
        if mode_used == "llm":
            score += 0.05
        score -= conflict_count * 0.25
        return round(max(0.0, min(score, 1.0)), 3)

    @staticmethod
    def _format_conversation_text(conversation: Conversation, max_chars: int = 3000) -> str:
        lines: list[str] = []
        total = 0
        for m in conversation.messages:
            if m.role in ("user", "assistant") and m.content:
                prefix = "学生" if m.role == "user" else "助手"
                line = f"{prefix}: {m.content[:500]}"
                total += len(line)
                if total > max_chars:
                    break
                lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _format_user_text(conversation: Conversation, max_chars: int = 2000) -> str:
        texts: list[str] = []
        total = 0
        for message in conversation.messages:
            if message.role != "user" or not message.content:
                continue
            content = message.content.strip()
            if not content:
                continue
            total += len(content)
            if total > max_chars:
                break
            texts.append(content)
        return "\n".join(texts)

    @staticmethod
    def _build_mastery_obs(extracted: dict) -> dict[str, str]:
        obs: dict[str, str] = {}
        for t in extracted.get("weak_points_observed", []):
            obs[t] = "weak"
        for t in extracted.get("strong_points_observed", []):
            obs[t] = "strong"
        return obs

    async def _save_tool_chains(self, conversation: Conversation) -> bool:
        tool_chain: list[str] = []
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
                try:
                    await self._skills.save_usage(conversation.user_id, record)
                    return True
                except Exception:
                    logger.warning("Failed to save tool usage record")
        return False
