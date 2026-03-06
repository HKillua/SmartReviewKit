"""Memory-Agent integration — enhancer and lifecycle hook.

Provides two core components:
  MemoryContextEnhancer  — reads memory stores and formats context for injection
  MemoryRecordHook       — extracts learning signals after each conversation
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Optional

from src.agent.hooks.lifecycle import LifecycleHook
from src.agent.types import Conversation

logger = logging.getLogger(__name__)

# ── preference detection patterns ──
_PREF_CONCISE_RE = re.compile(r"简洁|简单|简短|brief|concise|不用太详细", re.I)
_PREF_DETAILED_RE = re.compile(r"详细|详尽|展开|细致|detailed|elaborate|深入", re.I)
_PREF_EXAM_RE = re.compile(r"考点|考试|重点|要点|exam|key.?point|应试", re.I)
_PREF_EXAMPLE_RE = re.compile(r"举例|例子|示例|example|举个", re.I)

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
        """Build a structured text summary for system prompt injection."""
        sections: list[str] = []

        # ── Preference section ──
        if self._profile:
            try:
                profile = await self._profile.get_profile(user_id)
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
            except Exception:
                logger.warning("Failed to retrieve student profile")

        # ── Last session section ──
        if self._sessions:
            try:
                recent = await self._sessions.get_recent_sessions(user_id, limit=2)
                if recent:
                    last = recent[0]
                    lines: list[str] = []
                    if last.topics:
                        lines.append(f"- 话题: {', '.join(last.topics[:5])}")
                    if last.key_questions:
                        lines.append(f"- 关键问题: {last.key_questions[0]}")
                    obs = last.mastery_observations
                    if obs:
                        weak = [k for k, v in obs.items() if v == "weak"]
                        strong = [k for k, v in obs.items() if v == "strong"]
                        if weak:
                            lines.append(f"- 薄弱: {', '.join(weak[:3])}")
                        if strong:
                            lines.append(f"- 掌握: {', '.join(strong[:3])}")
                    if lines:
                        sections.append("### 上次学习\n" + "\n".join(lines))
            except Exception:
                logger.warning("Failed to retrieve session memory")

        # ── Due for review ──
        if self._kmap:
            try:
                due = await self._kmap.get_due_for_review(user_id)
                if due:
                    due_lines = []
                    for n in due[:5]:
                        days = ""
                        if n.last_reviewed:
                            d = (datetime.now() - n.last_reviewed).days
                            days = f", {d}天未复习"
                        due_lines.append(f"- {n.concept} (掌握度: {n.mastery_level:.0%}{days})")
                    sections.append("### 需要复习的知识点\n" + "\n".join(due_lines))

                weak = await self._kmap.get_weak_nodes(user_id)
                if weak:
                    weak_items = ", ".join(f"{n.concept}({n.mastery_level:.0%})" for n in weak[:5])
                    sections.append(f"### 低掌握度知识\n- {weak_items}")
            except Exception:
                logger.warning("Failed to retrieve knowledge map")

        # ── Error reminders ──
        if self._errors:
            try:
                errors = await self._errors.get_errors(user_id, mastered=False, limit=5)
                if errors:
                    err_lines = []
                    for e in errors[:3]:
                        err_lines.append(f"- {e.topic}: {e.question[:40]}… (错{1}次)")
                    sections.append("### 错题提醒\n" + "\n".join(err_lines))
            except Exception:
                logger.warning("Failed to retrieve error memory")

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
    ) -> None:
        self._profile = student_profile
        self._skills = skill_memory
        self._errors = error_memory
        self._kmap = knowledge_map
        self._sessions = session_memory
        self._llm = llm_service
        self._mode = extraction_mode

    async def after_message(self, conversation: Conversation) -> None:
        user_id = conversation.user_id

        # Step 1: Extract learning data from conversation
        extracted = await self._extract_learning_data(conversation)

        # Step 2: Save session summary (K1)
        if self._sessions and extracted:
            try:
                from src.agent.memory.session_memory import SessionSummary

                summary = SessionSummary(
                    session_id=conversation.id,
                    topics=extracted.get("topics_discussed", []),
                    key_questions=extracted.get("key_questions", []),
                    mastery_observations=self._build_mastery_obs(extracted),
                    preference_snapshot=extracted.get("user_preference", {}),
                    summary_text=extracted.get("summary", ""),
                )
                await self._sessions.save_session(user_id, summary)
            except Exception:
                logger.warning("Failed to save session summary", exc_info=True)

        # Step 3: Update student profile (K5 — fix total_sessions, sync topics)
        if self._profile:
            try:
                profile = await self._profile.get_profile(user_id)
                updates: dict[str, Any] = {
                    "total_sessions": profile.total_sessions + 1,
                    "last_active": datetime.now(),
                }

                # Merge preference updates
                pref = extracted.get("user_preference", {})
                if pref:
                    merged_prefs = dict(profile.preferences or {})
                    merged_prefs.update({k: v for k, v in pref.items() if v and v != "default" and v != "normal"})
                    updates["preferences"] = merged_prefs

                # Sync weak/strong from knowledge map
                if self._kmap:
                    try:
                        weak_nodes = await self._kmap.get_weak_nodes(user_id)
                        weak_from_kmap = [n.concept for n in weak_nodes[:10]]
                        if self._errors:
                            weak_from_err = await self._errors.get_weak_concepts(user_id)
                            weak_merged = list(dict.fromkeys(weak_from_kmap + weak_from_err[:10]))
                        else:
                            weak_merged = weak_from_kmap
                        if weak_merged:
                            updates["weak_topics"] = weak_merged[:15]

                        all_rows = await self._kmap._get_all_nodes(user_id)
                        strong = [n.concept for n in all_rows if n.mastery_level >= 0.8]
                        if strong:
                            updates["strong_topics"] = strong[:15]

                        # Compute overall accuracy
                        total_q = sum(n.quiz_count for n in all_rows)
                        total_c = sum(n.correct_count for n in all_rows)
                        if total_q > 0:
                            updates["overall_accuracy"] = round(total_c / total_q, 3)
                            updates["total_quizzes"] = total_q
                    except Exception:
                        logger.debug("Could not sync knowledge map to profile")

                await self._profile.update_profile(user_id, updates)
            except Exception:
                logger.warning("Failed to update student profile", exc_info=True)

        # Step 4: Save tool chains to skill memory (existing logic)
        if self._skills:
            try:
                self._save_tool_chains(conversation)
            except Exception:
                logger.warning("Failed to save tool chains")

    # ── Extraction engines ──

    async def _extract_learning_data(self, conversation: Conversation) -> dict:
        """Extract structured learning data from conversation."""
        if self._mode in ("llm", "both") and self._llm:
            try:
                result = await self._extract_via_llm(conversation)
                if result:
                    return result
            except Exception:
                logger.info("LLM extraction failed, falling back to rule-based")

        if self._mode in ("rule", "both"):
            return self._extract_via_rules(conversation)

        return {}

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

        text = response.content.strip()
        # Strip markdown code fence if present
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)

        return json.loads(text)

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

    # ── Helpers ──

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
    def _build_mastery_obs(extracted: dict) -> dict[str, str]:
        obs: dict[str, str] = {}
        for t in extracted.get("weak_points_observed", []):
            obs[t] = "weak"
        for t in extracted.get("strong_points_observed", []):
            obs[t] = "strong"
        return obs

    def _save_tool_chains(self, conversation: Conversation) -> None:
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
                import asyncio
                from src.agent.memory.skill_memory import ToolUsageRecord

                record = ToolUsageRecord(
                    question_pattern=user_msg[:200],
                    tool_chain=tool_chain,
                    quality_score=0.8,
                )
                asyncio.create_task(self._skills.save_usage(conversation.user_id, record))
