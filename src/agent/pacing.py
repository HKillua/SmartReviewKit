"""Adaptive pacing helpers shared by Agent runtime and memory hooks."""

from __future__ import annotations

import re
from statistics import mean

from src.agent.types import Conversation

_DEEP_QUERY_RE = re.compile(r"(为什么|底层|更深入|深入一点|原理|本质|机制)", re.IGNORECASE)
_QUIZ_CORRECT_RE = re.compile(r"(判定[:：]\s*✅\s*正确|✅\s*正确)")
_QUIZ_INCORRECT_RE = re.compile(r"(判定[:：]\s*❌\s*错误|❌\s*错误)")
_QUIZ_PARTIAL_RE = re.compile(r"(⚠️\s*部分正确|部分正确)")


def _tool_call_name_map(conversation: Conversation) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for message in conversation.messages:
        if message.role != "assistant" or not message.tool_calls:
            continue
        for call in message.tool_calls:
            if call.id:
                mapping[str(call.id)] = str(call.name)
    return mapping


def extract_recent_quiz_outcomes(
    conversation: Conversation,
    *,
    limit: int = 3,
) -> list[str]:
    """Return recent quiz outcomes as correct / incorrect / partial."""
    tool_names = _tool_call_name_map(conversation)
    outcomes: list[str] = []
    for message in conversation.messages:
        if message.role != "tool" or not message.tool_call_id:
            continue
        if tool_names.get(str(message.tool_call_id)) != "quiz_evaluator":
            continue
        metadata = getattr(message, "metadata", {}) or {}
        batch_results = metadata.get("batch_results", [])
        if isinstance(batch_results, list) and batch_results:
            for item in batch_results:
                if not isinstance(item, dict):
                    continue
                verdict = str(item.get("verdict", "") or "").strip().lower()
                if verdict in {"correct", "incorrect", "partial"}:
                    outcomes.append(verdict)
            continue
        content = (message.content or "").strip()
        if not content:
            continue
        if _QUIZ_CORRECT_RE.search(content):
            outcomes.append("correct")
        elif _QUIZ_INCORRECT_RE.search(content):
            outcomes.append("incorrect")
        elif _QUIZ_PARTIAL_RE.search(content):
            outcomes.append("partial")
    return outcomes[-max(limit, 1):]


def compute_pacing_from_conversation(
    conversation: Conversation,
) -> tuple[str, str]:
    """Infer the current teaching pace from quiz outcomes and user behavior."""
    outcomes = extract_recent_quiz_outcomes(conversation, limit=3)
    if len(outcomes) >= 2:
        recent = outcomes[-2:]
        if recent == ["correct", "correct"]:
            return "accelerate", "最近连续两次判题正确，可直接推进到更高阶内容。"
        if recent == ["incorrect", "incorrect"]:
            return "decelerate", "最近连续两次判题错误，需要放慢速度并补前置知识。"

    recent_user_messages = [
        (msg.content or "").strip()
        for msg in conversation.messages
        if msg.role == "user" and (msg.content or "").strip()
    ][-2:]
    if any(_DEEP_QUERY_RE.search(text) for text in recent_user_messages):
        return "accelerate", "用户持续追问原理和底层机制，适合提升讲解深度。"
    if recent_user_messages:
        avg_length = mean(len(text) for text in recent_user_messages)
        if avg_length < 5:
            return "decelerate", "最近两条用户消息都较短，可能存在困惑或不耐烦信号。"

    return "normal", "最近没有明显的做题趋势或交互信号，保持正常节奏。"
