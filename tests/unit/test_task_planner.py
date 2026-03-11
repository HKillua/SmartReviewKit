"""Tests for the task-level planner."""

from __future__ import annotations

from src.agent.planner import ControlMode, TaskIntent, TaskPlanner


def _fake_embed(texts):
    vectors = []
    for text in texts:
        lowered = text.lower()
        if any(token in lowered for token in ("drill", "quiz", "练习", "出题", "题")):
            vectors.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif any(token in lowered for token in ("review", "复习", "考点", "总结")):
            vectors.append([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        elif any(token in lowered for token in ("grade", "判分", "批改", "评估")):
            vectors.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        elif any(token in lowered for token in ("import", "导入", "入库", ".pdf", ".pptx")):
            vectors.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        elif any(token in lowered for token in ("hello", "thanks", "你好", "谢谢")):
            vectors.append([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        else:
            vectors.append([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return vectors


def test_rule_hit_review_summary_forces_tool() -> None:
    planner = TaskPlanner()
    decision = planner.plan("帮我总结一下 DNS 解析的复习要点")

    assert decision.task_intent == TaskIntent.REVIEW_SUMMARY
    assert decision.control_mode == ControlMode.FORCE_TOOL
    assert decision.selected_tool == "review_summary"
    assert decision.match_method == "rule"


def test_embedding_fallback_can_force_quiz_generator() -> None:
    planner = TaskPlanner(embedding_fn=_fake_embed)
    decision = planner.plan("Drill me on UDP basics")

    assert decision.task_intent == TaskIntent.QUIZ_GENERATOR
    assert decision.match_method == "embedding"
    assert decision.control_mode == ControlMode.FORCE_TOOL


def test_knowledge_query_is_only_advisory() -> None:
    planner = TaskPlanner()
    decision = planner.plan("TCP 三次握手为什么是三次？")

    assert decision.task_intent == TaskIntent.KNOWLEDGE_QUERY
    assert decision.control_mode == ControlMode.ADVISORY


def test_general_chat_passes_through() -> None:
    planner = TaskPlanner()
    decision = planner.plan("你好")

    assert decision.task_intent == TaskIntent.GENERAL_CHAT
    assert decision.control_mode == ControlMode.PASS_THROUGH
