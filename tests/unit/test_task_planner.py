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


def test_rule_hit_builds_composite_plan_in_user_order() -> None:
    planner = TaskPlanner()
    decision = planner.plan("帮我先总结 TCP 的重点，再出 3 道题")

    assert decision.is_composite is True
    assert decision.primary_intent == TaskIntent.REVIEW_SUMMARY
    assert [subtask.task_intent for subtask in decision.subtasks] == [
        TaskIntent.REVIEW_SUMMARY,
        TaskIntent.QUIZ_GENERATOR,
    ]
    assert decision.subtasks[0].selected_tool == "review_summary"
    assert decision.subtasks[1].selected_tool == "quiz_generator"


def test_rule_hit_builds_three_step_composite_plan() -> None:
    planner = TaskPlanner()
    decision = planner.plan("先解释 DNS，再帮我总结，最后出一道选择题")

    assert decision.is_composite is True
    assert [subtask.task_intent for subtask in decision.subtasks] == [
        TaskIntent.KNOWLEDGE_QUERY,
        TaskIntent.REVIEW_SUMMARY,
        TaskIntent.QUIZ_GENERATOR,
    ]
    assert decision.subtasks[0].source_span[0] < decision.subtasks[1].source_span[0]
    assert decision.subtasks[1].source_span[0] < decision.subtasks[2].source_span[0]


def test_embedding_only_fallback_does_not_create_composite_plan() -> None:
    planner = TaskPlanner(embedding_fn=_fake_embed)
    decision = planner.plan("Drill me on UDP basics")

    assert decision.is_composite is False
    assert decision.task_intent == TaskIntent.QUIZ_GENERATOR
