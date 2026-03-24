"""Tests for the task-level planner."""

from __future__ import annotations

from src.agent.planner import ControlMode, TaskIntent, TaskPlanner
from src.agent.skills.registry import SkillPolicy


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


def _fake_skill_match(text: str) -> str | None:
    lowered = text.lower()
    if "考前复习" in text or "期末复习" in text or "考试复习" in text:
        return "exam_prep"
    if "错题复盘" in text or "错题回顾" in text:
        return "error_review"
    if "刷题" in text or "出题" in text:
        return "quiz_drill"
    if "章节深入" in text or "深入讲" in text:
        return "chapter_deep_dive"
    if "知识图谱" in text or "掌握度" in text:
        return "knowledge_check"
    return None


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


def test_explicit_intro_query_hits_rule_before_embedding() -> None:
    planner = TaskPlanner(embedding_fn=_fake_embed)
    decision = planner.plan("请简单介绍一下 TCP 三次握手")

    assert decision.task_intent == TaskIntent.KNOWLEDGE_QUERY
    assert decision.match_method == "rule"
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


def test_plain_message_with_zai_does_not_false_split() -> None:
    planner = TaskPlanner()
    decision = planner.plan("我们再看一下 HTTP 为什么需要 TLS")

    assert decision.is_composite is False
    assert decision.task_intent == TaskIntent.KNOWLEDGE_QUERY


def test_sequence_composite_tracks_skill_interval() -> None:
    planner = TaskPlanner(skill_match_fn=_fake_skill_match)
    policy = SkillPolicy(
        allowed_tools=[
            "concept_graph_query",
            "knowledge_query",
            "review_summary",
            "quiz_generator",
            "network_calc",
        ],
        allow_autonomous=True,
        max_steps=5,
    )

    decision = planner.plan(
        "先讲一下 TCP 和 UDP 的区别，再帮我考前复习传输层，再出两道题",
        matched_skill="exam_prep",
        skill_policy=policy,
    )

    assert decision.is_composite is True
    assert decision.matched_skill == "exam_prep"
    assert decision.skill_start_index == 1
    assert decision.skill_end_index == 2
    assert decision.planner_execution_model == "skill_guided_agenda"
    assert [subtask.task_intent for subtask in decision.subtasks] == [
        TaskIntent.KNOWLEDGE_QUERY,
        TaskIntent.REVIEW_SUMMARY,
        TaskIntent.QUIZ_GENERATOR,
    ]


def test_skill_interval_stops_at_incompatible_tool() -> None:
    planner = TaskPlanner(skill_match_fn=_fake_skill_match)
    policy = SkillPolicy(
        allowed_tools=[
            "concept_graph_query",
            "knowledge_query",
            "review_summary",
            "quiz_generator",
            "network_calc",
        ],
        allow_autonomous=True,
        max_steps=5,
    )

    decision = planner.plan(
        "先帮我考前复习传输层，再把这个 PDF 导入知识库，最后出两道题",
        matched_skill="exam_prep",
        skill_policy=policy,
    )

    assert decision.is_composite is True
    assert decision.skill_start_index == 0
    assert decision.skill_end_index == 0
    assert [subtask.selected_tool for subtask in decision.subtasks] == [
        "review_summary",
        "document_ingest",
        "quiz_generator",
    ]
