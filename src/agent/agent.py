"""Agent core — ReAct tool-loop orchestrator with streaming output.

This is the central class that ties together LLM, Tools, Conversation,
LifecycleHooks, LlmMiddlewares, Memory and Skills.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from src.agent.config import AgentConfig
from src.agent.conversation import ConversationStore
from src.agent.grounding import (
    DEFAULT_BALANCED_LOW_EVIDENCE_NOTE,
    DEFAULT_LOW_EVIDENCE_MESSAGE,
    EvidenceBundle,
    GroundingAssessment,
    GroundingEvaluator,
    GroundingPolicyAction,
    build_evidence_summary,
    build_conservative_rewrite_messages,
    build_evidence_bundle,
    build_grounding_context,
    extract_citation_indices,
)
from src.agent.hooks.lifecycle import LifecycleHook
from src.agent.hooks.middleware import LlmMiddleware
from src.agent.llm.base import LlmService
from src.agent.planner import (
    ControlMode,
    PlannedSubtask,
    PlannerDecision,
    TaskIntent,
    TaskPlanner,
)
from src.agent.pacing import compute_pacing_from_conversation
from src.agent.quiz_batch import build_quiz_batch_alignment
from src.agent.prompt_builder import SystemPromptBuilder
from src.agent.skills.registry import SkillPolicy
from src.agent.tools.base import ToolRegistry
from src.agent.types import (
    AgendaGoal,
    AgendaState,
    Conversation,
    GoalStatus,
    LlmMessage,
    LlmRequest,
    LlmResponse,
    LlmStreamChunk,
    Message,
    RequestStatus,
    StreamEvent,
    StreamEventType,
    ToolCallData,
    ToolContext,
    ToolErrorType,
    ToolResult,
)
from src.core.trace.trace_collector import TraceCollector
from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

_TRACE_TEXT_LIMIT = 300
_CN_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_QUIZ_COUNT_RE = re.compile(r"(?P<count>\d+|[一二两三四五六七八九十])\s*道")
_QUESTION_TYPE_PATTERNS = (
    ("选择题", re.compile(r"选择题", re.IGNORECASE)),
    ("填空题", re.compile(r"填空题", re.IGNORECASE)),
    ("简答题", re.compile(r"简答题", re.IGNORECASE)),
    ("SQL题", re.compile(r"sql题|sql 题", re.IGNORECASE)),
)
_TOPIC_CLEANUP_PATTERNS = (
    re.compile(r"[，。！？；,!?;]+"),
    re.compile(r"\b(请|帮我|麻烦你|先|再|然后|接着|最后|并且|并|顺便|同时)\b"),
    re.compile(r"解释|讲解|说明|分析|详细解释|详细介绍|为什么|原理|是什么|区别|对比|比较"),
    re.compile(r"复习|总结|梳理|回顾|考点|重点|复习摘要|复习重点|考试复习|期末复习"),
    re.compile(r"出题|做题|练习|习题|刷题|测验|生成.*?题|来.*?道题|出.*?道题|题目"),
    re.compile(r"判分|评分|批改|评估答案|帮我判|检查答案|我的答案是|请帮我判"),
    re.compile(r"(?P<count>\d+|[一二两三四五六七八九十])\s*道"),
    re.compile(r"选择题|填空题|简答题|sql题|SQL题"),
    re.compile(r"难度\s*[1-5]|基础|中等|困难|简单"),
)
_EVALUATOR_FIELD_PATTERNS = {
    "question": re.compile(r"题目[:：]\s*(.+?)(?=(?:用户答案|我的答案|正确答案|$))", re.S),
    "user_answer": re.compile(r"(?:用户答案|我的答案)[:：]\s*(.+?)(?=(?:正确答案|$))", re.S),
    "correct_answer": re.compile(r"正确答案[:：]\s*(.+)$", re.S),
}
_DIRECT_INGEST_PATH_RE = re.compile(
    r"((?:/|\.{1,2}/|docs/)[^\n\r]*?\.(?:pdf|pptx))",
    re.IGNORECASE,
)
_NETWORK_CALC_HINT_RE = re.compile(
    r"(子网|掩码|cidr|crc|香农|奈奎斯特|吞吐|时延|rtt|利用率|窗口|go-back-n|selective repeat|/\d{1,2}|(?:\d{1,3}\.){3}\d{1,3})",
    re.IGNORECASE,
)
_PROTOCOL_SIM_HINT_RE = re.compile(
    r"(三次握手|四次挥手|拥塞控制|慢启动|快速重传|快速恢复|syn|ack|fin|超时|丢包|乱序|状态机|rip|路由更新)",
    re.IGNORECASE,
)
REPLAN_POLICY = """
## [Replan Policy]
每次工具调用后都评估结果是否足够：
1. knowledge_query 返回结果过少或证据不足：优先换更宽泛的主题、切换到 concept_graph_query，最后才允许做有限常识补充。
2. concept_graph_query 未找到结果：放宽主题名或切换到 knowledge_query 获取课程证据。
3. protocol_state_simulator 不支持当前协议：降级为 knowledge_query + 文字讲解。
4. network_calc 参数错误：优先向用户确认参数，不要强行计算。
当你决定调整计划时，请在 Thought 中明确说明原因和新的工具选择。
""".strip()

_WAITING_FOR_ANSWER_RE = re.compile(
    r"(我的答案是|用户答案|第\s*\d+\s*题|第[一二两三四五六七八九十]+\s*题|答案[:：]|回答[:：])",
    re.IGNORECASE,
)


@dataclass
class GoalExecutionResult:
    text: str = ""
    completion_hint: str = "step_done"
    goal_status: str = GoalStatus.COMPLETED.value
    metadata: dict[str, Any] = field(default_factory=dict)
    resume_payload: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def _serialize_subtasks(subtasks: list[PlannedSubtask]) -> list[dict[str, object]]:
    return [subtask.to_metadata() for subtask in subtasks]


def _hash_user_id(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def _truncate_text(value: str | None, limit: int = _TRACE_TEXT_LIMIT) -> str:
    if not value:
        return ""
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _summarize_arguments(arguments: dict) -> dict:
    summary: dict[str, str | int | float | bool | None] = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            summary[key] = _truncate_text(value, limit=120)
        elif isinstance(value, (int, float, bool)) or value is None:
            summary[key] = value
        else:
            summary[key] = _truncate_text(str(value), limit=120)
    return summary


def _restrict_tool_schemas(tool_schemas: list[dict], tool_name: str) -> list[dict]:
    restricted = [
        schema
        for schema in tool_schemas
        if schema.get("function", {}).get("name") == tool_name
    ]
    return restricted or tool_schemas


def _restrict_tool_schemas_to_allowed(
    tool_schemas: list[dict],
    allowed_tools: set[str] | list[str],
) -> list[dict]:
    allowed = {str(name) for name in allowed_tools if str(name).strip()}
    if not allowed:
        return tool_schemas
    restricted = [
        schema
        for schema in tool_schemas
        if schema.get("function", {}).get("name") in allowed
    ]
    return restricted or tool_schemas


def _exclude_tool_schema(tool_schemas: list[dict], tool_name: str) -> list[dict]:
    return [
        schema
        for schema in tool_schemas
        if schema.get("function", {}).get("name") != tool_name
    ]


def _build_planner_context(decision: PlannerDecision | None) -> str:
    if decision is None or decision.control_mode == ControlMode.PASS_THROUGH:
        return ""
    if decision.is_composite and decision.subtasks:
        lines = [
            f"任务识别: composite({decision.primary_intent.value if decision.primary_intent else decision.task_intent.value})",
            f"控制策略: {decision.control_mode.value}",
            f"匹配方式: {decision.match_method}",
            f"执行顺序: {decision.ordering_method or 'declared_order'}",
        ]
        if decision.matched_skill:
            lines.append(f"命中技能: {decision.matched_skill}")
        if decision.skill_start_index >= 0:
            lines.append(
                f"技能覆盖区间: {decision.skill_start_index + 1} -> {decision.skill_end_index + 1}"
            )
        if decision.planner_execution_model:
            lines.append(f"执行模型: {decision.planner_execution_model}")
        for index, subtask in enumerate(decision.subtasks, 1):
            lines.append(
                f"子任务 {index}: {subtask.task_intent.value} -> {subtask.selected_tool}"
            )
        if decision.planner_hint:
            lines.append(f"执行提示: {decision.planner_hint}")
        return "\n".join(f"- {line}" for line in lines)
    lines = [
        f"任务识别: {decision.task_intent.value}",
        f"控制策略: {decision.control_mode.value}",
        f"置信度: {decision.confidence:.2f}",
        f"匹配方式: {decision.match_method}",
    ]
    if decision.selected_tool:
        lines.append(f"建议工具: {decision.selected_tool}")
    if decision.planner_hint:
        lines.append(f"执行提示: {decision.planner_hint}")
    return "\n".join(f"- {line}" for line in lines)


def _build_skill_policy_context(
    matched_skill: str,
    policy: SkillPolicy | None,
) -> str:
    if not matched_skill or policy is None:
        return ""
    lines = [f"- 命中技能: {matched_skill}"]
    if policy.allowed_tools:
        lines.append(f"- 允许工具: {', '.join(policy.allowed_tools)}")
    if policy.required_memory:
        lines.append(f"- 需参考记忆: {', '.join(policy.required_memory)}")
    lines.append(f"- 是否允许自主模式: {'是' if policy.allow_autonomous else '否'}")
    lines.append(f"- 最大工具步数: {policy.max_steps}")
    if policy.output_contract:
        lines.append(f"- 回答应包含: {', '.join(policy.output_contract)}")
    if policy.post_actions:
        lines.append(f"- 结束后动作: {', '.join(policy.post_actions)}")
    return "\n".join(lines)


def _planner_metadata(decision: PlannerDecision | None) -> dict[str, object]:
    if decision is None:
        return {}
    metadata = {
        "planner_task_intent": decision.task_intent.value,
        "planner_control_mode": decision.control_mode.value,
        "planner_selected_tool": decision.selected_tool,
        "planner_confidence": round(decision.confidence, 3),
        "planner_match_method": decision.match_method,
        "planner_is_composite": decision.is_composite,
        "planner_primary_intent": (
            decision.primary_intent.value if decision.primary_intent is not None else ""
        ),
        "planner_ordering_method": decision.ordering_method,
        "planner_matched_skill": decision.matched_skill,
        "planner_skill_start_index": decision.skill_start_index,
        "planner_skill_end_index": decision.skill_end_index,
        "planner_execution_model": decision.planner_execution_model,
    }
    if decision.subtasks:
        metadata["subtask_plan"] = _serialize_subtasks(decision.subtasks)
    return metadata


def _detect_specialized_tool_hint(message: str) -> tuple[str, str] | None:
    text = (message or "").strip()
    if not text:
        return None
    if _NETWORK_CALC_HINT_RE.search(text):
        return (
            "network_calc",
            "这是需要精确数值计算的网络题，优先考虑调用 network_calc，必要时再结合课程证据解释。",
        )
    if _PROTOCOL_SIM_HINT_RE.search(text):
        return (
            "protocol_state_simulator",
            "这是协议过程或故障推演题，优先考虑调用 protocol_state_simulator，再结合课程证据解释。",
        )
    return None


def _is_course_task(decision: PlannerDecision | None) -> bool:
    return decision is not None and decision.task_intent != TaskIntent.GENERAL_CHAT


def _append_warning(text: str, warning: str) -> str:
    if not warning or warning in text:
        return text
    return f"{text.rstrip()}\n\n{warning}"


def _ensure_direct_answer_has_citation(
    text: str,
    bundle: EvidenceBundle | None,
) -> str:
    cleaned = (text or "").strip()
    if not cleaned or bundle is None or not bundle.citations:
        return cleaned
    if extract_citation_indices(cleaned):
        return cleaned
    first_index = int(bundle.citations[0].get("index") or 1)
    if re.search(r"[。！？!?]$", cleaned):
        return f"{cleaned}[{first_index}]"
    return f"{cleaned} [{first_index}]"


def _tool_output_kind(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    return str(metadata.get("tool_output_kind", "") or "").strip().lower()


def _tool_result_is_final_answer(tool_name: str, metadata: dict[str, Any] | None) -> bool:
    kind = _tool_output_kind(metadata)
    if kind == "final_answer":
        return True
    if kind == "evidence_context":
        return False
    if tool_name == "knowledge_query":
        return False
    return bool((metadata or {}).get("final_response_preferred", False))


def _build_pacing_prompt(pacing_level: str, pacing_reason: str) -> str:
    guidance = ""
    if pacing_level == "accelerate":
        guidance = "建议：减少基础铺垫，直接推进到更高阶内容或更难的问题。"
    elif pacing_level == "decelerate":
        guidance = "建议：放慢速度，多举例，必要时先查前置知识依赖。"
    else:
        guidance = "建议：保持正常节奏，先完成当前讲解或练习。"
    return (
        "## [Adaptive Pacing]\n"
        f"当前学习节奏: {pacing_level}\n"
        f"依据: {pacing_reason}\n"
        f"{guidance}"
    )


def _parse_count_token(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        return int(token)
    return _CN_NUMBERS.get(token)


def _extract_quiz_count(text: str) -> int | None:
    match = _QUIZ_COUNT_RE.search(text)
    if match is None:
        return None
    return _parse_count_token(match.group("count"))


def _extract_question_type(text: str) -> str | None:
    for value, pattern in _QUESTION_TYPE_PATTERNS:
        if pattern.search(text):
            return value
    return None


def _extract_quiz_difficulty(text: str) -> int | None:
    if re.search(r"难度\s*1|简单|基础", text):
        return 1
    if re.search(r"难度\s*2", text):
        return 2
    if re.search(r"难度\s*3|中等", text):
        return 3
    if re.search(r"难度\s*4|困难", text):
        return 4
    if re.search(r"难度\s*5", text):
        return 5
    return None


def _clean_topic_text(text: str) -> str:
    cleaned = text
    for pattern in _TOPIC_CLEANUP_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，。！？；,!?;：:")
    return cleaned


def _extract_shared_topic(message: str) -> str:
    cleaned = _clean_topic_text(message)
    if cleaned:
        return cleaned
    stripped = re.sub(r"\s+", " ", message).strip()
    return stripped[:120]


def _extract_evaluator_fields(message: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, pattern in _EVALUATOR_FIELD_PATTERNS.items():
        match = pattern.search(message)
        if match is not None:
            values[key] = " ".join(match.group(1).split())
    return values


def _latest_user_message(conversation: Conversation) -> str:
    return next(
        (
            msg.content
            for msg in reversed(conversation.messages)
            if msg.role == "user" and msg.content
        ),
        "",
    )


def _extract_ingest_file_path(message: str) -> str:
    match = _DIRECT_INGEST_PATH_RE.search(message or "")
    if match is None:
        return ""
    return match.group(1).strip().strip("\"'，。！？；,!?;")


def _composite_section_title(intent: TaskIntent) -> str:
    return {
        TaskIntent.KNOWLEDGE_QUERY: "知识讲解",
        TaskIntent.REVIEW_SUMMARY: "复习总结",
        TaskIntent.QUIZ_GENERATOR: "练习题",
        TaskIntent.QUIZ_EVALUATOR: "判题结果",
    }.get(intent, intent.value)


def _subtask_within_skill_interval(
    planner_decision: PlannerDecision,
    subtask_index: int,
) -> bool:
    return (
        planner_decision.skill_start_index >= 0
        and planner_decision.skill_start_index <= subtask_index <= planner_decision.skill_end_index
    )


def _build_evidence_metadata(
    bundle: EvidenceBundle | None,
    assessment: GroundingAssessment | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if bundle is not None:
        metadata.update(bundle.to_metadata())
    if assessment is not None:
        metadata.update(assessment.to_metadata())
    return metadata


def _build_batch_evaluation_metadata(metadata: dict[str, Any] | None) -> dict[str, object]:
    if not metadata:
        return {}
    keys = (
        "batch_evaluation",
        "question_count",
        "batch_results",
        "alignment_mode",
        "alignment_status",
        "split_confidence",
        "score_aggregation",
    )
    return {
        key: metadata[key]
        for key in keys
        if key in metadata
    }


def _tool_completion_hint(tool_name: str, metadata: dict[str, Any] | None) -> str:
    if metadata:
        explicit = str(metadata.get("completion_hint", "") or "").strip().lower()
        if explicit in {"continue", "step_done", "wait_user", "clarify"}:
            return explicit
    return "step_done" if _tool_result_is_final_answer(tool_name, metadata) else "continue"


def _agenda_metadata(conversation: Conversation) -> dict[str, Any]:
    raw = conversation.metadata if isinstance(conversation.metadata, dict) else {}
    return raw


def _load_agenda_state(conversation: Conversation) -> AgendaState | None:
    raw = _agenda_metadata(conversation).get("agenda_state")
    if not isinstance(raw, dict):
        return None
    try:
        return AgendaState.model_validate(raw)
    except Exception:
        logger.warning("Failed to load agenda_state from conversation metadata", exc_info=True)
        return None


def _save_agenda_state(conversation: Conversation, agenda_state: AgendaState | None) -> None:
    metadata = dict(_agenda_metadata(conversation))
    if agenda_state is None:
        metadata.pop("agenda_state", None)
    else:
        metadata["agenda_state"] = agenda_state.model_dump(mode="json")
    conversation.metadata = metadata


def _agenda_goal_from_subtask(subtask: PlannedSubtask) -> AgendaGoal:
    return AgendaGoal(
        goal_id=subtask.goal_id or f"goal_{subtask.source_span[0]}_{subtask.source_span[1]}",
        intent=subtask.task_intent.value,
        selected_tool=subtask.selected_tool,
        segment_text=subtask.segment_text,
        required=subtask.required,
        depends_on_user_input=subtask.depends_on_user_input,
        status=GoalStatus(subtask.status),
        match_method=subtask.match_method,
        source_span=[subtask.source_span[0], subtask.source_span[1]],
    )


def _agenda_goal_from_decision(decision: PlannerDecision) -> AgendaGoal:
    return AgendaGoal(
        goal_id="goal_1",
        intent=decision.task_intent.value,
        selected_tool=decision.selected_tool or decision.task_intent.value,
        segment_text="",
        required=True,
        depends_on_user_input=(decision.task_intent == TaskIntent.QUIZ_EVALUATOR),
        status=GoalStatus.PENDING,
        match_method=decision.match_method,
        source_span=[],
    )


def _should_add_quiz_follow_up_goal(
    decision: PlannerDecision | None,
    matched_skill: str,
) -> bool:
    if decision is None:
        return False
    if decision.is_composite:
        if not decision.subtasks:
            return False
        last_subtask = decision.subtasks[-1]
        return last_subtask.task_intent == TaskIntent.QUIZ_GENERATOR
    if decision.task_intent != TaskIntent.QUIZ_GENERATOR:
        return False
    return matched_skill in {"quiz_drill", "error_review"}


def _build_agenda_state(
    planner_decision: PlannerDecision,
    *,
    matched_skill: str = "",
) -> AgendaState | None:
    goals: list[AgendaGoal]
    if planner_decision.is_composite and planner_decision.subtasks:
        goals = [_agenda_goal_from_subtask(subtask) for subtask in planner_decision.subtasks]
    else:
        goals = [_agenda_goal_from_decision(planner_decision)]
    if _should_add_quiz_follow_up_goal(planner_decision, matched_skill):
        next_goal_number = len(goals) + 1
        goals.append(
            AgendaGoal(
                goal_id=f"goal_{next_goal_number}",
                intent=TaskIntent.QUIZ_EVALUATOR.value,
                selected_tool=TaskIntent.QUIZ_EVALUATOR.value,
                segment_text="根据用户后续作答进行判题。",
                required=True,
                depends_on_user_input=True,
                status=GoalStatus.PENDING,
                match_method="skill_follow_up",
                source_span=[],
            )
        )
    if not goals:
        return None
    return AgendaState(
        request_status=RequestStatus.ACTIVE,
        current_goal_index=0,
        matched_skill=matched_skill or planner_decision.matched_skill,
        skill_start_index=planner_decision.skill_start_index,
        skill_end_index=planner_decision.skill_end_index,
        goals=goals,
        resume_payload={},
    )


def _agenda_needs_runtime(
    planner_decision: PlannerDecision | None,
    matched_skill: str,
) -> bool:
    if planner_decision is None:
        return False
    if planner_decision.is_composite:
        return True
    return _should_add_quiz_follow_up_goal(planner_decision, matched_skill)


def _active_goal(agenda_state: AgendaState | None) -> AgendaGoal | None:
    if agenda_state is None:
        return None
    if not (0 <= agenda_state.current_goal_index < len(agenda_state.goals)):
        return None
    return agenda_state.goals[agenda_state.current_goal_index]


def _set_goal_status(goal: AgendaGoal, status: GoalStatus, *, summary: str = "") -> None:
    goal.status = status
    if summary:
        goal.result_summary = summary


def _goal_statuses(agenda_state: AgendaState | None) -> list[dict[str, str]]:
    if agenda_state is None:
        return []
    return [
        {
            "goal_id": goal.goal_id,
            "intent": goal.intent,
            "selected_tool": goal.selected_tool,
            "status": goal.status.value if isinstance(goal.status, GoalStatus) else str(goal.status),
        }
        for goal in agenda_state.goals
    ]


def _goal_execution_model(agenda_state: AgendaState | None) -> str:
    if agenda_state is None:
        return ""
    return "agenda_react"


def _goal_has_resume_candidate(message: str, goal: AgendaGoal | None) -> bool:
    if goal is None:
        return False
    text = (message or "").strip()
    if not text:
        return False
    if goal.selected_tool == TaskIntent.QUIZ_EVALUATOR.value:
        if _WAITING_FOR_ANSWER_RE.search(text):
            return True
        if re.fullmatch(r"(?:[A-Da-d](?:[\s,，、/]+|$)){1,8}", text):
            return True
        if re.search(r"(?:^|\n)\s*\d+\s*[.、:：)]?\s*[A-Da-d]\b", text):
            return True
        return False
    if goal.selected_tool == TaskIntent.DOCUMENT_INGEST.value:
        return bool(_extract_ingest_file_path(text))
    return False


def _goal_requires_waiting(goal: AgendaGoal | None) -> bool:
    if goal is None:
        return False
    status = goal.status.value if isinstance(goal.status, GoalStatus) else str(goal.status)
    return goal.depends_on_user_input or status == GoalStatus.WAITING_USER.value


def _restore_skill_runtime(
    skill_workflow: object | None,
    matched_skill: str,
) -> tuple[str, SkillPolicy | None]:
    if not matched_skill or skill_workflow is None:
        return "", None
    registry = getattr(skill_workflow, "_registry", None)
    if registry is None:
        return "", None
    try:
        instruction = registry.load_instruction(matched_skill)
        policy = registry.load_policy(matched_skill)
    except Exception:
        logger.warning("Failed to restore skill runtime for '%s'", matched_skill, exc_info=True)
        return "", None
    skill_ctx = ""
    if instruction is not None and getattr(instruction, "raw_body", ""):
        skill_ctx = (
            f"你正在执行技能「{matched_skill}」。请严格按照以下步骤操作：\n\n"
            f"{instruction.raw_body}"
        )
    return skill_ctx, policy


def _planner_decision_from_agenda_state(agenda_state: AgendaState) -> PlannerDecision | None:
    if not agenda_state.goals:
        return None
    subtasks: list[PlannedSubtask] = []
    for index, goal in enumerate(agenda_state.goals):
        try:
            intent = TaskIntent(goal.intent)
        except ValueError:
            continue
        source_span = tuple(goal.source_span[:2]) if len(goal.source_span) >= 2 else (index, index + 1)
        subtasks.append(
            PlannedSubtask(
                goal_id=goal.goal_id,
                task_intent=intent,
                selected_tool=goal.selected_tool or intent.value,
                confidence=1.0,
                source_span=source_span,
                match_method=goal.match_method or "agenda_resume",
                segment_text=goal.segment_text,
                required=goal.required,
                depends_on_user_input=goal.depends_on_user_input,
                status=goal.status.value if isinstance(goal.status, GoalStatus) else str(goal.status),
            )
        )
    if not subtasks:
        return None
    primary_intent = subtasks[0].task_intent
    if agenda_state.matched_skill:
        control_mode = ControlMode.AUTONOMOUS
    elif primary_intent == TaskIntent.KNOWLEDGE_QUERY:
        control_mode = ControlMode.ADVISORY
    elif primary_intent == TaskIntent.GENERAL_CHAT:
        control_mode = ControlMode.PASS_THROUGH
    else:
        control_mode = ControlMode.FORCE_TOOL
    return PlannerDecision(
        task_intent=primary_intent,
        confidence=1.0,
        match_method="agenda_resume",
        control_mode=control_mode,
        selected_tool=subtasks[0].selected_tool,
        planner_hint="恢复上一轮未完成的多目标任务，继续按既定 agenda 推进。",
        is_composite=len(subtasks) > 1,
        subtasks=subtasks,
        primary_intent=primary_intent,
        ordering_method="agenda_resume_order",
        matched_skill=agenda_state.matched_skill,
        skill_start_index=agenda_state.skill_start_index,
        skill_end_index=agenda_state.skill_end_index,
        planner_execution_model="agenda_resume",
    )


def _should_resume_existing_agenda(
    message: str,
    agenda_state: AgendaState | None,
    planner_decision: PlannerDecision | None,
) -> bool:
    active_goal = _active_goal(agenda_state)
    if active_goal is None:
        return False
    if _goal_has_resume_candidate(message, active_goal):
        return True
    if planner_decision is None or planner_decision.is_composite:
        return False
    return planner_decision.task_intent.value == active_goal.intent


def _should_abandon_existing_agenda(
    agenda_state: AgendaState | None,
    planner_decision: PlannerDecision | None,
) -> bool:
    active_goal = _active_goal(agenda_state)
    if active_goal is None or planner_decision is None:
        return False
    if planner_decision.is_composite:
        return True
    if planner_decision.task_intent == TaskIntent.GENERAL_CHAT:
        return False
    return planner_decision.task_intent.value != active_goal.intent


def _build_goal_planner_decision(
    planner_decision: PlannerDecision,
    goal: AgendaGoal,
    *,
    goal_index: int,
    matched_skill: str,
    skill_policy: SkillPolicy | None,
) -> PlannerDecision:
    intent = TaskIntent(goal.intent)
    skill_active = (
        planner_decision.skill_start_index >= 0
        and planner_decision.skill_start_index <= goal_index <= planner_decision.skill_end_index
    )
    if intent == TaskIntent.GENERAL_CHAT:
        control_mode = ControlMode.PASS_THROUGH
    elif intent == TaskIntent.KNOWLEDGE_QUERY:
        control_mode = ControlMode.ADVISORY
    else:
        control_mode = ControlMode.FORCE_TOOL
    selected_tool = goal.selected_tool or intent.value
    planner_hint = planner_decision.planner_hint
    if skill_active and skill_policy is not None and skill_policy.allow_autonomous:
        control_mode = ControlMode.AUTONOMOUS
    return PlannerDecision(
        task_intent=intent,
        confidence=planner_decision.confidence,
        match_method=planner_decision.match_method,
        control_mode=control_mode,
        selected_tool=selected_tool,
        planner_hint=planner_hint,
        is_composite=False,
        primary_intent=intent,
        ordering_method="agenda_goal_order",
        matched_skill=matched_skill if skill_active else "",
        skill_start_index=planner_decision.skill_start_index,
        skill_end_index=planner_decision.skill_end_index,
        planner_execution_model="agenda_goal",
    )


def _accumulate_tool_calls(chunks: list[LlmStreamChunk]) -> list[ToolCallData] | None:
    """Reassemble streamed tool-call deltas into complete ToolCallData objects."""
    by_index: dict[int, dict] = {}
    for c in chunks:
        if not c.delta_tool_calls:
            continue
        for delta in c.delta_tool_calls:
            idx = delta.get("index", 0)
            entry = by_index.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if delta.get("id"):
                entry["id"] = delta["id"]
            fn = delta.get("function") or {}
            if fn.get("name"):
                entry["name"] = fn["name"]
            if fn.get("arguments"):
                entry["arguments"] += fn["arguments"]
    if not by_index:
        return None

    import json
    result: list[ToolCallData] = []
    for _, entry in sorted(by_index.items()):
        try:
            args = json.loads(entry["arguments"]) if entry["arguments"] else {}
        except json.JSONDecodeError:
            args = {"_raw": entry["arguments"]}
        result.append(ToolCallData(id=entry["id"], name=entry["name"], arguments=args))
    return result or None


class Agent:
    """Database-course learning agent with ReAct tool loop."""

    def __init__(
        self,
        *,
        llm_service: LlmService,
        tool_registry: ToolRegistry,
        conversation_store: ConversationStore,
        config: AgentConfig,
        prompt_builder: SystemPromptBuilder | None = None,
        lifecycle_hooks: list[LifecycleHook] | None = None,
        llm_middlewares: list[LlmMiddleware] | None = None,
        memory_enhancer: object | None = None,
        task_planner: TaskPlanner | None = None,
        skill_workflow: object | None = None,
        context_filter: object | None = None,
        review_hook: object | None = None,
        post_action_adapter: object | None = None,
        trace_enabled: bool = False,
        trace_collector: TraceCollector | None = None,
        grounding_mode: str = "balanced",
        grounding_low_evidence_threshold: float = 0.4,
    ) -> None:
        self.llm = llm_service
        self.tools = tool_registry
        self.conversations = conversation_store
        self.config = config
        self.prompt_builder = prompt_builder or SystemPromptBuilder(config.system_prompt_path)
        self.hooks = lifecycle_hooks or []
        self.middlewares = llm_middlewares or []
        self.memory_enhancer = memory_enhancer
        self.task_planner = task_planner
        self.skill_workflow = skill_workflow
        self.context_filter = context_filter
        self.review_hook = review_hook
        self.post_action_adapter = post_action_adapter
        self._bg_tasks: set[asyncio.Task] = set()
        self._trace_enabled = trace_enabled
        self._trace_collector = trace_collector or (TraceCollector() if trace_enabled else None)
        self._grounding_mode = grounding_mode
        self._grounding_evaluator = GroundingEvaluator(threshold=grounding_low_evidence_threshold)

    async def chat(
        self,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Process a user message through the ReAct tool loop, yielding stream events."""
        request_id = uuid.uuid4().hex[:12]
        agent_trace: TraceContext | None = None
        if self._trace_enabled and self._trace_collector is not None:
            agent_trace = TraceContext(trace_type="agent")
            agent_trace.metadata.update(
                {
                    "request_id": request_id,
                    "conversation_id": conversation_id or "",
                    "user_id_hash": _hash_user_id(user_id),
                    "message_preview": _truncate_text(message, limit=120),
                    "stream_enabled": self.config.stream_responses,
                    "max_tool_iterations": self.config.max_tool_iterations,
                    "status": "in_progress",
                }
            )

        # --- Load or create conversation ---
        conversation: Conversation | None = None
        try:
            load_started = time.monotonic()
            if conversation_id:
                conversation = await self.conversations.get(conversation_id, user_id)
            loaded_existing = conversation is not None
            if conversation is None:
                conversation = await self.conversations.create(user_id)
            if agent_trace is not None:
                agent_trace.metadata["conversation_id"] = conversation.id
                agent_trace.record_stage(
                    "conversation_load",
                    {
                        "loaded_existing": loaded_existing,
                        "message_count": len(conversation.messages),
                    },
                    elapsed_ms=(time.monotonic() - load_started) * 1000.0,
                )

            # --- Before-message hooks ---
            effective_message = message
            modified_count = 0
            hooks_started = time.monotonic()
            for hook in self.hooks:
                try:
                    modified = await hook.before_message(user_id, effective_message)
                    if modified is not None:
                        effective_message = modified
                        modified_count += 1
                except Exception:
                    logger.exception("before_message hook failed")
            if agent_trace is not None:
                agent_trace.record_stage(
                    "before_message_hooks",
                    {
                        "hook_count": len(self.hooks),
                        "modified_count": modified_count,
                        "effective_message_preview": _truncate_text(effective_message, limit=120),
                    },
                    elapsed_ms=(time.monotonic() - hooks_started) * 1000.0,
                )

            # --- Append user message ---
            conversation.messages.append(Message(role="user", content=effective_message))

            # --- Auto-generate title from first user message ---
            if not conversation.title:
                conversation.title = effective_message[:30].strip()
                if len(effective_message) > 30:
                    conversation.title += "..."

            # --- Build system prompt (P1: parallel preprocess) ---
            tool_schemas = self.tools.get_all_schemas()

            async def _fetch_memory() -> str:
                if self.memory_enhancer and hasattr(self.memory_enhancer, "get_memory_summary"):
                    try:
                        return await self.memory_enhancer.get_memory_summary(user_id)
                    except Exception:
                        logger.exception("Memory enhancer failed")
                return ""

            async def _fetch_review() -> str:
                if self.review_hook and hasattr(self.review_hook, "get_review_context"):
                    try:
                        try:
                            result = await self.review_hook.get_review_context(
                                user_id,
                                effective_message,
                            )
                        except TypeError:
                            result = await self.review_hook.get_review_context(user_id)
                        if isinstance(result, tuple) and len(result) == 2:
                            return result
                        metadata = {}
                        if hasattr(self.review_hook, "last_metadata"):
                            candidate = getattr(self.review_hook, "last_metadata")
                            if isinstance(candidate, dict):
                                metadata = dict(candidate)
                        return result or "", metadata
                    except Exception:
                        logger.exception("Review hook failed")
                return "", {}

            async def _fetch_skill() -> object | None:
                if self.skill_workflow and hasattr(self.skill_workflow, "try_handle"):
                    try:
                        return await self.skill_workflow.try_handle(effective_message, user_id)
                    except Exception:
                        logger.exception("Skill workflow handler failed")
                return None

            preprocess_started = time.monotonic()
            memory_ctx, review_payload, wf_result = await asyncio.gather(
                _fetch_memory(), _fetch_review(), _fetch_skill()
            )
            review_ctx = ""
            review_meta: dict[str, object] = {}
            if isinstance(review_payload, tuple) and len(review_payload) == 2:
                review_ctx = str(review_payload[0] or "")
                raw_meta = review_payload[1] or {}
                if isinstance(raw_meta, dict):
                    review_meta = dict(raw_meta)
            elif isinstance(review_payload, str):
                review_ctx = review_payload

            if review_ctx:
                memory_ctx = (memory_ctx + "\n\n" + review_ctx).strip() if memory_ctx else review_ctx
            preprocess_metadata: dict[str, object] = {
                "proactive_triggered": bool(review_meta.get("proactive_triggered", False)),
                "proactive_reason": str(review_meta.get("proactive_reason", "") or ""),
                "proactive_signals": dict(review_meta.get("proactive_signals", {}) or {}),
            }

            skill_ctx = ""
            direct_response = ""
            matched_skill = ""
            skill_policy: SkillPolicy | None = None
            stored_agenda_state = _load_agenda_state(conversation)
            agenda_state_to_run: AgendaState | None = None
            agenda_resumed = False
            agenda_abandoned = False
            if wf_result is not None:
                if hasattr(wf_result, "direct_response") and wf_result.direct_response:
                    direct_response = wf_result.direct_response
                if hasattr(wf_result, "skill_instruction") and wf_result.skill_instruction:
                    skill_ctx = wf_result.skill_instruction
                if hasattr(wf_result, "matched_skill") and wf_result.matched_skill:
                    matched_skill = wf_result.matched_skill
                if hasattr(wf_result, "skill_policy"):
                    skill_policy = wf_result.skill_policy

            planner_decision: PlannerDecision | None = None
            planner_context = ""
            course_task = False
            if self.task_planner is not None:
                try:
                    try:
                        planner_decision = self.task_planner.plan(
                            effective_message,
                            matched_skill=matched_skill or None,
                            skill_policy=skill_policy,
                        )
                    except TypeError:
                        planner_decision = self.task_planner.plan(
                            effective_message,
                            matched_skill=matched_skill or None,
                        )
                    planner_decision = self._apply_skill_policy(
                        planner_decision,
                        matched_skill=matched_skill,
                        skill_policy=skill_policy,
                    )
                    if (
                        matched_skill == ""
                        and planner_decision is not None
                        and planner_decision.task_intent == TaskIntent.KNOWLEDGE_QUERY
                    ):
                        specialized = _detect_specialized_tool_hint(effective_message)
                        if specialized is not None:
                            tool_name, hint = specialized
                            planner_decision.control_mode = ControlMode.AUTONOMOUS
                            planner_decision.selected_tool = tool_name
                            planner_decision.planner_hint = hint
                    if (
                        stored_agenda_state is not None
                        and stored_agenda_state.request_status in {
                            RequestStatus.WAITING_USER,
                            RequestStatus.CLARIFICATION_REQUIRED,
                        }
                    ):
                        if _should_resume_existing_agenda(
                            effective_message,
                            stored_agenda_state,
                            planner_decision,
                        ):
                            agenda_state_to_run = stored_agenda_state.model_copy(deep=True)
                            restored_decision = _planner_decision_from_agenda_state(agenda_state_to_run)
                            if restored_decision is not None:
                                planner_decision = restored_decision
                            if agenda_state_to_run.matched_skill and not matched_skill:
                                matched_skill = agenda_state_to_run.matched_skill
                            if agenda_state_to_run.matched_skill and skill_policy is None:
                                restored_skill_ctx, restored_skill_policy = _restore_skill_runtime(
                                    self.skill_workflow,
                                    agenda_state_to_run.matched_skill,
                                )
                                if restored_skill_ctx and not skill_ctx:
                                    skill_ctx = restored_skill_ctx
                                if restored_skill_policy is not None:
                                    skill_policy = restored_skill_policy
                            agenda_state_to_run.request_status = RequestStatus.ACTIVE
                            agenda_resumed = True
                        elif _should_abandon_existing_agenda(stored_agenda_state, planner_decision):
                            _save_agenda_state(conversation, None)
                            agenda_abandoned = True
                        else:
                            agenda_state_to_run = stored_agenda_state.model_copy(deep=True)
                            restored_decision = _planner_decision_from_agenda_state(agenda_state_to_run)
                            if restored_decision is not None:
                                planner_decision = restored_decision
                            if agenda_state_to_run.matched_skill and not matched_skill:
                                matched_skill = agenda_state_to_run.matched_skill
                            if agenda_state_to_run.matched_skill and skill_policy is None:
                                restored_skill_ctx, restored_skill_policy = _restore_skill_runtime(
                                    self.skill_workflow,
                                    agenda_state_to_run.matched_skill,
                                )
                                if restored_skill_ctx and not skill_ctx:
                                    skill_ctx = restored_skill_ctx
                                if restored_skill_policy is not None:
                                    skill_policy = restored_skill_policy
                    planner_context = _build_planner_context(planner_decision)
                    if skill_policy is not None and matched_skill:
                        skill_policy_context = _build_skill_policy_context(matched_skill, skill_policy)
                        if skill_policy_context:
                            planner_context = (
                                planner_context + "\n" + skill_policy_context
                                if planner_context
                                else skill_policy_context
                            )
                    course_task = _is_course_task(planner_decision)
                except Exception:
                    logger.exception("Task planner failed")

            planner_runtime: dict[str, object] = {
                "final_control_mode": (
                    planner_decision.control_mode.value
                    if planner_decision is not None
                    else ControlMode.PASS_THROUGH.value
                ),
                "violation_count": 0,
                "force_retry_count": 0,
                "knowledge_retry_count": 0,
            }
            if matched_skill:
                planner_runtime["matched_skill"] = matched_skill
            if skill_policy is not None:
                planner_runtime["skill_allowed_tools"] = list(skill_policy.allowed_tools)
                planner_runtime["skill_max_steps"] = min(int(skill_policy.max_steps), 5)
            if agenda_resumed:
                planner_runtime["agenda_resumed"] = True
            if agenda_abandoned:
                planner_runtime["agenda_abandoned"] = True

            if planner_decision is None or planner_decision.task_intent != TaskIntent.DOCUMENT_INGEST:
                tool_schemas = _exclude_tool_schema(tool_schemas, "document_ingest")

            if agent_trace is not None:
                agent_trace.record_stage(
                    "preprocess",
                    {
                        "tool_schema_count": len(tool_schemas),
                        "has_memory_context": bool(memory_ctx),
                        "has_skill_instruction": bool(skill_ctx),
                        "has_direct_response": bool(direct_response),
                        "proactive_triggered": bool(preprocess_metadata["proactive_triggered"]),
                    },
                    elapsed_ms=(time.monotonic() - preprocess_started) * 1000.0,
                )
                agent_trace.record_stage(
                    "proactive_review",
                    {
                        "triggered": bool(preprocess_metadata["proactive_triggered"]),
                        "reason": preprocess_metadata["proactive_reason"],
                        "signals": preprocess_metadata["proactive_signals"],
                    },
                )
                if planner_decision is not None:
                    agent_trace.record_stage("planner_decision", planner_decision.to_metadata())
                    agent_trace.metadata.update(_planner_metadata(planner_decision))
                if skill_policy is not None and matched_skill:
                    agent_trace.record_stage(
                        "skill_policy_applied",
                        {
                            "matched_skill": matched_skill,
                            "allowed_tools": list(skill_policy.allowed_tools),
                            "required_memory": list(skill_policy.required_memory),
                            "allow_autonomous": bool(skill_policy.allow_autonomous),
                            "max_steps": min(int(skill_policy.max_steps), 5),
                            "post_actions": list(skill_policy.post_actions),
                        },
                    )

            prompt_started = time.monotonic()
            prompt_tool_schemas = tool_schemas
            if skill_policy is not None and skill_policy.allowed_tools:
                prompt_tool_schemas = _restrict_tool_schemas_to_allowed(
                    tool_schemas,
                    set(skill_policy.allowed_tools),
                )
            system_prompt = self.prompt_builder.build(
                tool_schemas=prompt_tool_schemas,
                memory_context=memory_ctx,
                planner_context=planner_context,
                grounding_context=build_grounding_context(None, course_task=course_task),
                active_skill=skill_ctx,
            )
            if agent_trace is not None:
                agent_trace.record_stage(
                    "prompt_build",
                    {
                        "system_prompt_length": len(system_prompt),
                        "tool_schema_count": len(tool_schemas),
                    },
                    elapsed_ms=(time.monotonic() - prompt_started) * 1000.0,
                )

            if direct_response:
                if agent_trace is not None:
                    agent_trace.metadata["status"] = "success"
                    agent_trace.record_stage(
                        "final_response",
                        {
                            "direct_response": True,
                            "content_length": len(direct_response),
                            "content_preview": _truncate_text(direct_response),
                        },
                    )
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=direct_response)
                done_metadata = {
                    "conversation_id": conversation.id,
                    "title": conversation.title,
                }
                if agent_trace is not None:
                    done_metadata["trace_id"] = agent_trace.trace_id
                done_metadata.update(preprocess_metadata)
                done_metadata.update(_planner_metadata(planner_decision))
                done_metadata["planner_final_control_mode"] = planner_runtime["final_control_mode"]
                done_metadata["planner_violation_count"] = planner_runtime["violation_count"]
                yield StreamEvent(type=StreamEventType.DONE, metadata=done_metadata)
                return

            recent_msgs = [
                {"role": m.role, "content": m.content or ""}
                for m in conversation.messages[-6:]
                if m.role in ("user", "assistant") and m.content
            ]
            tool_metadata = {}
            if agent_trace is not None:
                tool_metadata["agent_trace_id"] = agent_trace.trace_id
            if planner_decision is not None:
                tool_metadata.update(_planner_metadata(planner_decision))
            if matched_skill:
                tool_metadata["matched_skill"] = matched_skill
            tool_ctx = ToolContext(
                user_id=user_id,
                conversation_id=conversation.id,
                request_id=request_id,
                metadata={
                    **tool_metadata,
                    "default_collection": self.config.default_collection,
                    "response_profile": self.config.response_profile,
                    "skill_allowed_tools": list(skill_policy.allowed_tools) if skill_policy is not None else [],
                    "skill_max_steps": min(int(skill_policy.max_steps), 5) if skill_policy is not None else 0,
                    "skill_post_actions": list(skill_policy.post_actions) if skill_policy is not None else [],
                },
                recent_messages=recent_msgs,
            )

            use_composite_runtime = False
            use_agenda_runtime = False
            response_state: dict[str, object] = dict(preprocess_metadata)
            if agenda_resumed:
                response_state["agenda_resumed"] = True
            if agenda_abandoned:
                response_state["agenda_abandoned"] = True
            try:
                if agenda_state_to_run is None and _agenda_needs_runtime(planner_decision, matched_skill):
                    agenda_state_to_run = _build_agenda_state(
                        planner_decision,
                        matched_skill=matched_skill,
                    )
                use_agenda_runtime = planner_decision is not None and agenda_state_to_run is not None
                use_composite_runtime = (
                    not use_agenda_runtime
                    and planner_decision is not None
                    and planner_decision.is_composite
                )
                if use_agenda_runtime and agenda_state_to_run is not None and planner_decision is not None:
                    async for event in self._run_agenda_plan(
                        conversation,
                        system_prompt,
                        tool_schemas,
                        tool_ctx,
                        planner_decision,
                        agenda_state_to_run,
                        matched_skill=matched_skill,
                        skill_policy=skill_policy,
                        trace=agent_trace,
                        response_state=response_state,
                    ):
                        yield event
                elif use_composite_runtime:
                    async for event in self._run_composite_plan(
                        conversation,
                        tool_ctx,
                        planner_decision,
                        trace=agent_trace,
                        response_state=response_state,
                    ):
                        yield event
                else:
                    async for event in self._tool_loop(
                        conversation,
                        system_prompt,
                        tool_schemas,
                        tool_ctx,
                        trace=agent_trace,
                        planner_decision=planner_decision,
                        planner_runtime=planner_runtime,
                        response_state=response_state,
                        skill_policy=skill_policy,
                    ):
                        yield event
            except Exception as exc:
                logger.exception("Agent execution error")
                if agent_trace is not None:
                    agent_trace.metadata["status"] = "error"
                    agent_trace.record_stage(
                        "error",
                        {
                            "phase": (
                                "agenda_execution"
                                if use_agenda_runtime
                                else "composite_execution"
                                if use_composite_runtime
                                else "tool_loop"
                            ),
                            "error": _truncate_text(str(exc)),
                        },
                    )
                yield StreamEvent(type=StreamEventType.ERROR, content=str(exc))

            response_state.setdefault(
                "effective_control_mode",
                str(planner_runtime.get("final_control_mode") or ""),
            )
            if matched_skill:
                response_state.setdefault("matched_skill", matched_skill)
            if skill_policy is not None:
                response_state.setdefault("skill_allowed_tools", list(skill_policy.allowed_tools))
            artifacts = await self._run_post_actions(
                conversation=conversation,
                user_id=user_id,
                matched_skill=matched_skill,
                skill_policy=skill_policy,
                response_state=response_state,
                trace=agent_trace,
            )
            if artifacts:
                response_state["artifacts"] = artifacts

            done_metadata = {
                "conversation_id": conversation.id,
                "title": conversation.title,
            }
            if agent_trace is not None:
                done_metadata["trace_id"] = agent_trace.trace_id
                agent_trace.metadata["planner_final_control_mode"] = planner_runtime["final_control_mode"]
                agent_trace.metadata["planner_violation_count"] = planner_runtime["violation_count"]
                agent_trace.metadata.update(response_state)
            done_metadata.update(_planner_metadata(planner_decision))
            done_metadata["planner_final_control_mode"] = planner_runtime["final_control_mode"]
            done_metadata["planner_violation_count"] = planner_runtime["violation_count"]
            done_metadata.update(response_state)
            yield StreamEvent(type=StreamEventType.DONE, metadata=done_metadata)

            task = asyncio.create_task(self._post_message_tasks(conversation))
            self._bg_tasks.add(task)
            task.add_done_callback(self._on_bg_task_done)
            if agent_trace is not None:
                agent_trace.record_stage(
                    "background_tasks_scheduled",
                    {"pending_tasks": len(self._bg_tasks)},
                )
        finally:
            if agent_trace is not None and self._trace_collector is not None:
                if agent_trace.metadata.get("status") == "in_progress":
                    agent_trace.metadata["status"] = "success"
                self._trace_collector.collect(agent_trace)

    def _on_bg_task_done(self, task: asyncio.Task) -> None:
        self._bg_tasks.discard(task)
        if not task.cancelled() and task.exception():
            logger.error("Background task failed: %s", task.exception())

    async def flush(self) -> None:
        """Await all pending background tasks — call during graceful shutdown."""
        if self._bg_tasks:
            await asyncio.gather(*self._bg_tasks, return_exceptions=True)
            self._bg_tasks.clear()

    async def _post_message_tasks(self, conversation: Conversation) -> None:
        """Background task: save conversation and run after-message hooks."""
        try:
            await self.conversations.update(conversation)
        except Exception:
            logger.exception("Failed to save conversation in background")
        for hook in self.hooks:
            try:
                await hook.after_message(conversation)
            except Exception:
                logger.exception("after_message hook failed in background")

    def _apply_skill_policy(
        self,
        planner_decision: PlannerDecision | None,
        *,
        matched_skill: str,
        skill_policy: SkillPolicy | None,
    ) -> PlannerDecision | None:
        if planner_decision is None or skill_policy is None:
            return planner_decision
        if planner_decision.task_intent == TaskIntent.DOCUMENT_INGEST:
            return planner_decision
        if planner_decision.is_composite:
            if planner_decision.skill_start_index >= 0 and matched_skill:
                planner_decision.matched_skill = matched_skill
                planner_decision.planner_hint = (
                    planner_decision.planner_hint
                    + f" 当前命中技能「{matched_skill}」，仅在覆盖区间内应用受控策略。"
                ).strip()
                if skill_policy.allow_autonomous:
                    planner_decision.control_mode = ControlMode.AUTONOMOUS
            return planner_decision
        if skill_policy.allowed_tools:
            selected = planner_decision.selected_tool
            if not selected or selected not in skill_policy.allowed_tools:
                planner_decision.selected_tool = skill_policy.allowed_tools[0]
        if skill_policy.allow_autonomous:
            planner_decision.control_mode = ControlMode.AUTONOMOUS
            planner_decision.planner_hint = (
                planner_decision.planner_hint
                + f" 当前命中技能「{matched_skill}」，请在允许工具范围内进行受控自主决策。"
            ).strip()
        return planner_decision

    async def _run_post_actions(
        self,
        *,
        conversation: Conversation,
        user_id: str,
        matched_skill: str,
        skill_policy: SkillPolicy | None,
        response_state: dict[str, object],
        trace: TraceContext | None = None,
    ) -> list[dict[str, str]]:
        if self.post_action_adapter is None or skill_policy is None or not skill_policy.post_actions:
            return []
        final_message = next(
            (
                msg.content or ""
                for msg in reversed(conversation.messages)
                if msg.role == "assistant"
            ),
            "",
        ).strip()
        if not final_message:
            return []
        tool_path = [str(value) for value in response_state.get("tool_path", []) if str(value)]
        try:
            artifacts = await self.post_action_adapter.run(
                user_id=user_id,
                conversation_id=conversation.id,
                matched_skill=matched_skill,
                post_actions=list(skill_policy.post_actions),
                final_text=final_message,
                tool_path=tool_path,
            )
        except Exception:
            logger.exception("post actions failed")
            if trace is not None:
                trace.record_stage(
                    "post_actions",
                    {"status": "error", "post_actions": list(skill_policy.post_actions)},
                )
            return []
        if trace is not None:
            trace.record_stage(
                "post_actions",
                {
                    "status": "success" if artifacts else "skipped",
                    "post_actions": list(skill_policy.post_actions),
                    "artifact_count": len(artifacts),
                },
            )
        return artifacts

    def _response_profile(self) -> str:
        value = str(getattr(self.config, "response_profile", "") or "").strip().lower()
        return value if value in {"balanced_fast", "quality_first"} else "quality_first"

    def _compute_pacing(self, conversation: Conversation) -> tuple[str, str]:
        return compute_pacing_from_conversation(conversation)

    def _assess_replan_signal(
        self,
        tool_name: str,
        tool_result: ToolResult,
    ) -> tuple[bool, str, str]:
        metadata = tool_result.metadata or {}
        if not tool_result.success:
            if tool_name == "concept_graph_query":
                return True, tool_result.error or "知识图谱查询失败", "broaden_topic"
            if tool_name == "protocol_state_simulator":
                return True, tool_result.error or "协议模拟器当前不支持该场景", "fallback_to_knowledge_query"
            if tool_name == "network_calc":
                return True, tool_result.error or "网络计算参数不足或不合法", "clarify_parameters"
            return True, tool_result.error or "工具执行失败", "try_alternative_tool"

        if tool_name == "knowledge_query":
            source_count = int(metadata.get("source_count", 0) or 0)
            result_text = (tool_result.result_for_llm or "").strip()
            if source_count < 2:
                if "未找到与查询相关的知识库内容" in result_text or source_count == 0:
                    return True, f"knowledge_query 仅返回 {source_count} 条可用证据", "broaden_query_then_switch_tool"
                return True, f"knowledge_query 仅返回 {source_count} 条证据，证据较弱", "switch_tool_then_common_knowledge"

        if tool_name == "concept_graph_query":
            rows = list(metadata.get("graph_rows", []) or [])
            if not rows:
                return True, "concept_graph_query 未返回图谱节点", "broaden_topic"

        return False, "", ""

    def _should_use_direct_tool_path(
        self,
        planner_decision: PlannerDecision | None,
        latest_message: str = "",
    ) -> bool:
        if self._response_profile() != "balanced_fast":
            return False
        if planner_decision is None or planner_decision.is_composite:
            return False
        if planner_decision.control_mode == ControlMode.AUTONOMOUS:
            return False
        if (
            planner_decision.task_intent == TaskIntent.KNOWLEDGE_QUERY
            and _detect_specialized_tool_hint(latest_message) is not None
        ):
            return False
        return planner_decision.task_intent in {
            TaskIntent.KNOWLEDGE_QUERY,
            TaskIntent.DOCUMENT_INGEST,
            TaskIntent.REVIEW_SUMMARY,
            TaskIntent.QUIZ_GENERATOR,
            TaskIntent.QUIZ_EVALUATOR,
        } and bool(planner_decision.selected_tool)

    def _build_direct_tool_call(
        self,
        conversation: Conversation,
        tool_ctx: ToolContext,
        planner_decision: PlannerDecision,
    ) -> ToolCallData | None:
        user_message = _latest_user_message(conversation)
        selected_tool = planner_decision.selected_tool or planner_decision.task_intent.value
        default_collection = str(
            tool_ctx.metadata.get("default_collection", "") or self.config.default_collection
        ).strip()
        if planner_decision.task_intent == TaskIntent.KNOWLEDGE_QUERY:
            if not user_message:
                return None
            return ToolCallData(
                id="direct_knowledge_query",
                name=selected_tool,
                arguments={
                    "query": user_message,
                    "top_k": 3 if self._response_profile() == "balanced_fast" else 5,
                    "collection": default_collection,
                },
            )
        if planner_decision.task_intent == TaskIntent.DOCUMENT_INGEST:
            file_path = _extract_ingest_file_path(user_message)
            if not file_path:
                return None
            return ToolCallData(
                id="direct_document_ingest",
                name=selected_tool,
                arguments={
                    "file_path": file_path,
                    "collection": default_collection,
                },
            )
        if planner_decision.task_intent == TaskIntent.REVIEW_SUMMARY:
            topic = _clean_topic_text(user_message) or _extract_shared_topic(user_message) or user_message
            if not topic:
                return None
            return ToolCallData(
                id="direct_review_summary",
                name=selected_tool,
                arguments={
                    "topic": topic,
                },
            )
        if planner_decision.task_intent == TaskIntent.QUIZ_GENERATOR:
            topic = _clean_topic_text(user_message) or _extract_shared_topic(user_message) or user_message
            if not topic:
                return None
            return ToolCallData(
                id="direct_quiz_generator",
                name=selected_tool,
                arguments={
                    "topic": topic,
                    "question_type": _extract_question_type(user_message) or "选择题",
                    "count": _extract_quiz_count(user_message) or 3,
                    "difficulty": _extract_quiz_difficulty(user_message) or 3,
                },
            )
        if planner_decision.task_intent == TaskIntent.QUIZ_EVALUATOR:
            return None
        return None

    @staticmethod
    def _recent_alignment_messages(conversation: Conversation) -> list[dict[str, Any]]:
        return [
            {"role": message.role, "content": message.content or ""}
            for message in conversation.messages[-8:]
            if message.role in {"user", "assistant"} and (message.content or "").strip()
        ]

    async def _build_quiz_evaluator_arguments(
        self,
        *,
        message: str,
        recent_messages: list[dict[str, Any]],
        quiz_bundle: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        alignment = await build_quiz_batch_alignment(
            message=message,
            recent_messages=recent_messages,
            quiz_bundle=quiz_bundle,
            llm_service=self.llm,
            max_items=5,
        )
        return {
            "items": [item.to_tool_payload() for item in alignment.items],
            "alignment_mode": alignment.alignment_mode,
            "alignment_status": alignment.alignment_status,
            "split_confidence": alignment.split_confidence,
            "clarification_reason": alignment.clarification_reason,
        }

    async def _build_direct_quiz_evaluator_call(
        self,
        conversation: Conversation,
        tool_ctx: ToolContext,
        planner_decision: PlannerDecision,
    ) -> ToolCallData | None:
        user_message = _latest_user_message(conversation)
        if not user_message:
            return None
        resume_payload = tool_ctx.metadata.get("agenda_resume_payload", {})
        quiz_bundle = None
        if isinstance(resume_payload, dict):
            candidate = resume_payload.get("quiz_bundle")
            if isinstance(candidate, list):
                quiz_bundle = candidate
        arguments = await self._build_quiz_evaluator_arguments(
            message=user_message,
            recent_messages=self._recent_alignment_messages(conversation),
            quiz_bundle=quiz_bundle,
        )
        return ToolCallData(
            id="direct_quiz_evaluator",
            name=planner_decision.selected_tool or planner_decision.task_intent.value,
            arguments=arguments,
        )

    async def _generate_direct_knowledge_answer(
        self,
        *,
        system_prompt: str,
        question: str,
        tool_result: ToolResult,
        evidence_bundle: EvidenceBundle | None,
        trace: TraceContext | None = None,
    ) -> tuple[str, GroundingAssessment | None]:
        response_profile = self._response_profile()
        max_tokens = 320 if response_profile == "balanced_fast" else self.config.max_tokens
        system_content = (
            "你是一名计算机网络课程助教。你已经拿到课程证据，请直接给出准确、简洁、可引用的最终回答。"
            if response_profile == "balanced_fast"
            else system_prompt + "\n\n## [Direct Tool Answer]\n你已经拿到课程证据，直接形成最终回答。"
        )
        prompt = (
            "你已经拿到了课程知识库证据，请直接给出最终回答。\n"
            "要求：\n"
            "1. 只依据给定证据回答，不要再次调用工具。\n"
            "2. 回答尽量简洁聚焦，优先直接回答问题。\n"
            "3. 在关键结论后保留 `[1]`、`[2]` 这类引用编号。\n"
            "4. 如果证据不足，请明确说明缺口，但不要空泛重复。\n\n"
            f"用户问题：{question}\n\n"
            f"课程证据：\n{tool_result.result_for_llm}"
        )
        request = LlmRequest(
            messages=[
                LlmMessage(role="system", content=system_content),
                LlmMessage(role="user", content=prompt),
            ],
            temperature=0.1 if response_profile == "balanced_fast" else 0.2,
            max_tokens=max_tokens,
            stream=False,
            metadata={
                "course_task": True,
                "skip_reflection_warning": True,
                "citations": evidence_bundle.citations if evidence_bundle is not None else [],
                "evidence_summary": (
                    evidence_bundle.evidence_summary if evidence_bundle is not None else ""
                ),
                "source_count": (
                    evidence_bundle.source_count if evidence_bundle is not None else 0
                ),
                "generation_mode": "direct_knowledge_query",
            },
        )
        iteration_started = time.monotonic()
        for mw in self.middlewares:
            try:
                request = await mw.before_llm_request(request)
            except Exception:
                logger.exception("before_llm_request middleware failed")
        response = await self.llm.send_request(request)
        for mw in reversed(self.middlewares):
            try:
                response = await mw.after_llm_response(request, response)
            except Exception:
                logger.exception("after_llm_response middleware failed")
        if trace is not None:
            trace.record_stage(
                "llm_iteration",
                {
                    "iteration": 1,
                    "input_message_count": len(request.messages),
                    "stream": False,
                    "tool_names": [],
                    "tool_call_count": 0,
                    "output_text_length": len(response.content or ""),
                    "had_error": bool(response.error),
                    "planner_control_mode": "direct_tool_path",
                    "planner_forced_tool": TaskIntent.KNOWLEDGE_QUERY.value,
                    "visible_tool_count": 0,
                    "has_evidence": bool(evidence_bundle and evidence_bundle.has_evidence),
                    "generation_mode": "direct_knowledge_query",
                },
                elapsed_ms=(time.monotonic() - iteration_started) * 1000.0,
            )
        if response.error:
            fallback_text = tool_result.result_for_llm or DEFAULT_LOW_EVIDENCE_MESSAGE
            return await self._finalize_answer(
                fallback_text,
                course_task=True,
                evidence_bundle=evidence_bundle,
                fallback_text=fallback_text,
            )
        content = (response.content or "").strip() or tool_result.result_for_llm or DEFAULT_LOW_EVIDENCE_MESSAGE
        content = _ensure_direct_answer_has_citation(content, evidence_bundle)
        return await self._finalize_answer(
            content,
            course_task=True,
            evidence_bundle=evidence_bundle,
            fallback_text=tool_result.result_for_llm,
        )

    async def _run_direct_tool_path(
        self,
        conversation: Conversation,
        system_prompt: str,
        tool_ctx: ToolContext,
        *,
        planner_decision: PlannerDecision,
        trace: TraceContext | None = None,
        response_state: dict[str, object] | None = None,
        goal_result: GoalExecutionResult | None = None,
        suppress_text_output: bool = False,
        suppress_assistant_append: bool = False,
    ) -> AsyncGenerator[StreamEvent, None]:
        if planner_decision.task_intent == TaskIntent.QUIZ_EVALUATOR:
            tool_call = await self._build_direct_quiz_evaluator_call(conversation, tool_ctx, planner_decision)
        else:
            tool_call = self._build_direct_tool_call(conversation, tool_ctx, planner_decision)
        if tool_call is None:
            return

        yield StreamEvent(
            type=StreamEventType.TOOL_START,
            tool_name=tool_call.name,
            metadata={"arguments": tool_call.arguments},
        )

        tool_blocked = False
        for hook in self.hooks:
            try:
                await hook.before_tool(tool_call.name, tool_ctx)
            except Exception as exc:
                logger.warning("before_tool hook blocked %s: %s", tool_call.name, exc)
                tool_blocked = True
                break

        tool_started = time.monotonic()
        if tool_blocked:
            tool_result_obj = ToolResult(
                success=False,
                error="Tool execution blocked by hook",
                metadata={
                    "tool_name": tool_call.name,
                    "error_type": ToolErrorType.BLOCKED_BY_HOOK.value,
                    "retryable": False,
                },
            )
        else:
            tool_result_obj = await self.tools.execute(
                tool_call,
                tool_ctx,
                timeout=float(self.config.tool_timeout),
            )

        for hook in self.hooks:
            try:
                modified = await hook.after_tool(tool_call.name, tool_result_obj, context=tool_ctx)
                if modified is not None:
                    tool_result_obj = modified
            except Exception:
                logger.exception("after_tool hook failed")

        yield StreamEvent(
            type=StreamEventType.TOOL_RESULT,
            tool_name=tool_call.name,
            content=tool_result_obj.result_for_llm if tool_result_obj.success else tool_result_obj.error,
            metadata={
                "success": tool_result_obj.success,
                **tool_result_obj.metadata,
            },
        )
        conversation.messages.append(
            Message(
                role="tool",
                content=tool_result_obj.result_for_llm if tool_result_obj.success else (tool_result_obj.error or ""),
                tool_call_id=tool_call.id,
                metadata=dict(tool_result_obj.metadata),
            )
        )

        if trace is not None:
            trace.record_stage(
                "tool_execution",
                {
                    "iteration": 0,
                    "tool_name": tool_call.name,
                    "arguments": _summarize_arguments(tool_call.arguments),
                    "success": tool_result_obj.success,
                    "error_type": tool_result_obj.metadata.get("error_type", ""),
                    "retryable": bool(tool_result_obj.metadata.get("retryable", False)),
                    "query_trace_id": tool_result_obj.metadata.get("query_trace_id"),
                    "generation_mode": tool_result_obj.metadata.get("generation_mode", ""),
                    "evaluation_mode": tool_result_obj.metadata.get("evaluation_mode", ""),
                    "source_count": int(tool_result_obj.metadata.get("source_count", 0) or 0),
                    "error_summary": _truncate_text(tool_result_obj.error),
                    "result_summary": _truncate_text(tool_result_obj.result_for_llm),
                    "direct_path": True,
                },
                elapsed_ms=(time.monotonic() - tool_started) * 1000.0,
            )

        if not tool_result_obj.success:
            if goal_result is not None:
                goal_result.error = tool_result_obj.error or f"{tool_call.name} 执行失败"
                goal_result.goal_status = GoalStatus.BLOCKED.value
                goal_result.completion_hint = "clarify"
                goal_result.metadata = dict(tool_result_obj.metadata)
            if trace is not None:
                trace.metadata["status"] = "error"
                trace.record_stage(
                    "error",
                    {
                        "phase": "direct_tool_execution",
                        "tool_name": tool_call.name,
                        "error": _truncate_text(tool_result_obj.error),
                    },
                )
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content=tool_result_obj.error or f"{tool_call.name} 执行失败",
            )
            return

        evidence_bundle = build_evidence_bundle(tool_call.name, tool_result_obj.metadata)
        final_text = tool_result_obj.result_for_llm or ""
        generation_mode = str(tool_result_obj.metadata.get("generation_mode", "") or "")
        evaluation_mode = str(tool_result_obj.metadata.get("evaluation_mode", "") or "")
        grounding_assessment: GroundingAssessment | None = None

        final_response_preferred = _tool_result_is_final_answer(tool_call.name, tool_result_obj.metadata)
        grounding_passthrough = bool(tool_result_obj.metadata.get("grounding_passthrough", False))
        tool_output_kind = _tool_output_kind(tool_result_obj.metadata)

        if (
            planner_decision.task_intent == TaskIntent.KNOWLEDGE_QUERY
            and tool_output_kind != "final_answer"
        ):
            final_text, grounding_assessment = await self._generate_direct_knowledge_answer(
                system_prompt=system_prompt,
                question=_latest_user_message(conversation),
                tool_result=tool_result_obj,
                evidence_bundle=evidence_bundle,
                trace=trace,
            )
            generation_mode = "direct_knowledge_query"
        elif generation_mode in {"question_bank", "rag_backed"}:
            final_text = tool_result_obj.result_for_llm or ""
            grounding_assessment = GroundingAssessment(
                score=1.0,
                has_evidence=bool(evidence_bundle and evidence_bundle.has_evidence),
                citation_count=0,
                source_count=evidence_bundle.source_count if evidence_bundle is not None else 0,
            )
        elif grounding_passthrough or final_response_preferred:
            grounding_assessment = self._grounding_evaluator.assess(final_text, evidence_bundle)
        else:
            final_text, grounding_assessment = await self._finalize_answer(
                final_text,
                course_task=_is_course_task(planner_decision),
                evidence_bundle=evidence_bundle,
                fallback_text=tool_result_obj.result_for_llm,
            )

        if trace is not None and grounding_assessment is not None:
            trace.record_stage(
                "answer_grounding",
                {
                    **grounding_assessment.to_metadata(),
                    "generation_mode": generation_mode,
                    "evaluation_mode": evaluation_mode,
                },
            )
            trace.record_stage(
                "final_response",
                {
                    "iteration": 1 if planner_decision.task_intent == TaskIntent.KNOWLEDGE_QUERY else 0,
                    "content_length": len(final_text or ""),
                    "content_preview": _truncate_text(final_text),
                    "generation_mode": generation_mode,
                    "evaluation_mode": evaluation_mode,
                    "direct_path": True,
                },
            )
        if response_state is not None:
            response_state.update(_build_evidence_metadata(evidence_bundle, grounding_assessment))
            response_state.update(_build_batch_evaluation_metadata(tool_result_obj.metadata))
            if generation_mode:
                response_state["generation_mode"] = generation_mode
            if evaluation_mode:
                response_state["evaluation_mode"] = evaluation_mode
            response_state.setdefault("tool_path", [tool_call.name])
            response_state.setdefault(
                "effective_control_mode",
                planner_decision.control_mode.value,
            )
        if goal_result is not None:
            goal_result.text = final_text or ""
            goal_result.metadata = dict(tool_result_obj.metadata)
            goal_result.resume_payload = dict(tool_result_obj.metadata.get("resume_payload", {}) or {})
            goal_result.completion_hint = _tool_completion_hint(tool_call.name, tool_result_obj.metadata)
            if goal_result.completion_hint == "wait_user":
                goal_result.goal_status = GoalStatus.WAITING_USER.value
            else:
                goal_result.goal_status = GoalStatus.COMPLETED.value
        if final_text and not suppress_text_output:
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=final_text)
        if not suppress_assistant_append:
            conversation.messages.append(Message(role="assistant", content=final_text or ""))

    def _build_composite_tool_args(
        self,
        subtask: PlannedSubtask,
        *,
        original_request: str,
        handoff: dict[str, Any],
    ) -> dict[str, Any]:
        segment_text = str(subtask.segment_text or original_request).strip()
        shared_topic = str(handoff.get("shared_topic") or _extract_shared_topic(original_request))
        segment_topic = _clean_topic_text(segment_text)
        effective_topic = segment_topic or shared_topic or original_request

        if subtask.task_intent == TaskIntent.KNOWLEDGE_QUERY:
            return {
                "query": segment_topic or shared_topic or original_request,
                "top_k": 5,
            }
        if subtask.task_intent == TaskIntent.REVIEW_SUMMARY:
            return {
                "topic": effective_topic,
            }
        if subtask.task_intent == TaskIntent.QUIZ_GENERATOR:
            return {
                "topic": effective_topic,
                "question_type": _extract_question_type(segment_text)
                or _extract_question_type(original_request)
                or "选择题",
                "count": _extract_quiz_count(segment_text)
                or _extract_quiz_count(original_request)
                or 3,
                "difficulty": _extract_quiz_difficulty(segment_text)
                or _extract_quiz_difficulty(original_request)
                or 3,
            }
        if subtask.task_intent == TaskIntent.QUIZ_EVALUATOR:
            parsed = _extract_evaluator_fields(segment_text or original_request)
            return {
                "question": parsed.get("question") or effective_topic or original_request,
                "user_answer": parsed.get("user_answer", ""),
                "correct_answer": parsed.get("correct_answer", ""),
                "question_type": _extract_question_type(segment_text)
                or _extract_question_type(original_request)
                or "选择题",
                "topic": effective_topic,
                "concepts": [],
            }
        return {}

    def _build_composite_tool_context(
        self,
        base_context: ToolContext,
        *,
        subtask: PlannedSubtask,
        subtask_index: int,
        handoff: dict[str, Any],
        planner_decision: PlannerDecision,
    ) -> ToolContext:
        aggregate = dict(handoff.get("aggregate_evidence", {}) or {})
        handoff_snapshot = {
            "original_user_request": handoff.get("original_user_request", ""),
            "shared_topic": handoff.get("shared_topic", ""),
            "completed_subtasks": list(handoff.get("completed_subtasks", [])),
            "latest_result_text": handoff.get("latest_result_text", ""),
            "latest_evidence_summary": handoff.get("latest_evidence_summary", ""),
            "latest_citations": list(handoff.get("latest_citations", [])),
            "aggregate_evidence": {
                "citations": list(aggregate.get("citations", [])),
                "evidence_summary": aggregate.get("evidence_summary", ""),
                "evidence_texts": list(aggregate.get("evidence_texts", [])),
                "source_count": aggregate.get("source_count", 0),
                "query_trace_ids": list(aggregate.get("query_trace_ids", [])),
            },
        }
        metadata = dict(base_context.metadata)
        metadata.update(
            {
                "planner_task_intent": subtask.task_intent.value,
                "composite": True,
                "composite_mode": True,
                "composite_handoff": handoff_snapshot,
                "composite_subtask_index": subtask_index,
                "composite_subtask_intent": subtask.task_intent.value,
                "composite_parent_request_id": base_context.request_id,
            }
        )
        skill_active = _subtask_within_skill_interval(planner_decision, subtask_index)
        metadata["composite_skill_active"] = skill_active
        metadata["planner_skill_start_index"] = planner_decision.skill_start_index
        metadata["planner_skill_end_index"] = planner_decision.skill_end_index
        if skill_active:
            if planner_decision.matched_skill:
                metadata["matched_skill"] = planner_decision.matched_skill
        else:
            metadata["matched_skill"] = ""
            metadata["skill_allowed_tools"] = []
            metadata["skill_max_steps"] = 0
            metadata["skill_post_actions"] = []
        return base_context.model_copy(update={"metadata": metadata})

    def _build_agenda_tool_context(
        self,
        base_context: ToolContext,
        *,
        agenda_state: AgendaState,
        goal: AgendaGoal,
        goal_index: int,
        handoff: dict[str, Any],
        planner_decision: PlannerDecision,
        next_goal: AgendaGoal | None,
    ) -> ToolContext:
        aggregate = dict(handoff.get("aggregate_evidence", {}) or {})
        handoff_snapshot = {
            "original_user_request": handoff.get("original_user_request", ""),
            "shared_topic": handoff.get("shared_topic", ""),
            "completed_subtasks": list(handoff.get("completed_subtasks", [])),
            "latest_result_text": handoff.get("latest_result_text", ""),
            "latest_evidence_summary": handoff.get("latest_evidence_summary", ""),
            "latest_citations": list(handoff.get("latest_citations", [])),
            "aggregate_evidence": {
                "citations": list(aggregate.get("citations", [])),
                "evidence_summary": aggregate.get("evidence_summary", ""),
                "evidence_texts": list(aggregate.get("evidence_texts", [])),
                "source_count": aggregate.get("source_count", 0),
                "query_trace_ids": list(aggregate.get("query_trace_ids", [])),
            },
        }
        metadata = dict(base_context.metadata)
        metadata.update(
            {
                "planner_task_intent": goal.intent,
                "agenda_mode": True,
                "agenda_goal_id": goal.goal_id,
                "agenda_goal_index": goal_index,
                "agenda_request_status": agenda_state.request_status.value,
                "agenda_resume_payload": dict(agenda_state.resume_payload),
                "agenda_next_goal_depends_on_user_input": bool(
                    next_goal is not None and next_goal.depends_on_user_input
                ),
                "composite": True,
                "composite_mode": True,
                "composite_handoff": handoff_snapshot,
                "composite_subtask_index": goal_index,
                "composite_subtask_intent": goal.intent,
                "composite_parent_request_id": base_context.request_id,
            }
        )
        skill_active = (
            agenda_state.skill_start_index >= 0
            and agenda_state.skill_start_index <= goal_index <= agenda_state.skill_end_index
        )
        metadata["planner_skill_start_index"] = agenda_state.skill_start_index
        metadata["planner_skill_end_index"] = agenda_state.skill_end_index
        metadata["composite_skill_active"] = skill_active
        if skill_active and agenda_state.matched_skill:
            metadata["matched_skill"] = agenda_state.matched_skill
        else:
            metadata["matched_skill"] = ""
            metadata["skill_allowed_tools"] = []
            metadata["skill_max_steps"] = 0
            metadata["skill_post_actions"] = []
        return base_context.model_copy(update={"metadata": metadata})

    @staticmethod
    def _agenda_waiting_prompt(goal: AgendaGoal | None) -> str:
        if goal is None:
            return "当前任务还缺少后续输入，请继续补充信息。"
        if goal.selected_tool == TaskIntent.QUIZ_EVALUATOR.value:
            return "请直接回复你的作答内容，我会继续逐题批改。"
        return "请补充下一步需要的输入，我会继续完成剩余目标。"

    async def _run_agenda_plan(
        self,
        conversation: Conversation,
        system_prompt: str,
        tool_schemas: list[dict],
        tool_ctx: ToolContext,
        planner_decision: PlannerDecision,
        agenda_state: AgendaState,
        *,
        matched_skill: str,
        skill_policy: SkillPolicy | None,
        trace: TraceContext | None = None,
        response_state: dict[str, object] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        original_request = next(
            (
                msg.content
                for msg in reversed(conversation.messages)
                if msg.role == "user" and msg.content
            ),
            "",
        )
        handoff: dict[str, Any] = {
            "original_user_request": original_request,
            "shared_topic": _extract_shared_topic(original_request),
            "completed_subtasks": [],
            "latest_result_text": "",
            "latest_evidence_summary": "",
            "latest_citations": [],
            "aggregate_evidence": {
                "citations": [],
                "evidence_summary": "",
                "evidence_texts": [],
                "source_count": 0,
                "query_trace_ids": [],
            },
        }
        completed_sections: list[tuple[str, str]] = []
        completed_subtasks: list[dict[str, object]] = []
        last_generation_mode = ""
        last_evaluation_mode = ""
        last_batch_metadata: dict[str, object] = {}
        waiting_prompt = ""
        active_goal = _active_goal(agenda_state)
        latest_user_message = _latest_user_message(conversation)

        if trace is not None:
            trace.record_stage(
                "agenda_plan",
                {
                    "goal_count": len(agenda_state.goals),
                    "current_goal_index": agenda_state.current_goal_index,
                    "matched_skill": agenda_state.matched_skill,
                },
            )

        for goal_index in range(agenda_state.current_goal_index, len(agenda_state.goals)):
            goal = agenda_state.goals[goal_index]
            if goal.status == GoalStatus.COMPLETED:
                continue
            agenda_state.current_goal_index = goal_index
            next_goal = agenda_state.goals[goal_index + 1] if goal_index + 1 < len(agenda_state.goals) else None
            if goal.depends_on_user_input and not _goal_has_resume_candidate(latest_user_message, goal):
                goal.status = GoalStatus.WAITING_USER
                agenda_state.request_status = RequestStatus.WAITING_USER
                waiting_prompt = self._agenda_waiting_prompt(goal)
                _save_agenda_state(conversation, agenda_state)
                break

            goal.status = GoalStatus.ACTIVE
            goal_decision = _build_goal_planner_decision(
                planner_decision,
                goal,
                goal_index=goal_index,
                matched_skill=matched_skill,
                skill_policy=skill_policy,
            )
            goal_ctx = self._build_agenda_tool_context(
                tool_ctx,
                agenda_state=agenda_state,
                goal=goal,
                goal_index=goal_index,
                handoff=handoff,
                planner_decision=planner_decision,
                next_goal=next_goal,
            )
            goal_result = GoalExecutionResult()
            goal_response_state: dict[str, object] = {}
            async for event in self._tool_loop(
                conversation,
                system_prompt,
                tool_schemas,
                goal_ctx,
                trace=trace,
                planner_decision=goal_decision,
                planner_runtime={
                    "final_control_mode": goal_decision.control_mode.value,
                    "violation_count": 0,
                    "force_retry_count": 0,
                    "knowledge_retry_count": 0,
                },
                response_state=goal_response_state,
                skill_policy=skill_policy if goal_decision.matched_skill else None,
                goal_result=goal_result,
                suppress_text_output=True,
                suppress_assistant_append=True,
            ):
                if event.type in {
                    StreamEventType.TOOL_START,
                    StreamEventType.TOOL_RESULT,
                    StreamEventType.ERROR,
                }:
                    yield event

            if response_state is not None and goal_response_state:
                tool_path = list(response_state.get("tool_path", []))
                for tool_name in goal_response_state.get("tool_path", []):
                    if tool_name not in tool_path:
                        tool_path.append(tool_name)
                if tool_path:
                    response_state["tool_path"] = tool_path
                for key in (
                    "generation_mode",
                    "evaluation_mode",
                    "batch_evaluation",
                    "question_count",
                    "batch_results",
                    "alignment_mode",
                    "alignment_status",
                    "split_confidence",
                    "score_aggregation",
                    "pacing_level",
                    "pacing_reason",
                    "replan_triggered",
                    "replan_reason",
                    "replan_count",
                    "fallback_strategy",
                ):
                    if key in goal_response_state:
                        response_state[key] = goal_response_state[key]

            if goal_result.error:
                goal.status = GoalStatus.BLOCKED
                agenda_state.request_status = RequestStatus.FAILED
                _save_agenda_state(conversation, agenda_state)
                waiting_prompt = goal_result.error
                break

            if goal_result.text:
                completed_sections.append((_composite_section_title(TaskIntent(goal.intent)), goal_result.text))

            citations = list(goal_result.metadata.get("citations", []) or [])
            evidence_texts = list(goal_result.metadata.get("evidence_texts", []) or [])
            query_trace_ids = [
                str(value)
                for value in goal_result.metadata.get("query_trace_ids", [])
                if str(value)
            ]
            query_trace_id = str(goal_result.metadata.get("query_trace_id", "") or "")
            if query_trace_id and query_trace_id not in query_trace_ids:
                query_trace_ids.append(query_trace_id)
            aggregate = handoff["aggregate_evidence"]
            aggregate["citations"].extend(citations)
            aggregate["evidence_texts"].extend(evidence_texts)
            seen_query_trace_ids = set(aggregate["query_trace_ids"])
            for trace_id in query_trace_ids:
                if trace_id not in seen_query_trace_ids:
                    aggregate["query_trace_ids"].append(trace_id)
                    seen_query_trace_ids.add(trace_id)
            aggregate["source_count"] = len(aggregate["citations"])
            aggregate["evidence_summary"] = (
                build_evidence_summary(aggregate["citations"])
                if aggregate["citations"]
                else str(goal_result.metadata.get("evidence_summary", "") or "")
            )
            handoff["latest_result_text"] = goal_result.text
            handoff["latest_evidence_summary"] = str(goal_result.metadata.get("evidence_summary", "") or "")
            handoff["latest_citations"] = citations
            completed_subtasks.append(
                {
                    "goal_id": goal.goal_id,
                    "task_intent": goal.intent,
                    "selected_tool": goal.selected_tool,
                }
            )
            handoff["completed_subtasks"] = list(completed_subtasks)
            if goal_result.metadata.get("generation_mode"):
                last_generation_mode = str(goal_result.metadata.get("generation_mode") or "")
            if goal_result.metadata.get("evaluation_mode"):
                last_evaluation_mode = str(goal_result.metadata.get("evaluation_mode") or "")
            batch_metadata = _build_batch_evaluation_metadata(goal_result.metadata)
            if batch_metadata:
                last_batch_metadata = batch_metadata

            completion_hint = goal_result.completion_hint or "step_done"
            if completion_hint == "clarify":
                goal.status = GoalStatus.BLOCKED
                agenda_state.request_status = RequestStatus.CLARIFICATION_REQUIRED
                agenda_state.resume_payload = dict(goal_result.resume_payload)
                _save_agenda_state(conversation, agenda_state)
                waiting_prompt = goal_result.text or self._agenda_waiting_prompt(goal)
                break

            goal.status = GoalStatus.COMPLETED
            goal.result_summary = _truncate_text(goal_result.text, limit=160)

            if completion_hint == "wait_user":
                agenda_state.resume_payload = dict(goal_result.resume_payload)
                if next_goal is not None:
                    next_goal.status = GoalStatus.WAITING_USER
                    agenda_state.current_goal_index = goal_index + 1
                    waiting_prompt = self._agenda_waiting_prompt(next_goal)
                else:
                    goal.status = GoalStatus.WAITING_USER
                    agenda_state.current_goal_index = goal_index
                    waiting_prompt = self._agenda_waiting_prompt(goal)
                agenda_state.request_status = RequestStatus.WAITING_USER
                _save_agenda_state(conversation, agenda_state)
                break

        else:
            agenda_state.request_status = RequestStatus.COMPLETED
            _save_agenda_state(conversation, None)

        lines: list[str] = []
        for section_title, section_text in completed_sections:
            lines.append(f"## {section_title}")
            lines.append(section_text.strip())
            lines.append("")
        if waiting_prompt:
            lines.append(waiting_prompt.strip())
        final_text = "\n".join(line for line in lines if line is not None).strip()
        if not final_text:
            final_text = "当前多目标任务未生成可展示结果。"

        if response_state is not None:
            response_state.update(
                {
                    "agenda_mode": True,
                    "request_status": agenda_state.request_status.value,
                    "current_goal_index": agenda_state.current_goal_index,
                    "goal_statuses": _goal_statuses(agenda_state),
                    "agenda_execution_model": _goal_execution_model(agenda_state),
                    "completed_subtasks": list(completed_subtasks),
                }
            )
            if last_generation_mode:
                response_state["generation_mode"] = last_generation_mode
            if last_evaluation_mode:
                response_state["evaluation_mode"] = last_evaluation_mode
            if last_batch_metadata:
                response_state.update(last_batch_metadata)

        yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=final_text)
        conversation.messages.append(Message(role="assistant", content=final_text))

    async def _run_composite_plan(
        self,
        conversation: Conversation,
        tool_ctx: ToolContext,
        planner_decision: PlannerDecision,
        *,
        trace: TraceContext | None = None,
        response_state: dict[str, object] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        original_request = next(
            (
                msg.content
                for msg in reversed(conversation.messages)
                if msg.role == "user" and msg.content
            ),
            "",
        )
        handoff: dict[str, Any] = {
            "original_user_request": original_request,
            "shared_topic": _extract_shared_topic(original_request),
            "completed_subtasks": [],
            "latest_result_text": "",
            "latest_evidence_summary": "",
            "latest_citations": [],
            "aggregate_evidence": {
                "citations": [],
                "evidence_summary": "",
                "evidence_texts": [],
                "source_count": 0,
                "query_trace_ids": [],
            },
        }
        completed_sections: list[tuple[str, str]] = []
        completed_subtasks: list[dict[str, object]] = []
        failed_subtask: dict[str, object] | None = None
        completed_intents: set[TaskIntent] = set()
        last_generation_mode = ""
        last_evaluation_mode = ""
        last_batch_metadata: dict[str, object] = {}

        if trace is not None:
            trace.record_stage(
                "composite_plan",
                {
                    "subtask_count": len(planner_decision.subtasks),
                    "ordering_method": planner_decision.ordering_method,
                    "subtask_plan": _serialize_subtasks(planner_decision.subtasks),
                },
            )

        for subtask_index, subtask in enumerate(planner_decision.subtasks):
            if subtask.task_intent == TaskIntent.QUIZ_EVALUATOR:
                recent_messages = list(tool_ctx.recent_messages)
                latest_result_text = str(handoff.get("latest_result_text", "") or "").strip()
                if latest_result_text:
                    recent_messages.append({"role": "assistant", "content": latest_result_text})
                segment_text = str(subtask.segment_text or original_request).strip() or original_request
                arguments = await self._build_quiz_evaluator_arguments(
                    message=segment_text,
                    recent_messages=recent_messages,
                )
            else:
                arguments = self._build_composite_tool_args(
                    subtask,
                    original_request=original_request,
                    handoff=handoff,
                )
            sub_ctx = self._build_composite_tool_context(
                tool_ctx,
                subtask=subtask,
                subtask_index=subtask_index,
                handoff=handoff,
                planner_decision=planner_decision,
            )
            tool_call = ToolCallData(
                id=f"composite_{subtask_index + 1}",
                name=subtask.selected_tool,
                arguments=arguments,
            )

            yield StreamEvent(
                type=StreamEventType.TOOL_START,
                tool_name=subtask.selected_tool,
                metadata={"arguments": arguments},
            )

            tool_blocked = False
            for hook in self.hooks:
                try:
                    await hook.before_tool(subtask.selected_tool, sub_ctx)
                except Exception as exc:
                    logger.warning("before_tool hook blocked %s: %s", subtask.selected_tool, exc)
                    tool_blocked = True
                    break

            tool_started = time.monotonic()
            if tool_blocked:
                tool_result_obj = ToolResult(
                    success=False,
                    error="Tool execution blocked by hook",
                    metadata={
                        "tool_name": subtask.selected_tool,
                        "error_type": ToolErrorType.BLOCKED_BY_HOOK.value,
                        "retryable": False,
                    },
                )
            else:
                tool_result_obj = await self.tools.execute(
                    tool_call,
                    sub_ctx,
                    timeout=float(self.config.tool_timeout),
                )

            for hook in self.hooks:
                try:
                    modified = await hook.after_tool(
                        subtask.selected_tool,
                        tool_result_obj,
                        context=sub_ctx,
                    )
                    if modified is not None:
                        tool_result_obj = modified
                except Exception:
                    logger.exception("after_tool hook failed")

            yield StreamEvent(
                type=StreamEventType.TOOL_RESULT,
                tool_name=subtask.selected_tool,
                content=(
                    tool_result_obj.result_for_llm
                    if tool_result_obj.success
                    else tool_result_obj.error
                ),
                metadata={
                    "success": tool_result_obj.success,
                    **tool_result_obj.metadata,
                },
            )
            conversation.messages.append(
                Message(
                    role="tool",
                    content=tool_result_obj.result_for_llm if tool_result_obj.success else (tool_result_obj.error or ""),
                    tool_call_id=tool_call.id,
                    metadata=dict(tool_result_obj.metadata),
                )
            )

            if trace is not None:
                trace.record_stage(
                    "composite_subtask_execution",
                    {
                        "subtask_index": subtask_index,
                        "task_intent": subtask.task_intent.value,
                        "tool_name": subtask.selected_tool,
                        "arguments": _summarize_arguments(arguments),
                        "success": tool_result_obj.success,
                        "error_summary": _truncate_text(tool_result_obj.error),
                        "result_summary": _truncate_text(tool_result_obj.result_for_llm),
                        "query_trace_id": tool_result_obj.metadata.get("query_trace_id", ""),
                    },
                    elapsed_ms=(time.monotonic() - tool_started) * 1000.0,
                )

            if not tool_result_obj.success:
                failed_subtask = {
                    "subtask_index": subtask_index,
                    "task_intent": subtask.task_intent.value,
                    "selected_tool": subtask.selected_tool,
                    "error": tool_result_obj.error or "子任务执行失败",
                }
                break

            completed_intents.add(subtask.task_intent)
            completed_subtask = {
                "subtask_index": subtask_index,
                "task_intent": subtask.task_intent.value,
                "selected_tool": subtask.selected_tool,
            }
            completed_subtasks.append(completed_subtask)
            completed_sections.append(
                (
                    _composite_section_title(subtask.task_intent),
                    tool_result_obj.result_for_llm or "",
                )
            )

            citations = list(tool_result_obj.metadata.get("citations", []) or [])
            evidence_texts = list(tool_result_obj.metadata.get("evidence_texts", []) or [])
            query_trace_ids = [
                str(value)
                for value in tool_result_obj.metadata.get("query_trace_ids", [])
                if str(value)
            ]
            query_trace_id = str(tool_result_obj.metadata.get("query_trace_id", "") or "")
            if query_trace_id and query_trace_id not in query_trace_ids:
                query_trace_ids.append(query_trace_id)

            aggregate = handoff["aggregate_evidence"]
            aggregate["citations"].extend(citations)
            aggregate["evidence_texts"].extend(evidence_texts)
            seen_query_trace_ids = set(aggregate["query_trace_ids"])
            for trace_id in query_trace_ids:
                if trace_id not in seen_query_trace_ids:
                    aggregate["query_trace_ids"].append(trace_id)
                    seen_query_trace_ids.add(trace_id)
            aggregate["source_count"] = len(aggregate["citations"])
            aggregate["evidence_summary"] = (
                build_evidence_summary(aggregate["citations"])
                if aggregate["citations"]
                else str(tool_result_obj.metadata.get("evidence_summary", "") or "")
            )

            handoff["completed_subtasks"] = list(completed_subtasks)
            handoff["latest_result_text"] = tool_result_obj.result_for_llm or ""
            handoff["latest_evidence_summary"] = str(
                tool_result_obj.metadata.get("evidence_summary", "") or ""
            )
            handoff["latest_citations"] = citations
            last_generation_mode = str(
                tool_result_obj.metadata.get("generation_mode", "") or last_generation_mode
            )
            last_evaluation_mode = str(
                tool_result_obj.metadata.get("evaluation_mode", "") or last_evaluation_mode
            )
            batch_metadata = _build_batch_evaluation_metadata(tool_result_obj.metadata)
            if batch_metadata:
                last_batch_metadata = batch_metadata

        lines: list[str] = []
        for section_title, section_text in completed_sections:
            lines.append(f"## {section_title}")
            lines.append(section_text.strip())
            lines.append("")
        if failed_subtask is not None:
            lines.append("## 未完成项")
            lines.append(
                f"- 子任务 `{failed_subtask['selected_tool']}` 执行失败：{failed_subtask['error']}"
            )
        final_text = "\n".join(line for line in lines if line is not None).strip()
        if not final_text:
            final_text = "当前复合学习任务未生成可展示结果。"

        aggregate_metadata = {
            "grounding_capable": True,
            "citations": list(handoff["aggregate_evidence"]["citations"]),
            "evidence_summary": handoff["aggregate_evidence"]["evidence_summary"],
            "evidence_texts": list(handoff["aggregate_evidence"]["evidence_texts"]),
            "source_count": int(handoff["aggregate_evidence"]["source_count"] or 0),
            "query_trace_ids": list(handoff["aggregate_evidence"]["query_trace_ids"]),
            "generation_mode": "composite",
            "evaluation_mode": last_evaluation_mode,
        }
        aggregate_bundle = build_evidence_bundle("composite", aggregate_metadata)
        grounding_assessment: GroundingAssessment | None = None
        if (
            aggregate_bundle is not None
            and aggregate_bundle.has_evidence
            and TaskIntent.QUIZ_GENERATOR not in completed_intents
            and TaskIntent.QUIZ_EVALUATOR not in completed_intents
            and failed_subtask is None
        ):
            final_text, grounding_assessment = await self._finalize_answer(
                final_text,
                course_task=True,
                evidence_bundle=aggregate_bundle,
                fallback_text=final_text,
            )
            if trace is not None and grounding_assessment is not None:
                trace.record_stage(
                    "answer_grounding",
                    {
                        **grounding_assessment.to_metadata(),
                        "generation_mode": "composite",
                    },
                )

        if response_state is not None:
            response_state.update(
                {
                    "composite": True,
                    "planner_primary_intent": (
                        planner_decision.primary_intent.value
                        if planner_decision.primary_intent is not None
                        else planner_decision.task_intent.value
                    ),
                    "subtask_plan": _serialize_subtasks(planner_decision.subtasks),
                    "completed_subtasks": list(completed_subtasks),
                    "failed_subtask": failed_subtask or {},
                    "generation_mode": "composite",
                }
            )
            response_state.update(_build_evidence_metadata(aggregate_bundle, grounding_assessment))
            response_state.update(last_batch_metadata)
            if last_evaluation_mode:
                response_state["evaluation_mode"] = last_evaluation_mode

        if trace is not None:
            trace.record_stage(
                "composite_finalize",
                {
                    "completed_subtask_count": len(completed_subtasks),
                    "failed_subtask": failed_subtask or {},
                    "content_length": len(final_text),
                    "content_preview": _truncate_text(final_text),
                },
            )

        yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=final_text)
        conversation.messages.append(Message(role="assistant", content=final_text))

    async def _tool_loop(
        self,
        conversation: Conversation,
        system_prompt: str,
        tool_schemas: list[dict],
        tool_ctx: ToolContext,
        trace: TraceContext | None = None,
        planner_decision: PlannerDecision | None = None,
        planner_runtime: dict[str, object] | None = None,
        response_state: dict[str, object] | None = None,
        skill_policy: SkillPolicy | None = None,
        goal_result: GoalExecutionResult | None = None,
        suppress_text_output: bool = False,
        suppress_assistant_append: bool = False,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Inner ReAct loop: call LLM -> execute tools -> repeat.

        P0 fix: when streaming is enabled, text_delta events are yielded
        in real-time as each token arrives from the LLM, instead of being
        buffered into a list first.
        """
        force_active = (
            planner_decision is not None
            and planner_decision.control_mode == ControlMode.FORCE_TOOL
            and bool(planner_decision.selected_tool)
        )
        force_retry_limit = 1
        course_task = _is_course_task(planner_decision)
        evidence_bundle: EvidenceBundle | None = None
        latest_grounded_tool_text = ""
        policy_allowed_tools = (
            set(skill_policy.allowed_tools)
            if skill_policy is not None and skill_policy.allowed_tools
            else set()
        )
        autonomous_active = (
            planner_decision is not None
            and planner_decision.control_mode == ControlMode.AUTONOMOUS
        )
        autonomous_allowed_tools = (
            set(policy_allowed_tools or self.tools.tool_names)
            if autonomous_active
            else set()
        )
        if autonomous_active:
            autonomous_allowed_tools.discard("document_ingest")
        autonomous_max_steps = min(
            self.config.max_tool_iterations,
            int(getattr(skill_policy, "max_steps", 5) or 5) if autonomous_active else self.config.max_tool_iterations,
        )
        autonomous_tool_counts: dict[str, int] = {}
        tool_path: list[str] = []
        replan_count = 0
        last_replan_notice = ""

        if autonomous_active and response_state is not None:
            response_state.setdefault("replan_triggered", False)
            response_state.setdefault("replan_reason", "")
            response_state.setdefault("replan_count", 0)
            response_state.setdefault("fallback_strategy", "")
            response_state.setdefault("pacing_level", "normal")
            response_state.setdefault("pacing_reason", "")

        def _update_autonomous_state(stop_reason: str | None = None) -> None:
            if not autonomous_active:
                return
            if response_state is not None:
                response_state["autonomous_step_count"] = len(tool_path)
                response_state["tool_path"] = list(tool_path)
                if stop_reason:
                    response_state["autonomous_stop_reason"] = stop_reason
            if trace is not None and stop_reason:
                trace.record_stage(
                    "autonomous_stop_reason",
                    {
                        "reason": stop_reason,
                        "tool_path": list(tool_path),
                        "step_count": len(tool_path),
                    },
                )

        if self._should_use_direct_tool_path(planner_decision, _latest_user_message(conversation)):
            if (
                planner_decision is not None
                and planner_decision.task_intent == TaskIntent.QUIZ_EVALUATOR
            ):
                direct_tool_call = await self._build_direct_quiz_evaluator_call(
                    conversation,
                    tool_ctx,
                    planner_decision,
                )
            else:
                direct_tool_call = self._build_direct_tool_call(
                    conversation,
                    tool_ctx,
                    planner_decision,
                )
            if direct_tool_call is not None:
                if trace is not None:
                    trace.record_stage(
                        "direct_tool_dispatch",
                        {
                            "task_intent": planner_decision.task_intent.value,
                            "selected_tool": planner_decision.selected_tool,
                            "arguments": _summarize_arguments(direct_tool_call.arguments),
                            "response_profile": self._response_profile(),
                        },
                    )
                async for event in self._run_direct_tool_path(
                    conversation,
                    system_prompt,
                    tool_ctx,
                    planner_decision=planner_decision,
                    trace=trace,
                    response_state=response_state,
                    goal_result=goal_result,
                    suppress_text_output=suppress_text_output,
                    suppress_assistant_append=suppress_assistant_append,
                ):
                    yield event
                return

        for iteration in range(autonomous_max_steps if autonomous_active else self.config.max_tool_iterations):
            iteration_started = time.monotonic()
            iteration_prompt = system_prompt
            available_tool_schemas = tool_schemas
            knowledge_retry_active = (
                planner_decision is not None
                and planner_decision.task_intent == TaskIntent.KNOWLEDGE_QUERY
                and evidence_bundle is None
                and planner_runtime is not None
                and int(planner_runtime.get("knowledge_retry_count", 0)) > 0
            )
            effective_control_mode = (
                planner_runtime.get("final_control_mode")
                if planner_runtime is not None
                else (
                    planner_decision.control_mode.value
                    if planner_decision is not None
                    else ControlMode.PASS_THROUGH.value
                )
            )

            if autonomous_active:
                pacing_level, pacing_reason = self._compute_pacing(conversation)
                replan_notice = last_replan_notice
                last_replan_notice = ""
                visible_tools = {
                    name
                    for name in autonomous_allowed_tools
                    if autonomous_tool_counts.get(name, 0) < 2
                } or set(autonomous_allowed_tools)
                available_tool_schemas = _restrict_tool_schemas_to_allowed(
                    tool_schemas,
                    visible_tools,
                )
                iteration_prompt = (
                    system_prompt
                    + "\n\n## [Autonomous Policy]\n"
                    + "你处于受控自主模式。请根据上一步观察决定下一步，只调用最必要的一个工具；"
                    + "若已有足够信息，请直接收敛并回答用户。\n"
                    + f"允许工具: {', '.join(sorted(visible_tools))}\n"
                    + f"最大工具步数: {autonomous_max_steps}"
                    + "\n\n"
                    + REPLAN_POLICY
                    + "\n\n"
                    + _build_pacing_prompt(pacing_level, pacing_reason)
                )
                if replan_notice:
                    iteration_prompt += f"\n\n## [Replan Signal]\n{replan_notice}"
                if response_state is not None:
                    response_state["pacing_level"] = pacing_level
                    response_state["pacing_reason"] = pacing_reason
                if trace is not None:
                    trace.record_stage(
                        "autonomous_iteration",
                        {
                            "iteration": iteration + 1,
                            "allowed_tools": sorted(visible_tools),
                            "used_tool_counts": dict(autonomous_tool_counts),
                        },
                    )
                    trace.record_stage(
                        "adaptive_pacing",
                        {
                            "iteration": iteration + 1,
                            "pacing_level": pacing_level,
                            "pacing_reason": pacing_reason,
                        },
                    )
            elif force_active and planner_decision is not None:
                available_tool_schemas = _restrict_tool_schemas(
                    tool_schemas,
                    planner_decision.selected_tool,
                )
                reminder = (
                    f"你必须先调用工具 `{planner_decision.selected_tool}`，"
                    "不要直接回答用户。"
                )
                if planner_runtime is not None and int(planner_runtime.get("force_retry_count", 0)) > 0:
                    reminder = (
                        f"上一次你没有调用 `{planner_decision.selected_tool}`。"
                        f"这一次必须先调用 `{planner_decision.selected_tool}`，然后再继续回答。"
                    )
                iteration_prompt = system_prompt + f"\n\n## [Planner Enforcement]\n{reminder}"
            elif knowledge_retry_active:
                available_tool_schemas = _restrict_tool_schemas(
                    tool_schemas,
                    TaskIntent.KNOWLEDGE_QUERY.value,
                )
                retry_prompt = (
                    "这是明确的课程知识库问答。"
                    "你上一次没有先调用 `knowledge_query`。"
                    "这一次请先调用 `knowledge_query` 获取课程证据，再回答用户。"
                )
                iteration_prompt = system_prompt + f"\n\n## [Knowledge Query Retry]\n{retry_prompt}"
            elif evidence_bundle is not None:
                iteration_prompt = (
                    system_prompt
                    + "\n\n## [Grounding Context]\n"
                    + build_grounding_context(evidence_bundle, course_task=course_task)
                )

            if not autonomous_active and policy_allowed_tools:
                available_tool_schemas = _restrict_tool_schemas_to_allowed(
                    available_tool_schemas,
                    policy_allowed_tools,
                )

            llm_messages = await self._build_llm_messages(conversation, iteration_prompt)

            request = LlmRequest(
                messages=llm_messages,
                tools=available_tool_schemas if available_tool_schemas else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=self.config.stream_responses and not force_active and not course_task,
                metadata={
                    "course_task": course_task,
                    "skip_reflection_warning": course_task,
                    "citations": evidence_bundle.citations if evidence_bundle is not None else [],
                    "evidence_summary": (
                        evidence_bundle.evidence_summary if evidence_bundle is not None else ""
                    ),
                    "source_count": (
                        evidence_bundle.source_count if evidence_bundle is not None else 0
                    ),
                    "generation_mode": (
                        evidence_bundle.generation_mode if evidence_bundle is not None else ""
                    ),
                },
            )

            for mw in self.middlewares:
                try:
                    request = await mw.before_llm_request(request)
                except Exception:
                    logger.exception("before_llm_request middleware failed")

            if request.stream:
                # P0: consume stream in real-time — yield text_delta as each
                # token arrives, while accumulating the full LlmResponse.
                chunks: list[LlmStreamChunk] = []
                content_parts: list[str] = []

                async for chunk in self.llm.stream_request(request):
                    chunks.append(chunk)
                    if chunk.delta_content:
                        content_parts.append(chunk.delta_content)
                        yield StreamEvent(
                            type=StreamEventType.TEXT_DELTA,
                            content=chunk.delta_content,
                        )

                full_content = "".join(content_parts) or None
                tool_calls = _accumulate_tool_calls(chunks)
                response = LlmResponse(content=full_content, tool_calls=tool_calls)
            else:
                response = await self.llm.send_request(request)

            for mw in reversed(self.middlewares):
                try:
                    response = await mw.after_llm_response(request, response)
                except Exception:
                    logger.exception("after_llm_response middleware failed")

            llm_iteration_data = {
                "iteration": iteration + 1,
                "input_message_count": len(llm_messages),
                "stream": request.stream,
                "tool_names": [tc.name for tc in response.tool_calls or []],
                "tool_call_count": len(response.tool_calls or []),
                "output_text_length": len(response.content or ""),
                "had_error": bool(response.error),
                "planner_control_mode": effective_control_mode,
                "planner_forced_tool": (
                    planner_decision.selected_tool
                    if force_active and planner_decision is not None
                    else ""
                ),
                "visible_tool_count": len(available_tool_schemas),
                "has_evidence": bool(evidence_bundle and evidence_bundle.has_evidence),
                "generation_mode": (
                    evidence_bundle.generation_mode if evidence_bundle is not None else ""
                ),
            }
            if trace is not None:
                trace.record_stage(
                    "llm_iteration",
                    llm_iteration_data,
                    elapsed_ms=(time.monotonic() - iteration_started) * 1000.0,
                )

            if response.error:
                if trace is not None:
                    trace.metadata["status"] = "error"
                    trace.record_stage(
                        "error",
                        {
                            "phase": "llm_response",
                            "iteration": iteration + 1,
                            "error": _truncate_text(response.error),
                        },
                    )
                yield StreamEvent(type=StreamEventType.ERROR, content=response.error)
                return

            if (
                not response.tool_calls
                and not autonomous_active
                and planner_decision is not None
                and planner_decision.task_intent == TaskIntent.KNOWLEDGE_QUERY
                and evidence_bundle is None
                and planner_runtime is not None
                and int(planner_runtime.get("knowledge_retry_count", 0)) >= 1
                and not bool(planner_runtime.get("manual_knowledge_query_used", False))
            ):
                user_query = next(
                    (
                        msg.content
                        for msg in reversed(conversation.messages)
                        if msg.role == "user" and msg.content
                    ),
                    "",
                )
                if user_query:
                    planner_runtime["manual_knowledge_query_used"] = True
                    if trace is not None:
                        trace.record_stage(
                            "knowledge_query_fallback",
                            {
                                "iteration": iteration + 1,
                                "action": "manual_tool_dispatch",
                            },
                        )
                    response = response.model_copy(
                        update={
                            "tool_calls": [
                                ToolCallData(
                                    id=f"manual_kq_{iteration + 1}",
                                    name=TaskIntent.KNOWLEDGE_QUERY.value,
                                    arguments={
                                        "query": user_query,
                                        "top_k": 5,
                                    },
                                )
                            ]
                        }
                    )

            if response.tool_calls:
                conversation.messages.append(
                    Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
                )
                passthrough_payload: tuple[str, dict[str, Any]] | None = None

                for tc in response.tool_calls:
                    tool_started = time.monotonic()
                    yield StreamEvent(
                        type=StreamEventType.TOOL_START,
                        tool_name=tc.name,
                        metadata={"arguments": tc.arguments},
                    )

                    tool_result_obj: ToolResult
                    policy_blocked = False
                    effective_allowed_tools = autonomous_allowed_tools if autonomous_active else policy_allowed_tools
                    if effective_allowed_tools and tc.name not in effective_allowed_tools:
                        policy_blocked = True
                        tool_result_obj = ToolResult(
                            success=False,
                            error=f"工具 `{tc.name}` 不在当前技能允许范围内",
                            metadata={
                                "tool_name": tc.name,
                                "error_type": "policy_blocked",
                                "retryable": False,
                            },
                        )
                        if trace is not None:
                            trace.record_stage(
                                "autonomous_guardrail",
                                {
                                    "iteration": iteration + 1,
                                    "tool_name": tc.name,
                                    "reason": "tool_not_allowed",
                                },
                            )
                    elif autonomous_active and autonomous_tool_counts.get(tc.name, 0) >= 2:
                        policy_blocked = True
                        tool_result_obj = ToolResult(
                            success=False,
                            error=f"工具 `{tc.name}` 在本轮任务中已达到最大调用次数",
                            metadata={
                                "tool_name": tc.name,
                                "error_type": "policy_blocked",
                                "retryable": False,
                            },
                        )
                        if trace is not None:
                            trace.record_stage(
                                "autonomous_guardrail",
                                {
                                    "iteration": iteration + 1,
                                    "tool_name": tc.name,
                                    "reason": "tool_reuse_limit",
                                },
                            )
                    else:
                        if autonomous_active:
                            autonomous_tool_counts[tc.name] = autonomous_tool_counts.get(tc.name, 0) + 1
                            tool_path.append(tc.name)

                    tool_blocked = False
                    if not policy_blocked:
                        for hook in self.hooks:
                            try:
                                await hook.before_tool(tc.name, tool_ctx)
                            except Exception as exc:
                                logger.warning("before_tool hook blocked %s: %s", tc.name, exc)
                                tool_blocked = True
                                break

                    if policy_blocked:
                        pass
                    elif tool_blocked:
                        tool_result_obj = ToolResult(
                            success=False,
                            error="Tool execution blocked by hook",
                            metadata={
                                "tool_name": tc.name,
                                "error_type": ToolErrorType.BLOCKED_BY_HOOK.value,
                                "retryable": False,
                            },
                        )
                    else:
                        tool_result_obj = await self.tools.execute(
                            tc, tool_ctx, timeout=float(self.config.tool_timeout)
                        )

                    for hook in self.hooks:
                        try:
                            modified = await hook.after_tool(tc.name, tool_result_obj, context=tool_ctx)
                            if modified is not None:
                                tool_result_obj = modified
                        except Exception:
                            logger.exception("after_tool hook failed")

                    yield StreamEvent(
                        type=StreamEventType.TOOL_RESULT,
                        tool_name=tc.name,
                        content=tool_result_obj.result_for_llm if tool_result_obj.success else tool_result_obj.error,
                        metadata={
                            "success": tool_result_obj.success,
                            **tool_result_obj.metadata,
                        },
                    )

                    conversation.messages.append(
                        Message(
                            role="tool",
                            content=tool_result_obj.result_for_llm if tool_result_obj.success else (tool_result_obj.error or ""),
                            tool_call_id=tc.id,
                            metadata=dict(tool_result_obj.metadata),
                        )
                    )

                    bundle = build_evidence_bundle(tc.name, tool_result_obj.metadata)
                    if bundle is not None:
                        evidence_bundle = bundle
                        latest_grounded_tool_text = tool_result_obj.result_for_llm or ""
                    replan_needed = False
                    replan_reason = ""
                    fallback_strategy = ""
                    if autonomous_active:
                        replan_needed, replan_reason, fallback_strategy = self._assess_replan_signal(
                            tc.name,
                            tool_result_obj,
                        )
                        if replan_needed:
                            replan_count += 1
                            last_replan_notice = (
                                f"工具 `{tc.name}` 的结果不足以完成任务：{replan_reason}。"
                                f"优先采用策略：{fallback_strategy}。"
                            )
                            if response_state is not None:
                                response_state["replan_triggered"] = True
                                response_state["replan_reason"] = replan_reason
                                response_state["replan_count"] = replan_count
                                response_state["fallback_strategy"] = fallback_strategy
                            if trace is not None:
                                trace.record_stage(
                                    "replan_signal",
                                    {
                                        "iteration": iteration + 1,
                                        "tool_name": tc.name,
                                        "reason": replan_reason,
                                        "fallback_strategy": fallback_strategy,
                                    },
                                )
                    if _tool_result_is_final_answer(tc.name, tool_result_obj.metadata) and not (
                        autonomous_active and replan_needed
                    ):
                        passthrough_payload = (
                            tool_result_obj.result_for_llm if tool_result_obj.success else (tool_result_obj.error or ""),
                            tool_result_obj.metadata,
                        )
                    if autonomous_active and response_state is not None:
                        response_state["tool_path"] = list(tool_path)
                        response_state["autonomous_step_count"] = len(tool_path)

                    if trace is not None:
                        trace.record_stage(
                            "tool_execution",
                            {
                                "iteration": iteration + 1,
                                "tool_name": tc.name,
                                "arguments": _summarize_arguments(tc.arguments),
                                "success": tool_result_obj.success,
                                "error_type": tool_result_obj.metadata.get("error_type", ""),
                                "retryable": bool(tool_result_obj.metadata.get("retryable", False)),
                                "query_trace_id": tool_result_obj.metadata.get("query_trace_id"),
                                "generation_mode": tool_result_obj.metadata.get("generation_mode", ""),
                                "evaluation_mode": tool_result_obj.metadata.get("evaluation_mode", ""),
                                "source_count": int(tool_result_obj.metadata.get("source_count", 0) or 0),
                                "error_summary": _truncate_text(tool_result_obj.error),
                                "result_summary": _truncate_text(tool_result_obj.result_for_llm),
                            },
                            elapsed_ms=(time.monotonic() - tool_started) * 1000.0,
                        )

                if passthrough_payload is not None:
                    generation_mode = str(passthrough_payload[1].get("generation_mode", "") or "")
                    evaluation_mode = str(passthrough_payload[1].get("evaluation_mode", "") or "")
                    grounding_passthrough = bool(
                        passthrough_payload[1].get("grounding_passthrough", False)
                    )
                    if generation_mode in {"question_bank", "rag_backed"}:
                        final_text = passthrough_payload[0]
                        grounding_assessment = GroundingAssessment(
                            score=1.0,
                            has_evidence=bool(evidence_bundle and evidence_bundle.has_evidence),
                            citation_count=0,
                            source_count=evidence_bundle.source_count if evidence_bundle is not None else 0,
                        )
                    elif grounding_passthrough:
                        final_text = passthrough_payload[0]
                        grounding_assessment = self._grounding_evaluator.assess(
                            final_text,
                            evidence_bundle,
                        )
                    else:
                        final_text, grounding_assessment = await self._finalize_answer(
                            passthrough_payload[0],
                            course_task=course_task,
                            evidence_bundle=evidence_bundle,
                            fallback_text=latest_grounded_tool_text,
                        )
                    if trace is not None and grounding_assessment is not None:
                        trace.record_stage(
                            "answer_grounding",
                            {
                                **grounding_assessment.to_metadata(),
                                "generation_mode": generation_mode,
                                "evaluation_mode": evaluation_mode,
                            },
                        )
                    if response_state is not None:
                        response_state.update(
                            _build_evidence_metadata(evidence_bundle, grounding_assessment)
                        )
                        response_state.update(
                            _build_batch_evaluation_metadata(passthrough_payload[1])
                        )
                    if response_state is not None and generation_mode:
                        response_state["generation_mode"] = generation_mode
                    if trace is not None and generation_mode:
                        trace.metadata["generation_mode"] = generation_mode
                    if response_state is not None and evaluation_mode:
                        response_state["evaluation_mode"] = evaluation_mode
                    if trace is not None and evaluation_mode:
                        trace.metadata["evaluation_mode"] = evaluation_mode
                    if goal_result is not None:
                        goal_result.text = final_text or ""
                        goal_result.metadata = dict(passthrough_payload[1])
                        goal_result.resume_payload = dict(passthrough_payload[1].get("resume_payload", {}) or {})
                        goal_result.completion_hint = _tool_completion_hint(tc.name, passthrough_payload[1])
                        if goal_result.completion_hint == "wait_user":
                            goal_result.goal_status = GoalStatus.WAITING_USER.value
                        else:
                            goal_result.goal_status = GoalStatus.COMPLETED.value
                    if final_text and not suppress_text_output:
                        yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=final_text)
                    if trace is not None:
                        trace.record_stage(
                            "final_response",
                            {
                                "iteration": iteration + 1,
                                "content_length": len(final_text or ""),
                                "content_preview": _truncate_text(final_text),
                                "generation_mode": generation_mode,
                                "evaluation_mode": evaluation_mode,
                            },
                        )
                    _update_autonomous_state("tool_final_answer")
                    if not suppress_assistant_append:
                        conversation.messages.append(Message(role="assistant", content=final_text or ""))
                    return

                if force_active:
                    force_active = False
                continue

            if force_active and planner_decision is not None:
                if trace is not None:
                    trace.record_stage(
                        "planner_violation",
                        {
                            "iteration": iteration + 1,
                            "selected_tool": planner_decision.selected_tool,
                            "force_retry_count": (
                                planner_runtime.get("force_retry_count", 0)
                                if planner_runtime is not None
                                else 0
                            ),
                            "action": "retry_force_tool",
                        },
                    )
                if planner_runtime is not None:
                    planner_runtime["violation_count"] = int(
                        planner_runtime.get("violation_count", 0)
                    ) + 1
                if planner_runtime is not None and int(planner_runtime.get("force_retry_count", 0)) < force_retry_limit:
                    planner_runtime["force_retry_count"] = int(
                        planner_runtime.get("force_retry_count", 0)
                    ) + 1
                    continue
                force_active = False
                if planner_runtime is not None:
                    planner_runtime["final_control_mode"] = ControlMode.ADVISORY.value
                if trace is not None:
                    trace.record_stage(
                        "planner_violation",
                        {
                            "iteration": iteration + 1,
                            "selected_tool": planner_decision.selected_tool,
                            "action": "downgrade_to_advisory",
                        },
                    )
                continue

            if (
                not autonomous_active
                and
                planner_decision is not None
                and planner_decision.task_intent == TaskIntent.KNOWLEDGE_QUERY
                and evidence_bundle is None
                and planner_runtime is not None
                and int(planner_runtime.get("knowledge_retry_count", 0)) == 0
            ):
                planner_runtime["knowledge_retry_count"] = 1
                if trace is not None:
                    trace.record_stage(
                        "knowledge_query_retry",
                        {
                            "iteration": iteration + 1,
                            "action": "retry_with_knowledge_query_priority",
                        },
                    )
                continue

            final_content, grounding_assessment = await self._finalize_answer(
                response.content or "",
                course_task=course_task,
                evidence_bundle=evidence_bundle,
                fallback_text=latest_grounded_tool_text,
            )
            if final_content and not request.stream and not suppress_text_output:
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=final_content)
            if trace is not None and grounding_assessment is not None:
                trace.record_stage(
                    "answer_grounding",
                    {
                        **grounding_assessment.to_metadata(),
                        "generation_mode": (
                            evidence_bundle.generation_mode if evidence_bundle is not None else ""
                        ),
                        "evaluation_mode": (
                            evidence_bundle.evaluation_mode if evidence_bundle is not None else ""
                        ),
                    },
                )
            if response_state is not None:
                response_state.update(_build_evidence_metadata(evidence_bundle, grounding_assessment))
                if autonomous_active:
                    response_state["tool_path"] = list(tool_path)
                    response_state["autonomous_step_count"] = len(tool_path)
            if trace is not None:
                trace.record_stage(
                    "final_response",
                    {
                        "iteration": iteration + 1,
                        "content_length": len(final_content or ""),
                        "content_preview": _truncate_text(final_content),
                        "generation_mode": (
                            evidence_bundle.generation_mode if evidence_bundle is not None else ""
                        ),
                        "evaluation_mode": (
                            evidence_bundle.evaluation_mode if evidence_bundle is not None else ""
                        ),
                    },
                )
            _update_autonomous_state("assistant_answer")
            if goal_result is not None:
                goal_result.text = final_content or ""
                goal_result.metadata = {}
                goal_result.resume_payload = {}
                goal_result.completion_hint = "step_done"
                goal_result.goal_status = GoalStatus.COMPLETED.value
            if not suppress_assistant_append:
                conversation.messages.append(Message(role="assistant", content=final_content or ""))
            return

        if trace is not None:
            trace.metadata["status"] = "error"
            trace.record_stage(
                "error",
                {
                    "phase": "tool_loop",
                    "error": f"max_iterations_exceeded:{autonomous_max_steps if autonomous_active else self.config.max_tool_iterations}",
                },
            )
        _update_autonomous_state("max_iterations_exceeded")
        if goal_result is not None:
            goal_result.error = (
                f"达到最大工具调用轮次 ({autonomous_max_steps if autonomous_active else self.config.max_tool_iterations})"
            )
            goal_result.goal_status = GoalStatus.BLOCKED.value
            goal_result.completion_hint = "clarify"
        yield StreamEvent(
            type=StreamEventType.ERROR,
            content=f"达到最大工具调用轮次 ({autonomous_max_steps if autonomous_active else self.config.max_tool_iterations})，请简化问题后重试。",
        )

    async def _finalize_answer(
        self,
        content: str,
        *,
        course_task: bool,
        evidence_bundle: EvidenceBundle | None,
        fallback_text: str = "",
    ) -> tuple[str, GroundingAssessment | None]:
        if not course_task:
            return content, None

        assessment = self._grounding_evaluator.assess(content, evidence_bundle)
        if not assessment.low_evidence:
            return content, assessment

        if evidence_bundle is not None and evidence_bundle.has_evidence:
            if self._grounding_mode != "strict":
                assessment.policy_action = GroundingPolicyAction.LOW_EVIDENCE_WARNING.value
                softened = content or fallback_text or DEFAULT_LOW_EVIDENCE_MESSAGE
                return _append_warning(softened, DEFAULT_BALANCED_LOW_EVIDENCE_NOTE), assessment
            rewritten = await self._rewrite_answer_conservatively(content, evidence_bundle)
            if rewritten:
                rewritten_assessment = self._grounding_evaluator.assess(rewritten, evidence_bundle)
                rewritten_assessment.conservative_rewrite_used = True
                if not rewritten_assessment.low_evidence:
                    rewritten_assessment.policy_action = GroundingPolicyAction.CONSERVATIVE_REWRITE.value
                    return rewritten, rewritten_assessment
                rewritten_assessment.policy_action = GroundingPolicyAction.LOW_EVIDENCE_WARNING.value
                return (
                    _append_warning(
                        rewritten,
                        "> ⚠️ 现有课程证据不足，以上回答已按可验证范围保守表达。",
                    ),
                    rewritten_assessment,
                )

        conservative = fallback_text or content or DEFAULT_LOW_EVIDENCE_MESSAGE
        if not any(token in conservative for token in ("证据不足", "未找到", "暂不生成", "无法可靠")):
            conservative = DEFAULT_LOW_EVIDENCE_MESSAGE
        assessment.policy_action = GroundingPolicyAction.LOW_EVIDENCE_WARNING.value
        assessment.low_evidence = True
        return conservative, assessment

    async def _rewrite_answer_conservatively(
        self,
        content: str,
        evidence_bundle: EvidenceBundle,
    ) -> str:
        system_prompt, user_prompt = build_conservative_rewrite_messages(content, evidence_bundle)
        rewrite_request = LlmRequest(
            messages=[
                LlmMessage(role="system", content=system_prompt),
                LlmMessage(role="user", content=user_prompt),
            ],
            temperature=0.2,
            max_tokens=self.config.max_tokens,
            stream=False,
            metadata={
                "course_task": True,
                "skip_reflection_warning": True,
                "citations": evidence_bundle.citations,
                "evidence_summary": evidence_bundle.evidence_summary,
                "generation_mode": evidence_bundle.generation_mode,
            },
        )
        try:
            for mw in self.middlewares:
                rewrite_request = await mw.before_llm_request(rewrite_request)
            rewrite_response = await self.llm.send_request(rewrite_request)
            for mw in reversed(self.middlewares):
                rewrite_response = await mw.after_llm_response(rewrite_request, rewrite_response)
        except Exception:
            logger.exception("Conservative rewrite failed")
            return ""
        if rewrite_response.error:
            logger.warning("Conservative rewrite returned error: %s", rewrite_response.error)
            return ""
        return (rewrite_response.content or "").strip()

    async def _build_llm_messages(
        self, conversation: Conversation, system_prompt: str
    ) -> list[LlmMessage]:
        """Convert conversation history into LlmMessage list with system prompt."""
        messages = [LlmMessage(role="system", content=system_prompt)]

        # Apply context filter if available; otherwise simple sliding window
        if self.context_filter and hasattr(self.context_filter, "filter_messages_async"):
            filtered = await self.context_filter.filter_messages_async(conversation.messages)
        elif self.context_filter and hasattr(self.context_filter, "filter_messages"):
            filtered = self.context_filter.filter_messages(conversation.messages)
        else:
            filtered = conversation.messages[-self.config.max_context_messages:]

        for m in filtered:
            messages.append(
                LlmMessage(
                    role=m.role,
                    content=m.content,
                    tool_calls=m.tool_calls,
                    tool_call_id=m.tool_call_id,
                )
            )
        return messages
