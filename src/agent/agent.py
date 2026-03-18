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
from src.agent.prompt_builder import SystemPromptBuilder
from src.agent.tools.base import ToolRegistry
from src.agent.types import (
    Conversation,
    LlmMessage,
    LlmRequest,
    LlmResponse,
    LlmStreamChunk,
    Message,
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
    }
    if decision.subtasks:
        metadata["subtask_plan"] = _serialize_subtasks(decision.subtasks)
    return metadata


def _is_course_task(decision: PlannerDecision | None) -> bool:
    return decision is not None and decision.task_intent != TaskIntent.GENERAL_CHAT


def _append_warning(text: str, warning: str) -> str:
    if not warning or warning in text:
        return text
    return f"{text.rstrip()}\n\n{warning}"


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
                        return await self.review_hook.get_review_context(user_id)
                    except Exception:
                        logger.exception("Review hook failed")
                return ""

            async def _fetch_skill() -> object | None:
                if self.skill_workflow and hasattr(self.skill_workflow, "try_handle"):
                    try:
                        return await self.skill_workflow.try_handle(effective_message, user_id)
                    except Exception:
                        logger.exception("Skill workflow handler failed")
                return None

            preprocess_started = time.monotonic()
            memory_ctx, review_ctx, wf_result = await asyncio.gather(
                _fetch_memory(), _fetch_review(), _fetch_skill()
            )

            if review_ctx:
                memory_ctx = (memory_ctx + "\n\n" + review_ctx).strip() if memory_ctx else review_ctx

            skill_ctx = ""
            direct_response = ""
            matched_skill = ""
            if wf_result is not None:
                if hasattr(wf_result, "direct_response") and wf_result.direct_response:
                    direct_response = wf_result.direct_response
                if hasattr(wf_result, "skill_instruction") and wf_result.skill_instruction:
                    skill_ctx = wf_result.skill_instruction
                if hasattr(wf_result, "matched_skill") and wf_result.matched_skill:
                    matched_skill = wf_result.matched_skill

            planner_decision: PlannerDecision | None = None
            planner_context = ""
            course_task = False
            if self.task_planner is not None:
                try:
                    planner_decision = self.task_planner.plan(
                        effective_message,
                        matched_skill=matched_skill or None,
                    )
                    planner_context = _build_planner_context(planner_decision)
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
                    },
                    elapsed_ms=(time.monotonic() - preprocess_started) * 1000.0,
                )
                if planner_decision is not None:
                    agent_trace.record_stage("planner_decision", planner_decision.to_metadata())
                    agent_trace.metadata.update(_planner_metadata(planner_decision))

            prompt_started = time.monotonic()
            system_prompt = self.prompt_builder.build(
                tool_schemas=tool_schemas,
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
                },
                recent_messages=recent_msgs,
            )

            try:
                response_state: dict[str, object] = {}
                if planner_decision is not None and planner_decision.is_composite:
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
                                "composite_execution"
                                if planner_decision is not None and planner_decision.is_composite
                                else "tool_loop"
                            ),
                            "error": _truncate_text(str(exc)),
                        },
                    )
                yield StreamEvent(type=StreamEventType.ERROR, content=str(exc))

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

    def _response_profile(self) -> str:
        value = str(getattr(self.config, "response_profile", "") or "").strip().lower()
        return value if value in {"balanced_fast", "quality_first"} else "quality_first"

    def _should_use_direct_tool_path(
        self,
        planner_decision: PlannerDecision | None,
    ) -> bool:
        if self._response_profile() != "balanced_fast":
            return False
        if planner_decision is None or planner_decision.is_composite:
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
            parsed = _extract_evaluator_fields(user_message)
            question = parsed.get("question") or _clean_topic_text(user_message) or user_message
            if not question:
                return None
            return ToolCallData(
                id="direct_quiz_evaluator",
                name=selected_tool,
                arguments={
                    "question": question,
                    "user_answer": parsed.get("user_answer", ""),
                    "correct_answer": parsed.get("correct_answer", ""),
                    "question_type": _extract_question_type(user_message) or "选择题",
                    "topic": _extract_shared_topic(user_message) or question,
                    "concepts": [],
                },
            )
        return None

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
    ) -> AsyncGenerator[StreamEvent, None]:
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
            if generation_mode:
                response_state["generation_mode"] = generation_mode
            if evaluation_mode:
                response_state["evaluation_mode"] = evaluation_mode
        if final_text:
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=final_text)
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
        return base_context.model_copy(update={"metadata": metadata})

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

        if self._should_use_direct_tool_path(planner_decision):
            direct_tool_call = self._build_direct_tool_call(conversation, tool_ctx, planner_decision)
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
                ):
                    yield event
                return

        for iteration in range(self.config.max_tool_iterations):
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

            if force_active and planner_decision is not None:
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

                    tool_blocked = False
                    for hook in self.hooks:
                        try:
                            await hook.before_tool(tc.name, tool_ctx)
                        except Exception as exc:
                            logger.warning("before_tool hook blocked %s: %s", tc.name, exc)
                            tool_blocked = True
                            break

                    if tool_blocked:
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
                        )
                    )

                    bundle = build_evidence_bundle(tc.name, tool_result_obj.metadata)
                    if bundle is not None:
                        evidence_bundle = bundle
                        latest_grounded_tool_text = tool_result_obj.result_for_llm or ""
                    if _tool_result_is_final_answer(tc.name, tool_result_obj.metadata):
                        passthrough_payload = (
                            tool_result_obj.result_for_llm if tool_result_obj.success else (tool_result_obj.error or ""),
                            tool_result_obj.metadata,
                        )

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
                    if response_state is not None and generation_mode:
                        response_state["generation_mode"] = generation_mode
                    if trace is not None and generation_mode:
                        trace.metadata["generation_mode"] = generation_mode
                    if response_state is not None and evaluation_mode:
                        response_state["evaluation_mode"] = evaluation_mode
                    if trace is not None and evaluation_mode:
                        trace.metadata["evaluation_mode"] = evaluation_mode
                    if final_text:
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
            if final_content and not request.stream:
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
            conversation.messages.append(Message(role="assistant", content=final_content or ""))
            return

        if trace is not None:
            trace.metadata["status"] = "error"
            trace.record_stage(
                "error",
                {
                    "phase": "tool_loop",
                    "error": f"max_iterations_exceeded:{self.config.max_tool_iterations}",
                },
            )
        yield StreamEvent(
            type=StreamEventType.ERROR,
            content=f"达到最大工具调用轮次 ({self.config.max_tool_iterations})，请简化问题后重试。",
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
