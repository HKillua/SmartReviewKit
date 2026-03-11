"""Agent core — ReAct tool-loop orchestrator with streaming output.

This is the central class that ties together LLM, Tools, Conversation,
LifecycleHooks, LlmMiddlewares, Memory and Skills.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from typing import AsyncGenerator, Optional

from src.agent.config import AgentConfig
from src.agent.conversation import ConversationStore
from src.agent.hooks.lifecycle import LifecycleHook
from src.agent.hooks.middleware import LlmMiddleware
from src.agent.llm.base import LlmService
from src.agent.planner import ControlMode, PlannerDecision, TaskPlanner
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


def _build_planner_context(decision: PlannerDecision | None) -> str:
    if decision is None or decision.control_mode == ControlMode.PASS_THROUGH:
        return ""
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
    return {
        "planner_task_intent": decision.task_intent.value,
        "planner_control_mode": decision.control_mode.value,
        "planner_selected_tool": decision.selected_tool,
        "planner_confidence": round(decision.confidence, 3),
        "planner_match_method": decision.match_method,
    }


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
            if self.task_planner is not None:
                try:
                    planner_decision = self.task_planner.plan(
                        effective_message,
                        matched_skill=matched_skill or None,
                    )
                    planner_context = _build_planner_context(planner_decision)
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
            }

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
            tool_ctx = ToolContext(
                user_id=user_id,
                conversation_id=conversation.id,
                request_id=request_id,
                metadata=tool_metadata,
                recent_messages=recent_msgs,
            )

            # --- Tool loop ---
            try:
                async for event in self._tool_loop(
                    conversation,
                    system_prompt,
                    tool_schemas,
                    tool_ctx,
                    trace=agent_trace,
                    planner_decision=planner_decision,
                    planner_runtime=planner_runtime,
                ):
                    yield event
            except Exception as exc:
                logger.exception("Agent tool loop error")
                if agent_trace is not None:
                    agent_trace.metadata["status"] = "error"
                    agent_trace.record_stage(
                        "error",
                        {"phase": "tool_loop", "error": _truncate_text(str(exc))},
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
            done_metadata.update(_planner_metadata(planner_decision))
            done_metadata["planner_final_control_mode"] = planner_runtime["final_control_mode"]
            done_metadata["planner_violation_count"] = planner_runtime["violation_count"]
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

    async def _tool_loop(
        self,
        conversation: Conversation,
        system_prompt: str,
        tool_schemas: list[dict],
        tool_ctx: ToolContext,
        trace: TraceContext | None = None,
        planner_decision: PlannerDecision | None = None,
        planner_runtime: dict[str, object] | None = None,
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

        for iteration in range(self.config.max_tool_iterations):
            iteration_started = time.monotonic()
            iteration_prompt = system_prompt
            available_tool_schemas = tool_schemas
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

            llm_messages = await self._build_llm_messages(conversation, iteration_prompt)

            request = LlmRequest(
                messages=llm_messages,
                tools=available_tool_schemas if available_tool_schemas else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=self.config.stream_responses and not force_active,
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

            if response.tool_calls:
                conversation.messages.append(
                    Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
                )

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
                                "error_summary": _truncate_text(tool_result_obj.error),
                                "result_summary": _truncate_text(tool_result_obj.result_for_llm),
                            },
                            elapsed_ms=(time.monotonic() - tool_started) * 1000.0,
                        )

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

            if response.content and not request.stream:
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=response.content)
            if trace is not None:
                trace.record_stage(
                    "final_response",
                    {
                        "iteration": iteration + 1,
                        "content_length": len(response.content or ""),
                        "content_preview": _truncate_text(response.content),
                    },
                )
            conversation.messages.append(Message(role="assistant", content=response.content or ""))
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
