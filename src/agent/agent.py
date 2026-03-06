"""Agent core — ReAct tool-loop orchestrator with streaming output.

This is the central class that ties together LLM, Tools, Conversation,
LifecycleHooks, LlmMiddlewares, Memory and Skills.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import AsyncGenerator, Optional

from src.agent.config import AgentConfig
from src.agent.conversation import ConversationStore
from src.agent.hooks.lifecycle import LifecycleHook
from src.agent.hooks.middleware import LlmMiddleware
from src.agent.llm.base import LlmService
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
    ToolResult,
)

logger = logging.getLogger(__name__)


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
        skill_workflow: object | None = None,
        context_filter: object | None = None,
        review_hook: object | None = None,
    ) -> None:
        self.llm = llm_service
        self.tools = tool_registry
        self.conversations = conversation_store
        self.config = config
        self.prompt_builder = prompt_builder or SystemPromptBuilder(config.system_prompt_path)
        self.hooks = lifecycle_hooks or []
        self.middlewares = llm_middlewares or []
        self.memory_enhancer = memory_enhancer
        self.skill_workflow = skill_workflow
        self.context_filter = context_filter
        self.review_hook = review_hook

    async def chat(
        self,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Process a user message through the ReAct tool loop, yielding stream events."""
        request_id = uuid.uuid4().hex[:12]

        # --- Load or create conversation ---
        conversation: Conversation | None = None
        if conversation_id:
            conversation = await self.conversations.get(conversation_id, user_id)
        if conversation is None:
            conversation = await self.conversations.create(user_id)

        # --- Before-message hooks ---
        effective_message = message
        for hook in self.hooks:
            try:
                modified = await hook.before_message(user_id, effective_message)
                if modified is not None:
                    effective_message = modified
            except Exception:
                logger.exception("before_message hook failed")

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

        memory_ctx, review_ctx, wf_result = await asyncio.gather(
            _fetch_memory(), _fetch_review(), _fetch_skill()
        )

        if review_ctx:
            memory_ctx = (memory_ctx + "\n\n" + review_ctx).strip() if memory_ctx else review_ctx

        skill_ctx = ""
        if wf_result is not None:
            if hasattr(wf_result, "direct_response") and wf_result.direct_response:
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=wf_result.direct_response)
                yield StreamEvent(type=StreamEventType.DONE)
                return
            if hasattr(wf_result, "skill_instruction") and wf_result.skill_instruction:
                skill_ctx = wf_result.skill_instruction

        system_prompt = self.prompt_builder.build(
            tool_schemas=tool_schemas,
            memory_context=memory_ctx,
            active_skill=skill_ctx,
        )

        tool_ctx = ToolContext(
            user_id=user_id,
            conversation_id=conversation.id,
            request_id=request_id,
        )

        # --- Tool loop ---
        try:
            async for event in self._tool_loop(conversation, system_prompt, tool_schemas, tool_ctx):
                yield event
        except Exception as exc:
            logger.exception("Agent tool loop error")
            yield StreamEvent(type=StreamEventType.ERROR, content=str(exc))

        yield StreamEvent(
            type=StreamEventType.DONE,
            metadata={"conversation_id": conversation.id, "title": conversation.title},
        )

        # P8: save + after-message hooks run in background so the SSE generator
        # completes immediately after DONE, freeing the HTTP connection.
        asyncio.create_task(self._post_message_tasks(conversation))

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
    ) -> AsyncGenerator[StreamEvent, None]:
        """Inner ReAct loop: call LLM -> execute tools -> repeat.

        P0 fix: when streaming is enabled, text_delta events are yielded
        in real-time as each token arrives from the LLM, instead of being
        buffered into a list first.
        """
        for iteration in range(self.config.max_tool_iterations):
            llm_messages = await self._build_llm_messages(conversation, system_prompt)

            request = LlmRequest(
                messages=llm_messages,
                tools=tool_schemas if tool_schemas else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=self.config.stream_responses,
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

            if response.error:
                yield StreamEvent(type=StreamEventType.ERROR, content=response.error)
                return

            if response.tool_calls:
                conversation.messages.append(
                    Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
                )

                for tc in response.tool_calls:
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
                            success=False, error="Tool execution blocked by hook"
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
                        metadata={"success": tool_result_obj.success},
                    )

                    conversation.messages.append(
                        Message(
                            role="tool",
                            content=tool_result_obj.result_for_llm if tool_result_obj.success else (tool_result_obj.error or ""),
                            tool_call_id=tc.id,
                        )
                    )

                continue

            if response.content and not request.stream:
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=response.content)
            conversation.messages.append(Message(role="assistant", content=response.content or ""))
            return

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
