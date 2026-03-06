"""Agent core — ReAct tool-loop orchestrator with streaming output.

This is the central class that ties together LLM, Tools, Conversation,
LifecycleHooks, LlmMiddlewares, Memory and Skills.
"""

from __future__ import annotations

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

        # --- Build system prompt ---
        tool_schemas = self.tools.get_all_schemas()

        memory_ctx = ""
        if self.memory_enhancer and hasattr(self.memory_enhancer, "get_memory_summary"):
            try:
                memory_ctx = await self.memory_enhancer.get_memory_summary(user_id)
            except Exception:
                logger.exception("Memory enhancer failed")

        # Proactive review recommendations (K4)
        review_ctx = ""
        if self.review_hook and hasattr(self.review_hook, "get_review_context"):
            try:
                review_ctx = await self.review_hook.get_review_context(user_id)
            except Exception:
                logger.exception("Review hook failed")
        if review_ctx:
            memory_ctx = (memory_ctx + "\n\n" + review_ctx).strip() if memory_ctx else review_ctx

        skill_ctx = ""
        if self.skill_workflow and hasattr(self.skill_workflow, "try_handle"):
            try:
                wf_result = await self.skill_workflow.try_handle(effective_message, user_id)
                if hasattr(wf_result, "direct_response") and wf_result.direct_response:
                    yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=wf_result.direct_response)
                    yield StreamEvent(type=StreamEventType.DONE)
                    return
                if hasattr(wf_result, "skill_instruction") and wf_result.skill_instruction:
                    skill_ctx = wf_result.skill_instruction
            except Exception:
                logger.exception("Skill workflow handler failed")

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

        # --- Save conversation ---
        await self.conversations.update(conversation)

        # --- After-message hooks ---
        for hook in self.hooks:
            try:
                await hook.after_message(conversation)
            except Exception:
                logger.exception("after_message hook failed")

    async def _tool_loop(
        self,
        conversation: Conversation,
        system_prompt: str,
        tool_schemas: list[dict],
        tool_ctx: ToolContext,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Inner ReAct loop: call LLM → execute tools → repeat."""
        for iteration in range(self.config.max_tool_iterations):
            # Build messages
            llm_messages = await self._build_llm_messages(conversation, system_prompt)

            request = LlmRequest(
                messages=llm_messages,
                tools=tool_schemas if tool_schemas else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=self.config.stream_responses,
            )

            # Apply before-request middlewares
            for mw in self.middlewares:
                try:
                    request = await mw.before_llm_request(request)
                except Exception:
                    logger.exception("before_llm_request middleware failed")

            if request.stream:
                response, events = await self._handle_streaming(request)
                for ev in events:
                    yield ev
            else:
                response = await self.llm.send_request(request)

            # Apply after-response middlewares (reverse order)
            for mw in reversed(self.middlewares):
                try:
                    response = await mw.after_llm_response(request, response)
                except Exception:
                    logger.exception("after_llm_response middleware failed")

            if response.error:
                yield StreamEvent(type=StreamEventType.ERROR, content=response.error)
                return

            # If LLM returned tool calls → execute tools
            if response.tool_calls:
                # Record assistant message with tool calls
                conversation.messages.append(
                    Message(role="assistant", tool_calls=response.tool_calls)
                )

                for tc in response.tool_calls:
                    yield StreamEvent(
                        type=StreamEventType.TOOL_START,
                        tool_name=tc.name,
                        metadata={"arguments": tc.arguments},
                    )

                    # Before-tool hooks
                    tool_blocked = False
                    for hook in self.hooks:
                        try:
                            await hook.before_tool(tc.name, tool_ctx)
                        except Exception as exc:
                            logger.warning("before_tool hook blocked %s: %s", tc.name, exc)
                            tool_blocked = True
                            break

                    if tool_blocked:
                        tool_result_obj = __import__("src.agent.types", fromlist=["ToolResult"]).ToolResult(
                            success=False, error="Tool execution blocked by hook"
                        )
                    else:
                        tool_result_obj = await self.tools.execute(
                            tc, tool_ctx, timeout=float(self.config.tool_timeout)
                        )

                    # After-tool hooks
                    for hook in self.hooks:
                        try:
                            modified = await hook.after_tool(tc.name, tool_result_obj)
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

                    # Record tool result message
                    conversation.messages.append(
                        Message(
                            role="tool",
                            content=tool_result_obj.result_for_llm if tool_result_obj.success else (tool_result_obj.error or ""),
                            tool_call_id=tc.id,
                        )
                    )

                # Continue the loop for next LLM call
                continue

            # No tool calls — plain text response
            if response.content and not request.stream:
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, content=response.content)
            conversation.messages.append(Message(role="assistant", content=response.content or ""))
            return

        # Exhausted iterations
        yield StreamEvent(
            type=StreamEventType.ERROR,
            content=f"达到最大工具调用轮次 ({self.config.max_tool_iterations})，请简化问题后重试。",
        )

    async def _handle_streaming(
        self, request: LlmRequest
    ) -> tuple[LlmResponse, list[StreamEvent]]:
        """Consume the stream, yielding text_delta events and assembling the final LlmResponse."""
        events: list[StreamEvent] = []
        chunks: list[LlmStreamChunk] = []
        content_parts: list[str] = []

        async for chunk in self.llm.stream_request(request):
            chunks.append(chunk)
            if chunk.delta_content:
                content_parts.append(chunk.delta_content)
                events.append(
                    StreamEvent(type=StreamEventType.TEXT_DELTA, content=chunk.delta_content)
                )

        full_content = "".join(content_parts) or None
        tool_calls = _accumulate_tool_calls(chunks)

        return LlmResponse(content=full_content, tool_calls=tool_calls), events

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
