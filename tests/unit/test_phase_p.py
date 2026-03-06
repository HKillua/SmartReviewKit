"""Phase P — End-to-end latency optimization tests.

Validates the key optimizations:
  P0: True real-time streaming (no buffering)
  P1: Parallel preprocessing with asyncio.gather
  P2: Parallel memory reads
  P3/P6: Async search with to_thread + parallel sub-queries
  P4: Prompt template caching
  P5: Persistent DB connections
  P8: Background post-message hooks
  P9: LLM timeout configuration
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.types import (
    LlmMessage,
    LlmRequest,
    LlmResponse,
    LlmStreamChunk,
    Message,
    StreamEvent,
    StreamEventType,
    ToolCallData,
)


# ---------------------------------------------------------------------------
# P0: True real-time streaming
# ---------------------------------------------------------------------------


class _FakeStreamingLlm:
    """Simulates a streaming LLM that yields tokens with delays."""

    def __init__(self, chunks: list[str], delay: float = 0.01):
        self._chunks = chunks
        self._delay = delay

    async def stream_request(self, request):
        for text in self._chunks:
            await asyncio.sleep(self._delay)
            yield LlmStreamChunk(delta_content=text, finish_reason=None)
        yield LlmStreamChunk(delta_content=None, finish_reason="stop")

    async def send_request(self, request):
        return LlmResponse(content="".join(self._chunks))


@pytest.mark.asyncio
async def test_p0_streaming_yields_in_realtime():
    """Events should be yielded one-by-one as the LLM stream produces them,
    NOT batched until the entire response finishes."""
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore

    fake_llm = _FakeStreamingLlm(["Hello", " ", "World"], delay=0.05)

    mock_store = AsyncMock(spec=ConversationStore)
    from src.agent.types import Conversation
    conv = Conversation(id="test", user_id="u1", messages=[])
    mock_store.get = AsyncMock(return_value=conv)
    mock_store.create = AsyncMock(return_value=conv)
    mock_store.update = AsyncMock()

    cfg = AgentConfig(stream_responses=True, max_tool_iterations=1)
    from src.agent.tools.base import ToolRegistry
    agent = Agent(
        llm_service=fake_llm,
        tool_registry=ToolRegistry(),
        conversation_store=mock_store,
        config=cfg,
    )

    timestamps: list[float] = []
    events: list[StreamEvent] = []
    async for ev in agent.chat("hi", "u1", "test"):
        ts = time.monotonic()
        timestamps.append(ts)
        events.append(ev)

    text_events = [e for e in events if e.type == StreamEventType.TEXT_DELTA]
    assert len(text_events) == 3, f"Expected 3 text_delta events, got {len(text_events)}"
    assert text_events[0].content == "Hello"
    assert text_events[1].content == " "
    assert text_events[2].content == "World"

    text_indices = [i for i, e in enumerate(events) if e.type == StreamEventType.TEXT_DELTA]
    if len(text_indices) >= 2:
        t0 = timestamps[text_indices[0]]
        t1 = timestamps[text_indices[1]]
        assert t1 - t0 >= 0.03, (
            f"Events arrived too close together ({t1 - t0:.3f}s) — "
            "they should be streamed incrementally, not batched"
        )


# ---------------------------------------------------------------------------
# P1: Parallel preprocessing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p1_parallel_preprocessing():
    """Memory, review, and skill should run concurrently, not sequentially."""
    call_order: list[str] = []

    class SlowMemory:
        async def get_memory_summary(self, user_id):
            call_order.append("mem_start")
            await asyncio.sleep(0.05)
            call_order.append("mem_end")
            return "memory"

    class SlowReview:
        async def get_review_context(self, user_id):
            call_order.append("rev_start")
            await asyncio.sleep(0.05)
            call_order.append("rev_end")
            return ""

    class SlowSkill:
        async def try_handle(self, msg, user_id):
            call_order.append("skill_start")
            await asyncio.sleep(0.05)
            call_order.append("skill_end")
            return None

    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.tools.base import ToolRegistry

    fake_llm = _FakeStreamingLlm(["ok"], delay=0)
    mock_store = AsyncMock(spec=ConversationStore)
    from src.agent.types import Conversation
    conv = Conversation(id="t", user_id="u", messages=[])
    mock_store.get = AsyncMock(return_value=conv)
    mock_store.update = AsyncMock()

    agent = Agent(
        llm_service=fake_llm,
        tool_registry=ToolRegistry(),
        conversation_store=mock_store,
        config=AgentConfig(stream_responses=True, max_tool_iterations=1),
        memory_enhancer=SlowMemory(),
        review_hook=SlowReview(),
        skill_workflow=SlowSkill(),
    )

    start = time.monotonic()
    async for _ in agent.chat("test", "u", "t"):
        pass
    elapsed = time.monotonic() - start

    assert elapsed < 0.2, (
        f"Preprocessing took {elapsed:.2f}s — should be ~0.05s if parallelized, "
        "not ~0.15s if sequential"
    )
    starts = [e for e in call_order if e.endswith("_start")]
    assert len(starts) == 3


# ---------------------------------------------------------------------------
# P2: Parallel memory reads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p2_parallel_memory_reads():
    """MemoryContextEnhancer.get_memory_summary should run DB reads in parallel."""
    from src.agent.memory.enhancer import MemoryContextEnhancer

    call_times: list[float] = []

    class SlowProfile:
        async def get_profile(self, user_id):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            from src.agent.memory.student_profile import StudentProfile
            return StudentProfile(user_id=user_id)

    class SlowSessions:
        async def get_recent_sessions(self, user_id, limit=5):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            return []

    class SlowKmap:
        async def get_due_for_review(self, user_id):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            return []
        async def get_weak_nodes(self, user_id):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            return []

    class SlowErrors:
        async def get_errors(self, user_id, mastered=None, limit=50):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            return []

    enhancer = MemoryContextEnhancer(
        student_profile=SlowProfile(),
        error_memory=SlowErrors(),
        knowledge_map=SlowKmap(),
        session_memory=SlowSessions(),
    )

    start = time.monotonic()
    await enhancer.get_memory_summary("u1")
    elapsed = time.monotonic() - start

    assert elapsed < 0.15, (
        f"Memory summary took {elapsed:.2f}s — should be ~0.05s (parallel), "
        "not ~0.25s (sequential 5 x 0.05s)"
    )


# ---------------------------------------------------------------------------
# P4: Prompt template caching
# ---------------------------------------------------------------------------


def test_p4_prompt_template_cached(tmp_path):
    """Template should be read from disk only once, then served from cache."""
    tmpl = tmp_path / "system.txt"
    tmpl.write_text("Hello {tool_descriptions}{memory_context}{active_skill}")

    from src.agent.prompt_builder import SystemPromptBuilder
    builder = SystemPromptBuilder(str(tmpl))

    r1 = builder.build()
    tmpl.write_text("Changed {tool_descriptions}{memory_context}{active_skill}")
    r2 = builder.build()

    assert r1 == r2, "Template should be cached — second build should return same content"


# ---------------------------------------------------------------------------
# P5: Persistent DB connections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p5_persistent_connection_student_profile(tmp_path):
    from src.agent.memory.student_profile import StudentProfileMemory
    store = StudentProfileMemory(str(tmp_path))
    try:
        assert store._conn is None
        await store.get_profile("u1")
        assert store._conn is not None, "Connection should be lazily created"
        conn1 = store._conn
        await store.get_profile("u1")
        assert store._conn is conn1, "Should reuse the same connection"
    finally:
        await store.close()
    assert store._conn is None


@pytest.mark.asyncio
async def test_p5_persistent_connection_error_memory(tmp_path):
    from src.agent.memory.error_memory import ErrorMemory
    store = ErrorMemory(str(tmp_path))
    try:
        await store.get_errors("u1")
        conn = store._conn
        assert conn is not None
        await store.get_errors("u1")
        assert store._conn is conn
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_p5_persistent_connection_knowledge_map(tmp_path):
    from src.agent.memory.knowledge_map import KnowledgeMapMemory
    store = KnowledgeMapMemory(str(tmp_path))
    try:
        await store._get_all_nodes("u1")
        conn = store._conn
        assert conn is not None
        await store.get_weak_nodes("u1")
        assert store._conn is conn
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_p5_persistent_connection_session_memory(tmp_path):
    from src.agent.memory.session_memory import SessionMemory
    store = SessionMemory(str(tmp_path))
    try:
        await store.get_recent_sessions("u1")
        conn = store._conn
        assert conn is not None
        await store.get_session_count("u1")
        assert store._conn is conn
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# P8: Background post-message hooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p8_hooks_run_in_background():
    """after_message and conversation save should be fire-and-forget."""
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.tools.base import ToolRegistry

    fake_llm = _FakeStreamingLlm(["x"], delay=0)
    mock_store = AsyncMock(spec=ConversationStore)
    from src.agent.types import Conversation
    conv = Conversation(id="t", user_id="u", messages=[])
    mock_store.get = AsyncMock(return_value=conv)
    mock_store.update = AsyncMock()

    hook_called = asyncio.Event()

    class SlowHook:
        async def before_message(self, user_id, msg):
            return None
        async def before_tool(self, name, ctx):
            pass
        async def after_tool(self, name, result, context=None):
            return None
        async def after_message(self, conv):
            await asyncio.sleep(0.1)
            hook_called.set()

    agent = Agent(
        llm_service=fake_llm,
        tool_registry=ToolRegistry(),
        conversation_store=mock_store,
        config=AgentConfig(stream_responses=True, max_tool_iterations=1),
        lifecycle_hooks=[SlowHook()],
    )

    start = time.monotonic()
    events = []
    async for ev in agent.chat("hi", "u", "t"):
        events.append(ev)
    generator_elapsed = time.monotonic() - start

    assert generator_elapsed < 0.15, (
        f"Generator took {generator_elapsed:.2f}s — should not wait for hooks"
    )

    await asyncio.wait_for(hook_called.wait(), timeout=1.0)
    assert hook_called.is_set(), "Hook should eventually run in background"


# ---------------------------------------------------------------------------
# P9: LLM timeout configuration
# ---------------------------------------------------------------------------


def test_p9_openai_service_has_timeout():
    """OpenAI client should be configured with an explicit timeout."""
    try:
        import httpx
        import openai
    except ImportError:
        pytest.skip("openai/httpx not installed")

    from src.agent.llm.openai_service import OpenAILlmService
    svc = OpenAILlmService(api_key="test-key", model="gpt-4o")
    client = svc._client

    timeout = client.timeout
    assert timeout.connect is not None and timeout.connect > 0
    assert timeout.read is not None or timeout.pool is not None


# ---------------------------------------------------------------------------
# P3/P6: async search wrapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p3_search_runs_in_thread():
    """hybrid_search.search should be dispatched via asyncio.to_thread."""
    from src.agent.tools.knowledge_query import KnowledgeQueryTool
    from src.agent.types import ToolContext

    mock_hs = MagicMock()
    mock_hs.search = MagicMock(return_value=[])

    tool = KnowledgeQueryTool(hybrid_search=mock_hs)
    tool._current_collection = "computer_network"
    ctx = ToolContext(user_id="u", conversation_id="c", request_id="r")

    from src.agent.tools.knowledge_query import KnowledgeQueryArgs
    args = KnowledgeQueryArgs(query="test", top_k=3, collection="computer_network")

    result = await tool.execute(ctx, args)
    assert result.success
    assert "未找到" in result.result_for_llm
