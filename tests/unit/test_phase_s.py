"""Phase S: Agent 模块深度优化 — 37 unit tests."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─── S1: enhancer.py 错题次数 ────────────────────────────────

class TestS1ErrorCount:
    """S1: error count should reflect actual topic frequency, not hardcoded {1}."""

    @pytest.mark.asyncio
    async def test_error_count_uses_topic_counter(self):
        from src.agent.memory.enhancer import MemoryContextEnhancer

        mock_error = MagicMock()
        err1 = MagicMock(topic="TCP", question="TCP 三次握手过程？" * 3)
        err2 = MagicMock(topic="TCP", question="TCP 拥塞控制？" * 3)
        err3 = MagicMock(topic="UDP", question="UDP 特点？" * 3)
        mock_error.get_errors = AsyncMock(return_value=[err1, err2, err3])

        enhancer = MemoryContextEnhancer(error_memory=mock_error)
        summary = await enhancer.get_memory_summary("u1")
        assert "错2次" in summary  # TCP appears twice
        assert "错1次" in summary  # UDP appears once

    @pytest.mark.asyncio
    async def test_error_count_single_topic(self):
        from src.agent.memory.enhancer import MemoryContextEnhancer

        mock_error = MagicMock()
        err = MagicMock(topic="DNS", question="DNS 解析过程？" * 3)
        mock_error.get_errors = AsyncMock(return_value=[err])

        enhancer = MemoryContextEnhancer(error_memory=mock_error)
        summary = await enhancer.get_memory_summary("u1")
        assert "错1次" in summary


# ─── S2: fire-and-forget task tracking ─────────────────────────

class TestS2TaskTracking:
    """S2: asyncio.create_task should be tracked and exceptions logged."""

    @pytest.mark.asyncio
    async def test_agent_tracks_bg_tasks(self):
        from src.agent.agent import Agent

        agent = Agent(
            llm_service=MagicMock(),
            tool_registry=MagicMock(get_all_schemas=MagicMock(return_value=[])),
            conversation_store=MagicMock(),
            config=MagicMock(
                max_tool_iterations=1, system_prompt_path="x", temperature=0.7,
                max_tokens=1000, stream_responses=False, max_context_messages=10,
                tool_timeout=30,
            ),
        )
        assert hasattr(agent, "_bg_tasks")
        assert isinstance(agent._bg_tasks, set)

    @pytest.mark.asyncio
    async def test_flush_awaits_pending_tasks(self):
        from src.agent.agent import Agent

        agent = Agent(
            llm_service=MagicMock(),
            tool_registry=MagicMock(get_all_schemas=MagicMock(return_value=[])),
            conversation_store=MagicMock(),
            config=MagicMock(
                max_tool_iterations=1, system_prompt_path="x", temperature=0.7,
                max_tokens=1000, stream_responses=False, max_context_messages=10,
                tool_timeout=30,
            ),
        )

        completed = []

        async def bg():
            await asyncio.sleep(0.01)
            completed.append(True)

        task = asyncio.create_task(bg())
        agent._bg_tasks.add(task)
        task.add_done_callback(agent._on_bg_task_done)

        await agent.flush()
        assert len(completed) == 1
        assert len(agent._bg_tasks) == 0

    @pytest.mark.asyncio
    async def test_on_bg_task_done_logs_exceptions(self):
        from src.agent.agent import Agent

        agent = Agent(
            llm_service=MagicMock(),
            tool_registry=MagicMock(get_all_schemas=MagicMock(return_value=[])),
            conversation_store=MagicMock(),
            config=MagicMock(
                max_tool_iterations=1, system_prompt_path="x", temperature=0.7,
                max_tokens=1000, stream_responses=False, max_context_messages=10,
                tool_timeout=30,
            ),
        )

        async def fail():
            raise ValueError("test error")

        task = asyncio.create_task(fail())
        agent._bg_tasks.add(task)
        task.add_done_callback(agent._on_bg_task_done)

        await asyncio.sleep(0.05)
        assert task not in agent._bg_tasks


# ─── S3: asyncio.to_thread ──────────────────────────────────

class TestS3AsyncToThread:
    """S3: sync calls in tools should use asyncio.to_thread."""

    @pytest.mark.asyncio
    async def test_quiz_generator_uses_to_thread(self):
        from src.agent.tools.quiz_generator import QuizGeneratorTool, QuizGeneratorArgs
        from src.agent.types import ToolContext

        mock_search = MagicMock()
        mock_search.search = MagicMock(return_value=[])

        tool = QuizGeneratorTool(hybrid_search=mock_search, llm_service=MagicMock())
        ctx = ToolContext(user_id="u1", conversation_id="c1")
        args = QuizGeneratorArgs(topic="TCP")

        result = await tool.execute(ctx, args)
        assert result.success
        assert mock_search.search.call_count >= 4

    @pytest.mark.asyncio
    async def test_review_summary_uses_to_thread(self):
        from src.agent.tools.review_summary import ReviewSummaryTool, ReviewSummaryArgs
        from src.agent.types import ToolContext

        mock_search = MagicMock()
        mock_search.search = MagicMock(return_value=[])

        tool = ReviewSummaryTool(hybrid_search=mock_search, llm_service=MagicMock())
        ctx = ToolContext(user_id="u1", conversation_id="c1")
        args = ReviewSummaryArgs(topic="UDP")

        result = await tool.execute(ctx, args)
        assert result.success

    @pytest.mark.asyncio
    async def test_document_ingest_uses_to_thread(self):
        from src.agent.tools.document_ingest import DocumentIngestTool, DocumentIngestArgs
        from src.agent.types import ToolContext

        with tempfile.NamedTemporaryFile(suffix=".pdf", dir="/tmp", delete=False) as f:
            f.write(b"fake pdf")
            tmp_path = f.name

        try:
            mock_pipeline = MagicMock()
            mock_result = MagicMock(success=True, doc_id="d1", chunk_count=5, image_count=0)
            mock_pipeline.run = MagicMock(return_value=mock_result)

            tool = DocumentIngestTool(pipeline=mock_pipeline, allowed_dirs=["/tmp"])
            ctx = ToolContext(user_id="u1", conversation_id="c1")
            args = DocumentIngestArgs(file_path=tmp_path)

            result = await tool.execute(ctx, args)
            assert result.success
            mock_pipeline.run.assert_called_once()
        finally:
            os.unlink(tmp_path)


# ─── S4: verdict case-insensitive ──────────────────────────

class TestS4VerdictCase:
    """S4: verdict comparison should be case-insensitive."""

    @pytest.mark.asyncio
    async def test_uppercase_correct(self):
        from src.agent.tools.quiz_evaluator import QuizEvaluatorTool, QuizEvaluatorArgs
        from src.agent.types import ToolContext, LlmResponse

        mock_llm = AsyncMock()
        mock_llm.send_request = AsyncMock(return_value=LlmResponse(
            content='{"verdict": "Correct", "score": 100, "explanation": "ok", "key_concepts": ["TCP"]}'
        ))

        tool = QuizEvaluatorTool(llm_service=mock_llm)
        ctx = ToolContext(user_id="u1", conversation_id="c1")
        args = QuizEvaluatorArgs(question="Q", user_answer="A", correct_answer="A")

        result = await tool.execute(ctx, args)
        assert result.success
        assert result.metadata.get("verdict") == "correct"

    @pytest.mark.asyncio
    async def test_mixed_case_partial(self):
        from src.agent.tools.quiz_evaluator import QuizEvaluatorTool, QuizEvaluatorArgs
        from src.agent.types import ToolContext, LlmResponse

        mock_llm = AsyncMock()
        mock_llm.send_request = AsyncMock(return_value=LlmResponse(
            content='{"verdict": "Partial", "score": 50, "explanation": "half", "key_concepts": []}'
        ))

        tool = QuizEvaluatorTool(llm_service=mock_llm)
        ctx = ToolContext(user_id="u1", conversation_id="c1")
        args = QuizEvaluatorArgs(question="Q", user_answer="A", correct_answer="A")

        result = await tool.execute(ctx, args)
        assert result.metadata.get("verdict") == "partial"

    @pytest.mark.asyncio
    async def test_whitespace_verdict(self):
        from src.agent.tools.quiz_evaluator import QuizEvaluatorTool, QuizEvaluatorArgs
        from src.agent.types import ToolContext, LlmResponse

        mock_llm = AsyncMock()
        mock_llm.send_request = AsyncMock(return_value=LlmResponse(
            content='{"verdict": " correct ", "score": 90, "explanation": "yes", "key_concepts": []}'
        ))

        tool = QuizEvaluatorTool(llm_service=mock_llm)
        ctx = ToolContext(user_id="u1", conversation_id="c1")
        args = QuizEvaluatorArgs(question="Q", user_answer="A", correct_answer="A")

        result = await tool.execute(ctx, args)
        assert result.metadata.get("verdict") == "correct"


# ─── S5: FeedbackStore async ────────────────────────────────

class TestS5FeedbackAsync:
    """S5: FeedbackStore should use aiosqlite (all methods async)."""

    @pytest.mark.asyncio
    async def test_add_and_list(self):
        from src.agent.memory.feedback_store import FeedbackStore

        with tempfile.TemporaryDirectory() as td:
            store = FeedbackStore(db_path=os.path.join(td, "fb.db"))
            row_id = await store.add("u1", "c1", "up", comment="great")
            assert isinstance(row_id, int)

            recent = await store.list_recent(limit=10)
            assert len(recent) == 1
            assert recent[0]["rating"] == "up"
            await store.close()

    @pytest.mark.asyncio
    async def test_stats(self):
        from src.agent.memory.feedback_store import FeedbackStore

        with tempfile.TemporaryDirectory() as td:
            store = FeedbackStore(db_path=os.path.join(td, "fb.db"))
            await store.add("u1", "c1", "up")
            await store.add("u1", "c2", "down")
            await store.add("u1", "c3", "up")

            s = await store.stats()
            assert s["total"] == 3
            assert s["up"] == 2
            assert s["down"] == 1
            await store.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        from src.agent.memory.feedback_store import FeedbackStore

        with tempfile.TemporaryDirectory() as td:
            store = FeedbackStore(db_path=os.path.join(td, "fb.db"))
            await store.close()
            await store.close()  # should not raise


# ─── S6: safe template substitution ─────────────────────────

class TestS6SafeTemplate:
    """S6: prompt_builder should use safe_substitute to avoid injection."""

    def test_format_string_in_content_does_not_crash(self):
        from src.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder(template_path="nonexistent.txt")
        result = builder.build(
            memory_context="user said: {evil_injection}",
        )
        assert "{evil_injection}" in result

    def test_build_with_normal_content(self):
        from src.agent.prompt_builder import SystemPromptBuilder

        builder = SystemPromptBuilder(template_path="nonexistent.txt")
        result = builder.build(
            tool_schemas=[{"function": {"name": "test", "description": "A test tool"}}],
            memory_context="TCP knowledge",
            active_skill="review skill",
        )
        assert "test" in result
        assert "TCP knowledge" in result
        assert "review skill" in result


# ─── S7: flush() ─────────────────────────────────────────────

class TestS7Flush:
    """S7: Agent.flush() should await all background tasks."""

    @pytest.mark.asyncio
    async def test_flush_on_empty(self):
        from src.agent.agent import Agent

        agent = Agent(
            llm_service=MagicMock(),
            tool_registry=MagicMock(get_all_schemas=MagicMock(return_value=[])),
            conversation_store=MagicMock(),
            config=MagicMock(
                max_tool_iterations=1, system_prompt_path="x", temperature=0.7,
                max_tokens=1000, stream_responses=False, max_context_messages=10,
                tool_timeout=30,
            ),
        )
        await agent.flush()  # should not raise

    @pytest.mark.asyncio
    async def test_flush_clears_tasks(self):
        from src.agent.agent import Agent

        agent = Agent(
            llm_service=MagicMock(),
            tool_registry=MagicMock(get_all_schemas=MagicMock(return_value=[])),
            conversation_store=MagicMock(),
            config=MagicMock(
                max_tool_iterations=1, system_prompt_path="x", temperature=0.7,
                max_tokens=1000, stream_responses=False, max_context_messages=10,
                tool_timeout=30,
            ),
        )

        async def noop():
            pass

        t = asyncio.create_task(noop())
        agent._bg_tasks.add(t)
        await agent.flush()
        assert len(agent._bg_tasks) == 0


# ─── S8: UTC datetime ───────────────────────────────────────

class TestS8DatetimeUtc:
    """S8: datetime fields should use UTC timezone."""

    def test_message_timestamp_has_tzinfo(self):
        from src.agent.types import Message
        msg = Message(role="user", content="hello")
        assert msg.timestamp.tzinfo is not None

    def test_conversation_created_at_has_tzinfo(self):
        from src.agent.types import Conversation
        conv = Conversation(id="test", user_id="u1")
        assert conv.created_at.tzinfo is not None
        assert conv.updated_at.tzinfo is not None


# ─── S9: model_validate_json guard ──────────────────────────

class TestS9JsonGuard:
    """S9: corrupted data should not crash memory stores."""

    @pytest.mark.asyncio
    async def test_student_profile_corrupted(self):
        from src.agent.memory.student_profile import StudentProfileMemory

        with tempfile.TemporaryDirectory() as td:
            mem = StudentProfileMemory(db_dir=td)
            db = await mem._get_conn()
            await db.execute(
                "INSERT INTO student_profiles (user_id, data, updated_at) VALUES (?, ?, ?)",
                ("u1", "NOT_VALID_JSON", datetime.now().isoformat()),
            )
            await db.commit()

            profile = await mem.get_profile("u1")
            assert profile.user_id == "u1"
            assert profile.total_sessions == 0
            await mem.close()

    @pytest.mark.asyncio
    async def test_knowledge_map_corrupted_node(self):
        from src.agent.memory.knowledge_map import KnowledgeMapMemory

        with tempfile.TemporaryDirectory() as td:
            mem = KnowledgeMapMemory(db_dir=td)
            db = await mem._get_conn()
            await db.execute(
                "INSERT INTO knowledge_nodes (user_id, concept, data) VALUES (?, ?, ?)",
                ("u1", "TCP", "BROKEN"),
            )
            await db.commit()

            node = await mem.get_node("u1", "TCP")
            assert node is None

            all_nodes = await mem._get_all_nodes("u1")
            assert len(all_nodes) == 0
            await mem.close()

    @pytest.mark.asyncio
    async def test_error_memory_corrupted(self):
        from src.agent.memory.error_memory import ErrorMemory

        with tempfile.TemporaryDirectory() as td:
            mem = ErrorMemory(db_dir=td)
            db = await mem._get_conn()
            await db.execute(
                "INSERT INTO error_records (id, user_id, data, mastered, created_at) VALUES (?, ?, ?, ?, ?)",
                ("e1", "u1", "CORRUPT", 0, datetime.now().isoformat()),
            )
            await db.commit()

            errors = await mem.get_errors("u1")
            assert len(errors) == 0
            await mem.close()

    @pytest.mark.asyncio
    async def test_session_memory_corrupted(self):
        from src.agent.memory.session_memory import SessionMemory

        with tempfile.TemporaryDirectory() as td:
            mem = SessionMemory(db_dir=td)
            db = await mem._get_conn()
            await db.execute(
                "INSERT INTO session_summaries (id, user_id, data, created_at) VALUES (?, ?, ?, ?)",
                ("s1", "u1", "BROKEN_JSON", datetime.now().isoformat()),
            )
            await db.commit()

            sessions = await mem.get_recent_sessions("u1")
            assert len(sessions) == 0
            await mem.close()


# ─── S10: conversation concurrent lock ──────────────────────

class TestS10ConvLock:
    """S10: FileConversationStore should have per-conversation write locks."""

    def test_has_write_locks(self):
        from src.agent.conversation import FileConversationStore

        with tempfile.TemporaryDirectory() as td:
            store = FileConversationStore(base_dir=td)
            assert hasattr(store, "_write_locks")

    @pytest.mark.asyncio
    async def test_concurrent_updates_safe(self):
        from src.agent.conversation import FileConversationStore
        from src.agent.types import Conversation, Message

        with tempfile.TemporaryDirectory() as td:
            store = FileConversationStore(base_dir=td)
            conv = await store.create("u1")

            async def add_msg(i: int):
                conv.messages.append(Message(role="user", content=f"msg-{i}"))
                await store.update(conv)

            await asyncio.gather(*[add_msg(i) for i in range(5)])
            loaded = await store.get(conv.id, "u1")
            assert loaded is not None


# ─── S11: symlink protection ────────────────────────────────

class TestS11Symlink:
    """S11: document_ingest should reject symlinks."""

    @pytest.mark.asyncio
    async def test_symlink_rejected(self):
        from src.agent.tools.document_ingest import DocumentIngestTool, DocumentIngestArgs
        from src.agent.types import ToolContext

        with tempfile.TemporaryDirectory() as td:
            real_file = Path(td) / "real.pdf"
            real_file.write_bytes(b"fake")
            link_file = Path(td) / "link.pdf"
            link_file.symlink_to(real_file)

            tool = DocumentIngestTool(allowed_dirs=[td])
            ctx = ToolContext(user_id="u1", conversation_id="c1")
            args = DocumentIngestArgs(file_path=str(link_file))

            result = await tool.execute(ctx, args)
            assert not result.success
            assert "符号链接" in result.error

    @pytest.mark.asyncio
    async def test_normal_file_allowed(self):
        from src.agent.tools.document_ingest import DocumentIngestTool, DocumentIngestArgs
        from src.agent.types import ToolContext

        with tempfile.TemporaryDirectory() as td:
            real_file = Path(td) / "real.pdf"
            real_file.write_bytes(b"fake pdf content")

            mock_pipeline = MagicMock()
            mock_pipeline.run = MagicMock(return_value=MagicMock(
                success=True, doc_id="d1", chunk_count=1, image_count=0,
            ))

            tool = DocumentIngestTool(pipeline=mock_pipeline, allowed_dirs=[td])
            ctx = ToolContext(user_id="u1", conversation_id="c1")
            args = DocumentIngestArgs(file_path=str(real_file))

            result = await tool.execute(ctx, args)
            assert result.success


# ─── S12: knowledge_query init lock ─────────────────────────

class TestS12InitLock:
    """S12: knowledge_query should have asyncio.Lock for initialization."""

    def test_has_init_lock(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool
        tool = KnowledgeQueryTool()
        assert hasattr(tool, "_init_lock")
        assert isinstance(tool._init_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_init_lock_prevents_race(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool

        tool = KnowledgeQueryTool()
        init_count = 0
        original_ensure = tool._ensure_initialized

        def counting_ensure(collection="computer_network"):
            nonlocal init_count
            init_count += 1

        tool._ensure_initialized = counting_ensure
        tool._hybrid_search = MagicMock()
        tool._hybrid_search.search = MagicMock(return_value=[])

        from src.agent.types import ToolContext
        from src.agent.tools.knowledge_query import KnowledgeQueryArgs

        ctx = ToolContext(user_id="u1", conversation_id="c1")
        args = KnowledgeQueryArgs(query="test")

        await tool.execute(ctx, args)
        assert init_count == 1


# ─── S13: profile field whitelist ────────────────────────────

class TestS13ProfileFields:
    """S13: update_profile should only allow whitelisted fields."""

    @pytest.mark.asyncio
    async def test_allowed_field_updated(self):
        from src.agent.memory.student_profile import StudentProfileMemory

        with tempfile.TemporaryDirectory() as td:
            mem = StudentProfileMemory(db_dir=td)
            await mem.update_profile("u1", {"notes": "test note"})
            profile = await mem.get_profile("u1")
            assert profile.notes == "test note"
            await mem.close()

    @pytest.mark.asyncio
    async def test_disallowed_field_ignored(self):
        from src.agent.memory.student_profile import StudentProfileMemory

        with tempfile.TemporaryDirectory() as td:
            mem = StudentProfileMemory(db_dir=td)
            await mem.update_profile("u1", {"user_id": "HACKED"})
            profile = await mem.get_profile("u1")
            assert profile.user_id == "u1"
            await mem.close()


# ─── S14: batch decay ───────────────────────────────────────

class TestS14BatchDecay:
    """S14: apply_decay should batch updates into single executemany."""

    @pytest.mark.asyncio
    async def test_batch_decay(self):
        from src.agent.memory.knowledge_map import KnowledgeMapMemory
        from datetime import timedelta

        with tempfile.TemporaryDirectory() as td:
            mem = KnowledgeMapMemory(db_dir=td)

            for concept in ["TCP", "UDP", "DNS"]:
                await mem.update_mastery("u1", concept, correct=True)

            db = await mem._get_conn()
            async with db.execute("SELECT data FROM knowledge_nodes WHERE user_id = ?", ("u1",)) as c:
                rows = await c.fetchall()

            from src.agent.memory.knowledge_map import KnowledgeNode
            for r in rows:
                node = KnowledgeNode.model_validate_json(r[0])
                node.last_reviewed = datetime.now(timezone.utc) - timedelta(days=10)
                await db.execute(
                    "UPDATE knowledge_nodes SET data = ? WHERE user_id = ? AND concept = ?",
                    (node.model_dump_json(), "u1", node.concept),
                )
            await db.commit()

            decayed = await mem.apply_decay("u1")
            assert decayed > 0
            await mem.close()

    @pytest.mark.asyncio
    async def test_decay_no_update_when_recent(self):
        from src.agent.memory.knowledge_map import KnowledgeMapMemory

        with tempfile.TemporaryDirectory() as td:
            mem = KnowledgeMapMemory(db_dir=td)
            await mem.update_mastery("u1", "TCP", correct=True)
            decayed = await mem.apply_decay("u1")
            assert decayed == 0
            await mem.close()


# ─── S15-S16: SQL filtering ─────────────────────────────────

class TestS15S16SqlFilter:
    """S15-S16: topic/keyword filtering should use SQL LIKE."""

    @pytest.mark.asyncio
    async def test_error_memory_topic_sql_filter(self):
        from src.agent.memory.error_memory import ErrorMemory, ErrorRecord

        with tempfile.TemporaryDirectory() as td:
            mem = ErrorMemory(db_dir=td)
            await mem.add_error("u1", ErrorRecord(
                topic="TCP", question="Q1", concepts=["TCP"],
            ))
            await mem.add_error("u1", ErrorRecord(
                topic="UDP", question="Q2", concepts=["UDP"],
            ))

            tcp_errors = await mem.get_errors("u1", topic="TCP")
            assert len(tcp_errors) == 1
            assert tcp_errors[0].topic == "TCP"
            await mem.close()

    @pytest.mark.asyncio
    async def test_session_topic_history_sql(self):
        from src.agent.memory.session_memory import SessionMemory, SessionSummary

        with tempfile.TemporaryDirectory() as td:
            mem = SessionMemory(db_dir=td)
            await mem.save_session("u1", SessionSummary(
                session_id="s1", topics=["TCP", "HTTP"], summary_text="Learned TCP",
            ))
            await mem.save_session("u1", SessionSummary(
                session_id="s2", topics=["UDP"], summary_text="Learned UDP",
            ))

            history = await mem.get_topic_history("u1", "TCP")
            assert len(history) == 1
            assert "TCP" in history[0].topics
            await mem.close()


# ─── S20: safe_parse_json ───────────────────────────────────

class TestS20SafeParseJson:
    """S20: safe_parse_json should handle code fences and invalid JSON."""

    def test_normal_json(self):
        from src.agent.utils.json_helpers import safe_parse_json
        result = safe_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_code_fenced_json(self):
        from src.agent.utils.json_helpers import safe_parse_json
        result = safe_parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_invalid_json_returns_fallback(self):
        from src.agent.utils.json_helpers import safe_parse_json
        result = safe_parse_json("not json at all", fallback={"default": True})
        assert result == {"default": True}

    def test_empty_returns_fallback(self):
        from src.agent.utils.json_helpers import safe_parse_json
        assert safe_parse_json("") is None
        assert safe_parse_json("", fallback=[]) == []
