"""Phase S: Agent 模块深度优化 — 5 end-to-end tests."""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestE2EAgentBackgroundLifecycle:
    """E2E: Agent background task full lifecycle (create → track → flush)."""

    @pytest.mark.asyncio
    async def test_background_task_lifecycle(self):
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

        results = []

        async def bg_work(label: str):
            await asyncio.sleep(0.01)
            results.append(label)

        for i in range(3):
            task = asyncio.create_task(bg_work(f"task-{i}"))
            agent._bg_tasks.add(task)
            task.add_done_callback(agent._on_bg_task_done)

        assert len(agent._bg_tasks) <= 3

        await agent.flush()
        assert len(results) == 3
        assert len(agent._bg_tasks) == 0


class TestE2EMemoryCorruptionRecovery:
    """E2E: All memory stores should survive corrupted DB rows gracefully."""

    @pytest.mark.asyncio
    async def test_mixed_valid_and_corrupt_data(self):
        from src.agent.memory.student_profile import StudentProfileMemory
        from src.agent.memory.knowledge_map import KnowledgeMapMemory
        from src.agent.memory.error_memory import ErrorMemory, ErrorRecord

        with tempfile.TemporaryDirectory() as td:
            profile_mem = StudentProfileMemory(db_dir=td)
            kmap_mem = KnowledgeMapMemory(db_dir=td)
            error_mem = ErrorMemory(db_dir=td)

            await profile_mem.update_profile("u1", {"notes": "valid"})
            db = await profile_mem._get_conn()
            await db.execute(
                "INSERT OR REPLACE INTO student_profiles (user_id, data, updated_at) VALUES (?, ?, ?)",
                ("u2", "CORRUPT_DATA", datetime.now().isoformat()),
            )
            await db.commit()

            p1 = await profile_mem.get_profile("u1")
            assert p1.notes == "valid"

            p2 = await profile_mem.get_profile("u2")
            assert p2.user_id == "u2"
            assert p2.total_sessions == 0

            await kmap_mem.update_mastery("u1", "TCP", correct=True)
            kdb = await kmap_mem._get_conn()
            await kdb.execute(
                "INSERT OR REPLACE INTO knowledge_nodes (user_id, concept, data) VALUES (?, ?, ?)",
                ("u1", "BAD_NODE", "{invalid}"),
            )
            await kdb.commit()

            nodes = await kmap_mem._get_all_nodes("u1")
            assert len(nodes) == 1
            assert nodes[0].concept == "TCP"

            await error_mem.add_error("u1", ErrorRecord(
                topic="UDP", question="Q1", concepts=["UDP"],
            ))
            edb = await error_mem._get_conn()
            await edb.execute(
                "INSERT INTO error_records (id, user_id, data, mastered, created_at) VALUES (?, ?, ?, ?, ?)",
                ("bad1", "u1", "CORRUPT", 0, datetime.now().isoformat()),
            )
            await edb.commit()

            errors = await error_mem.get_errors("u1")
            assert len(errors) == 1
            assert errors[0].topic == "UDP"

            await profile_mem.close()
            await kmap_mem.close()
            await error_mem.close()


class TestE2EFeedbackStoreAsync:
    """E2E: FeedbackStore full async lifecycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        from src.agent.memory.feedback_store import FeedbackStore

        with tempfile.TemporaryDirectory() as td:
            store = FeedbackStore(db_path=os.path.join(td, "fb.db"))

            ids = []
            for i in range(5):
                rid = await store.add(
                    f"user_{i % 2}", f"conv_{i}", "up" if i % 2 == 0 else "down",
                    comment=f"feedback {i}",
                )
                ids.append(rid)

            assert len(ids) == 5

            recent = await store.list_recent(limit=3)
            assert len(recent) == 3

            stats = await store.stats()
            assert stats["total"] == 5
            assert stats["up"] == 3
            assert stats["down"] == 2

            await store.close()

            store2 = FeedbackStore(db_path=os.path.join(td, "fb.db"))
            stats2 = await store2.stats()
            assert stats2["total"] == 5
            await store2.close()


class TestE2EConcurrentConversationWrite:
    """E2E: Concurrent writes to the same conversation should be safe."""

    @pytest.mark.asyncio
    async def test_concurrent_writes(self):
        from src.agent.conversation import FileConversationStore
        from src.agent.types import Message

        with tempfile.TemporaryDirectory() as td:
            store = FileConversationStore(base_dir=td)
            conv = await store.create("u1")

            async def write_messages(start: int, count: int):
                for i in range(count):
                    conv.messages.append(
                        Message(role="user", content=f"msg-{start + i}")
                    )
                    await store.update(conv)
                    await asyncio.sleep(0.001)

            await asyncio.gather(
                write_messages(0, 5),
                write_messages(100, 5),
            )

            loaded = await store.get(conv.id, "u1")
            assert loaded is not None
            assert len(loaded.messages) > 0


class TestE2EToolAsyncToThread:
    """E2E: Tools with sync search should not block the event loop."""

    @pytest.mark.asyncio
    async def test_quiz_and_review_parallel(self):
        from src.agent.tools.quiz_generator import QuizGeneratorTool, QuizGeneratorArgs
        from src.agent.tools.review_summary import ReviewSummaryTool, ReviewSummaryArgs
        from src.agent.types import ToolContext

        mock_search = MagicMock()
        mock_result = MagicMock(text="TCP is a transport protocol", metadata={"source_path": "test.pdf"}, chunk_id="c1", score=0.9)
        mock_search.search = MagicMock(return_value=[mock_result])

        mock_llm = AsyncMock()
        mock_llm.send_request = AsyncMock(return_value=MagicMock(
            content='[{"question": "Q?", "answer": "A", "explanation": "E", "concepts": ["TCP"]}]',
            error=None,
        ))

        quiz_tool = QuizGeneratorTool(hybrid_search=mock_search, llm_service=mock_llm)
        review_tool = ReviewSummaryTool(hybrid_search=mock_search, llm_service=mock_llm)

        ctx = ToolContext(user_id="u1", conversation_id="c1")

        results = await asyncio.gather(
            quiz_tool.execute(ctx, QuizGeneratorArgs(topic="TCP")),
            review_tool.execute(ctx, ReviewSummaryArgs(topic="TCP")),
        )

        assert all(r.success for r in results)
        assert mock_search.search.call_count == 2
