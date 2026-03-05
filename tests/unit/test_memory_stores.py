"""Unit tests for memory stores."""

import tempfile
import pytest

from src.agent.memory.student_profile import StudentProfileMemory
from src.agent.memory.error_memory import ErrorMemory, ErrorRecord
from src.agent.memory.knowledge_map import KnowledgeMapMemory
from src.agent.memory.skill_memory import SkillMemory, ToolUsageRecord


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.mark.asyncio
async def test_student_profile(tmp_dir):
    mem = StudentProfileMemory(tmp_dir)
    profile = await mem.get_profile("u1")
    assert profile.user_id == "u1"
    assert profile.total_sessions == 0

    await mem.update_profile("u1", {"total_sessions": 5, "weak_topics": ["SQL"]})
    profile = await mem.get_profile("u1")
    assert profile.total_sessions == 5
    assert "SQL" in profile.weak_topics


@pytest.mark.asyncio
async def test_error_memory(tmp_dir):
    mem = ErrorMemory(tmp_dir)
    record = ErrorRecord(
        question="What is a primary key?",
        topic="Keys",
        concepts=["primary_key", "unique"],
        user_answer="wrong",
        correct_answer="right",
    )
    await mem.add_error("u1", record)
    errors = await mem.get_errors("u1")
    assert len(errors) == 1
    assert errors[0].question == "What is a primary key?"

    weak = await mem.get_weak_concepts("u1")
    assert "primary_key" in weak


@pytest.mark.asyncio
async def test_knowledge_map(tmp_dir):
    mem = KnowledgeMapMemory(tmp_dir)

    await mem.update_mastery("u1", "SQL_JOIN", correct=True)
    node = await mem.get_node("u1", "SQL_JOIN")
    assert node is not None
    assert node.mastery_level == pytest.approx(0.1)
    assert node.quiz_count == 1

    await mem.update_mastery("u1", "SQL_JOIN", correct=False)
    node = await mem.get_node("u1", "SQL_JOIN")
    assert node.mastery_level == pytest.approx(0.0)  # 0.1 - 0.15 clamped to 0

    weak = await mem.get_weak_nodes("u1")
    assert len(weak) == 1


@pytest.mark.asyncio
async def test_skill_memory(tmp_dir):
    mem = SkillMemory(tmp_dir)
    record = ToolUsageRecord(
        question_pattern="帮我复习 SQL 连接",
        tool_chain=["knowledge_query", "review_summary"],
        quality_score=0.9,
    )
    await mem.save_usage("u1", record)
    results = await mem.search_similar("u1", "复习 SQL 连接")
    assert len(results) >= 1
