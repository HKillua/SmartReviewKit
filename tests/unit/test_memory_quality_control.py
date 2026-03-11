from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_conflicting_preferences_do_not_update_profile(tmp_path) -> None:
    from src.agent.memory.enhancer import MemoryRecordHook
    from src.agent.memory.session_memory import SessionMemory
    from src.agent.memory.student_profile import StudentProfileMemory
    from src.agent.types import Conversation, Message

    profile_mem = StudentProfileMemory(str(tmp_path))
    session_mem = SessionMemory(str(tmp_path))
    hook = MemoryRecordHook(
        student_profile=profile_mem,
        session_memory=session_mem,
        extraction_mode="rule",
        write_gating_enabled=True,
        preference_write_min_confidence=0.65,
        preference_conflict_guard=True,
    )
    conversation = Conversation(
        id="conv_conflict",
        user_id="u1",
        messages=[
            Message(role="user", content="请简洁一点讲 TCP 三次握手。"),
            Message(role="assistant", content="好的。"),
            Message(role="user", content="还是详细一点讲 TCP 三次握手的过程和原因？"),
        ],
    )

    await hook.after_message(conversation)

    profile = await profile_mem.get_profile("u1")
    sessions = await session_mem.get_recent_sessions("u1")

    assert profile.preferences["detail_level"] == "normal"
    assert len(sessions) == 1
    assert "detail_level" in sessions[0].extraction_metadata["preference_conflicts"]
    assert sessions[0].extraction_metadata["write_decisions"]["profile_preferences_updated"] is False
    assert (
        sessions[0].extraction_metadata["write_decisions"]["profile_preferences_reason"]
        == "preference_conflict"
    )


@pytest.mark.asyncio
async def test_high_signal_conversation_updates_preferences_and_metadata(tmp_path) -> None:
    from src.agent.memory.enhancer import MemoryRecordHook
    from src.agent.memory.session_memory import SessionMemory
    from src.agent.memory.student_profile import StudentProfileMemory
    from src.agent.types import Conversation, Message

    profile_mem = StudentProfileMemory(str(tmp_path))
    session_mem = SessionMemory(str(tmp_path))
    hook = MemoryRecordHook(
        student_profile=profile_mem,
        session_memory=session_mem,
        extraction_mode="rule",
        write_gating_enabled=True,
        preference_write_min_confidence=0.65,
    )
    conversation = Conversation(
        id="conv_pref",
        user_id="u1",
        messages=[
            Message(role="user", content="请简洁一点总结 TCP 三次握手和四次挥手？"),
            Message(role="assistant", content="可以，我会按考点方式简要总结。"),
        ],
    )

    await hook.after_message(conversation)

    profile = await profile_mem.get_profile("u1")
    sessions = await session_mem.get_recent_sessions("u1")

    assert profile.preferences["detail_level"] == "concise"
    assert profile.total_sessions == 1
    assert len(sessions) == 1
    assert sessions[0].extraction_metadata["confidence"] >= 0.65
    assert sessions[0].extraction_metadata["write_decisions"]["profile_preferences_updated"] is True
    assert sessions[0].extraction_metadata["mode"] == "rule"


@pytest.mark.asyncio
async def test_low_signal_conversation_skips_session_write(tmp_path) -> None:
    from src.agent.memory.enhancer import MemoryRecordHook
    from src.agent.memory.session_memory import SessionMemory
    from src.agent.memory.student_profile import StudentProfileMemory
    from src.agent.types import Conversation, Message

    profile_mem = StudentProfileMemory(str(tmp_path))
    session_mem = SessionMemory(str(tmp_path))
    hook = MemoryRecordHook(
        student_profile=profile_mem,
        session_memory=session_mem,
        extraction_mode="rule",
        write_gating_enabled=True,
    )
    conversation = Conversation(
        id="conv_low_signal",
        user_id="u1",
        messages=[
            Message(role="user", content="好的"),
            Message(role="assistant", content="好的"),
        ],
    )

    await hook.after_message(conversation)

    profile = await profile_mem.get_profile("u1")
    sessions = await session_mem.get_recent_sessions("u1")

    assert profile.total_sessions == 1
    assert sessions == []


@pytest.mark.asyncio
async def test_llm_fallback_records_actual_rule_mode(tmp_path) -> None:
    from src.agent.memory.enhancer import MemoryRecordHook
    from src.agent.memory.session_memory import SessionMemory
    from src.agent.types import Conversation, Message

    session_mem = SessionMemory(str(tmp_path))
    llm = AsyncMock()
    llm.send_request = AsyncMock(side_effect=RuntimeError("boom"))
    hook = MemoryRecordHook(
        session_memory=session_mem,
        llm_service=llm,
        extraction_mode="both",
        write_gating_enabled=True,
    )
    conversation = Conversation(
        id="conv_fallback",
        user_id="u1",
        messages=[
            Message(role="user", content="请简洁一点讲 TCP 三次握手？"),
            Message(role="assistant", content="当然可以。"),
        ],
    )

    await hook.after_message(conversation)

    sessions = await session_mem.get_recent_sessions("u1")
    assert len(sessions) == 1
    assert sessions[0].extraction_metadata["mode"] == "rule"


@pytest.mark.asyncio
async def test_session_summary_roundtrip_keeps_extraction_metadata(tmp_path) -> None:
    from src.agent.memory.session_memory import SessionMemory, SessionSummary

    mem = SessionMemory(str(tmp_path))
    summary = SessionSummary(
        session_id="sess_meta",
        topics=["TCP"],
        summary_text="Reviewed TCP",
        extraction_metadata={
            "mode": "rule",
            "confidence": 0.72,
            "signal_counts": {"topics": 1},
            "preference_conflicts": [],
            "write_decisions": {"session_saved": True},
        },
    )

    await mem.save_session("u1", summary)
    sessions = await mem.get_recent_sessions("u1")

    assert len(sessions) == 1
    assert sessions[0].extraction_metadata["mode"] == "rule"
    assert sessions[0].extraction_metadata["write_decisions"]["session_saved"] is True
