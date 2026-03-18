from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agent.hooks.review_schedule import ReviewScheduleHook
from src.agent.memory.enhancer import MemoryRecordHook
from src.agent.memory.knowledge_map import KnowledgeMapMemory, KnowledgeNode
from src.agent.memory.student_profile import StudentProfile
from src.agent.pacing import compute_pacing_from_conversation
from src.agent.skills.registry import SkillPolicy
from src.agent.skills.workflow import WorkflowResult
from src.agent.types import Conversation, LlmResponse, Message, ToolCallData, ToolContext, ToolResult


class _ReviewKMap:
    def __init__(self) -> None:
        self.decay_calls = 0

    async def apply_decay(self, user_id: str) -> int:
        self.decay_calls += 1
        return 2

    async def get_decayed_nodes(self, user_id: str, *, threshold: float = 0.45, limit: int = 5):
        return [
            KnowledgeNode(concept="拥塞控制", mastery_level=0.21),
            KnowledgeNode(concept="流量控制", mastery_level=0.38),
        ][:limit]


class _ReviewErrors:
    async def get_weak_concepts(self, user_id: str) -> list[str]:
        return ["拥塞控制", "滑动窗口"]


class _ReviewProfile:
    async def get_profile(self, user_id: str) -> StudentProfile:
        return StudentProfile(
            user_id=user_id,
            last_active=datetime.now(timezone.utc) - timedelta(days=6),
        )


class _PassiveLlm:
    async def send_request(self, request):
        return LlmResponse(content="你好，我可以帮你复习运输层。")


class _AutonomousKnowledgeLlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.calls = 0

    async def send_request(self, request):
        self.calls += 1
        self.prompts.append(request.messages[0].content or "")
        if self.calls == 1:
            return LlmResponse(
                content="先检索证据。",
                tool_calls=[
                    ToolCallData(
                        id="kq1",
                        name="knowledge_query",
                        arguments={"query": "TCP 拥塞控制", "top_k": 3},
                    )
                ],
            )
        if self.calls == 2:
            return LlmResponse(
                content="证据不足，切到图谱看薄弱点。",
                tool_calls=[
                    ToolCallData(
                        id="cg1",
                        name="concept_graph_query",
                        arguments={"topic": "传输层", "query_type": "subtopics"},
                    )
                ],
            )
        return LlmResponse(content="这轮应优先复习拥塞控制和流量控制。")


class _AutonomousProtocolFallbackLlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.calls = 0

    async def send_request(self, request):
        self.calls += 1
        self.prompts.append(request.messages[0].content or "")
        if self.calls == 1:
            return LlmResponse(
                content="先尝试做协议模拟。",
                tool_calls=[
                    ToolCallData(
                        id="ps1",
                        name="protocol_state_simulator",
                        arguments={"protocol": "unsupported_protocol", "params": {}},
                    )
                ],
            )
        if self.calls == 2:
            return LlmResponse(
                content="模拟器不支持，改查知识证据。",
                tool_calls=[
                    ToolCallData(
                        id="kq2",
                        name="knowledge_query",
                        arguments={"query": "TCP 三次握手", "top_k": 3},
                    )
                ],
            )
        return LlmResponse(content="TCP 三次握手通过 SYN、SYN-ACK、ACK 完成。")


class _KnowledgeToolInsufficient:
    @property
    def name(self) -> str:
        return "knowledge_query"

    @property
    def description(self) -> str:
        return "fake knowledge tool"

    def get_args_schema(self):
        from pydantic import BaseModel, Field

        class _Args(BaseModel):
            query: str = Field(default="")
            top_k: int = Field(default=3)

        return _Args

    async def execute(self, context, args):
        return ToolResult(
            success=True,
            result_for_llm="[1] `slides.pdf`\nTCP 拥塞控制包括慢启动。",
            metadata={
                "tool_output_kind": "evidence_context",
                "grounding_capable": True,
                "citations": [{"index": 1, "source": "slides.pdf", "chunk_id": "c1"}],
                "evidence_summary": "[1] slides.pdf",
                "source_count": 1,
            },
        )

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_args_schema().model_json_schema(),
            },
        }


class _KnowledgeToolSufficient(_KnowledgeToolInsufficient):
    async def execute(self, context, args):
        return ToolResult(
            success=True,
            result_for_llm="[1] `slides.pdf`\nTCP 三次握手通过 SYN、SYN-ACK、ACK 完成。",
            metadata={
                "tool_output_kind": "evidence_context",
                "grounding_capable": True,
                "citations": [{"index": 1, "source": "slides.pdf", "chunk_id": "c1"}],
                "evidence_summary": "[1] slides.pdf",
                "source_count": 2,
            },
        )


class _ConceptGraphTool:
    @property
    def name(self) -> str:
        return "concept_graph_query"

    @property
    def description(self) -> str:
        return "fake graph"

    def get_args_schema(self):
        from pydantic import BaseModel, Field

        class _Args(BaseModel):
            topic: str = Field(default="")
            query_type: str = Field(default="subtopics")

        return _Args

    async def execute(self, context, args):
        return ToolResult(
            success=True,
            result_for_llm="主题: 传输层\n1. 拥塞控制，掌握度 20% ⚠️薄弱",
            metadata={
                "tool_output_kind": "analysis_context",
                "graph_rows": [{"concept": "拥塞控制", "mastery": 0.2, "weak": True}],
            },
        )

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_args_schema().model_json_schema(),
            },
        }


class _ProtocolSimulatorUnsupported:
    @property
    def name(self) -> str:
        return "protocol_state_simulator"

    @property
    def description(self) -> str:
        return "fake protocol simulator"

    def get_args_schema(self):
        from pydantic import BaseModel, Field

        class _Args(BaseModel):
            protocol: str = Field(default="")
            params: dict = Field(default_factory=dict)

        return _Args

    async def execute(self, context, args):
        return ToolResult(success=False, error="不支持的协议模拟类型: unsupported_protocol")

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_args_schema().model_json_schema(),
            },
        }


class _AutonomousSkillWorkflow:
    def __init__(self, allowed_tools: list[str]) -> None:
        self._allowed_tools = allowed_tools

    async def try_handle(self, user_message: str, user_id: str):
        return WorkflowResult(
            matched_skill="exam_prep",
            skill_instruction="先观察，再自主选择工具。",
            skill_policy=SkillPolicy(
                allowed_tools=list(self._allowed_tools),
                required_memory=["knowledge_map"],
                allow_autonomous=True,
                max_steps=4,
                output_contract=["review_summary"],
            ),
        )


@pytest.mark.asyncio
async def test_get_decayed_nodes_applies_decay_and_returns_weakest_first(tmp_path: Path) -> None:
    store = KnowledgeMapMemory(str(tmp_path))
    try:
        db = await store._get_conn()
        now = datetime.now(timezone.utc)
        nodes = [
            KnowledgeNode(
                concept="流量控制",
                mastery_level=0.42,
                last_reviewed=now - timedelta(days=5),
                review_interval_days=2,
            ),
            KnowledgeNode(
                concept="拥塞控制",
                mastery_level=0.28,
                last_reviewed=now - timedelta(days=6),
                review_interval_days=2,
            ),
            KnowledgeNode(
                concept="UDP 特点",
                mastery_level=0.91,
                last_reviewed=now - timedelta(hours=2),
                review_interval_days=7,
            ),
        ]
        await db.executemany(
            "INSERT OR REPLACE INTO knowledge_nodes (user_id, concept, data) VALUES (?, ?, ?)",
            [("u1", node.concept, node.model_dump_json()) for node in nodes],
        )
        await db.commit()

        results = await store.get_decayed_nodes("u1", threshold=0.45, limit=5)

        assert [node.concept for node in results] == ["拥塞控制", "流量控制"]
        assert results[0].mastery_level <= results[1].mastery_level
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_review_schedule_hook_triggers_only_for_low_information_message() -> None:
    hook = ReviewScheduleHook(
        knowledge_map=_ReviewKMap(),
        error_memory=_ReviewErrors(),
        student_profile=_ReviewProfile(),
    )

    context, metadata = await hook.get_review_context("u1", "你好")
    assert metadata["proactive_triggered"] is True
    assert "拥塞控制" in context
    assert metadata["proactive_signals"]["inactivity_days"] == 6

    normal_context, normal_meta = await hook.get_review_context("u1", "帮我讲讲 TCP")
    assert normal_context == ""
    assert normal_meta["proactive_triggered"] is False
    assert normal_meta["proactive_reason"] == "message_not_low_information"


def test_compute_pacing_from_conversation_uses_recent_quiz_outcomes() -> None:
    conversation = Conversation(
        id="conv_pacing",
        user_id="u1",
        messages=[
            Message(
                role="assistant",
                tool_calls=[ToolCallData(id="quiz_1", name="quiz_evaluator", arguments={})],
            ),
            Message(role="tool", tool_call_id="quiz_1", content="判定: ✅ 正确"),
            Message(
                role="assistant",
                tool_calls=[ToolCallData(id="quiz_2", name="quiz_evaluator", arguments={})],
            ),
            Message(role="tool", tool_call_id="quiz_2", content="判定: ✅ 正确"),
        ],
    )

    pacing_level, pacing_reason = compute_pacing_from_conversation(conversation)

    assert pacing_level == "accelerate"
    assert "连续两次判题正确" in pacing_reason


@pytest.mark.asyncio
async def test_memory_record_hook_persists_learning_pace() -> None:
    profile = AsyncMock()
    profile.get_profile.return_value = StudentProfile(user_id="u1")
    profile.update_profile = AsyncMock()

    hook = MemoryRecordHook(student_profile=profile, extraction_mode="rule")
    conversation = Conversation(
        id="conv_memory",
        user_id="u1",
        messages=[
            Message(role="user", content="为什么 TCP 三次握手是三次"),
            Message(
                role="assistant",
                tool_calls=[ToolCallData(id="quiz_1", name="quiz_evaluator", arguments={})],
            ),
            Message(role="tool", tool_call_id="quiz_1", content="判定: ❌ 错误"),
            Message(
                role="assistant",
                tool_calls=[ToolCallData(id="quiz_2", name="quiz_evaluator", arguments={})],
            ),
            Message(role="tool", tool_call_id="quiz_2", content="判定: ❌ 错误"),
        ],
    )

    await hook.after_message(conversation)

    updates = profile.update_profile.await_args.args[1]
    assert updates["learning_pace"] == "decelerate"


@pytest.mark.asyncio
async def test_agent_done_metadata_includes_proactive_fields(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.tools.base import ToolRegistry

    conversation = Conversation(id="conv_proactive", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    hook = ReviewScheduleHook(
        knowledge_map=_ReviewKMap(),
        error_memory=_ReviewErrors(),
        student_profile=_ReviewProfile(),
    )

    agent = Agent(
        llm_service=_PassiveLlm(),
        tool_registry=ToolRegistry(),
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=1),
        review_hook=hook,
    )

    events = [event async for event in agent.chat("你好", "u1", "conv_proactive")]
    done_event = next(event for event in events if event.type.value == "done")

    assert done_event.metadata["proactive_triggered"] is True
    assert done_event.metadata["proactive_reason"] == "low_information_message_with_learning_signals"
    assert "weak_concepts" in done_event.metadata["proactive_signals"]


@pytest.mark.asyncio
async def test_autonomous_agent_records_replan_for_insufficient_knowledge_query(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.planner import TaskPlanner
    from src.agent.tools.base import ToolRegistry

    conversation = Conversation(id="conv_replan", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    registry = ToolRegistry()
    registry.register(_KnowledgeToolInsufficient())
    registry.register(_ConceptGraphTool())

    llm = _AutonomousKnowledgeLlm()
    agent = Agent(
        llm_service=llm,
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=4, response_profile="quality_first"),
        task_planner=TaskPlanner(),
        skill_workflow=_AutonomousSkillWorkflow(["knowledge_query", "concept_graph_query"]),
    )

    events = [event async for event in agent.chat("帮我复习传输层", "u1", "conv_replan")]
    done_event = next(event for event in events if event.type.value == "done")

    assert done_event.metadata["effective_control_mode"] == "autonomous"
    assert done_event.metadata["replan_triggered"] is True
    assert done_event.metadata["replan_count"] >= 1
    assert done_event.metadata["fallback_strategy"] == "switch_tool_then_common_knowledge"
    assert done_event.metadata["tool_path"] == ["knowledge_query", "concept_graph_query"]
    assert any("## [Replan Signal]" in prompt for prompt in llm.prompts[1:])


@pytest.mark.asyncio
async def test_autonomous_agent_replans_protocol_fallback_and_records_pacing(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.planner import TaskPlanner
    from src.agent.tools.base import ToolRegistry

    conversation = Conversation(
        id="conv_protocol_fallback",
        user_id="u1",
        messages=[
            Message(
                role="assistant",
                tool_calls=[ToolCallData(id="quiz_ok_1", name="quiz_evaluator", arguments={})],
            ),
            Message(role="tool", tool_call_id="quiz_ok_1", content="判定: ✅ 正确"),
            Message(
                role="assistant",
                tool_calls=[ToolCallData(id="quiz_ok_2", name="quiz_evaluator", arguments={})],
            ),
            Message(role="tool", tool_call_id="quiz_ok_2", content="判定: ✅ 正确"),
        ],
    )
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    registry = ToolRegistry()
    registry.register(_ProtocolSimulatorUnsupported())
    registry.register(_KnowledgeToolSufficient())

    llm = _AutonomousProtocolFallbackLlm()
    agent = Agent(
        llm_service=llm,
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=4, response_profile="quality_first"),
        task_planner=TaskPlanner(),
        skill_workflow=_AutonomousSkillWorkflow(["protocol_state_simulator", "knowledge_query"]),
    )

    events = [event async for event in agent.chat("如果 SYN-ACK 丢了会怎样？", "u1", "conv_protocol_fallback")]
    done_event = next(event for event in events if event.type.value == "done")

    assert done_event.metadata["replan_triggered"] is True
    assert done_event.metadata["fallback_strategy"] == "fallback_to_knowledge_query"
    assert done_event.metadata["tool_path"] == ["protocol_state_simulator", "knowledge_query"]
    assert done_event.metadata["pacing_level"] == "accelerate"
    assert "连续两次判题正确" in done_event.metadata["pacing_reason"]
