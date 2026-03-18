from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from src.agent.post_actions import ArtifactPostActionAdapter
from src.agent.skills.registry import SkillPolicy
from src.agent.skills.workflow import WorkflowResult
from src.agent.types import Conversation, LlmResponse, ToolCallData, ToolContext, ToolResult
from src.core.trace.trace_collector import TraceCollector
from src.observability.dashboard.services.trace_service import TraceService
from src.storage.object_store import LocalObjectStore


class _GraphArgs(BaseModel):
    topic: str = Field(default="")
    query_type: str = Field(default="subtopics")


class _KnowledgeArgs(BaseModel):
    query: str = Field(default="")
    top_k: int = Field(default=5)


class _ConceptGraphTool:
    @property
    def name(self) -> str:
        return "concept_graph_query"

    @property
    def description(self) -> str:
        return "fake graph tool"

    def get_args_schema(self):
        return _GraphArgs

    async def execute(self, context, args):
        return ToolResult(
            success=True,
            result_for_llm="主题: 传输层\n1. 拥塞控制，掌握度 20% ⚠️薄弱",
            metadata={"tool_output_kind": "analysis_context", "graph_rows": [{"concept": "拥塞控制", "weak": True}]},
        )

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_args_schema().model_json_schema(),
            },
        }


class _KnowledgeTool:
    @property
    def name(self) -> str:
        return "knowledge_query"

    @property
    def description(self) -> str:
        return "fake knowledge query"

    def get_args_schema(self):
        return _KnowledgeArgs

    async def execute(self, context, args):
        return ToolResult(
            success=True,
            result_for_llm="以下是与问题最相关的课程证据。\n\n[1] `tcp.pdf`\nTCP 拥塞控制包括慢启动和拥塞避免。",
            metadata={
                "tool_output_kind": "evidence_context",
                "grounding_capable": True,
                "citations": [{"index": 1, "source": "tcp.pdf", "chunk_id": "c1"}],
                "evidence_summary": "[1] tcp.pdf",
                "source_count": 1,
                "query_trace_ids": ["qt-1"],
            },
        )

    def get_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_args_schema().model_json_schema(),
            },
        }


class _AutonomousLlm:
    def __init__(self) -> None:
        self.calls = 0

    async def send_request(self, request):
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(
                content="先看掌握情况。",
                tool_calls=[
                    ToolCallData(
                        id="tc_1",
                        name="concept_graph_query",
                        arguments={"topic": "传输层", "query_type": "subtopics"},
                    )
                ],
            )
        if self.calls == 2:
            return LlmResponse(
                content="再检索薄弱点证据。",
                tool_calls=[
                    ToolCallData(
                        id="tc_2",
                        name="knowledge_query",
                        arguments={"query": "TCP 拥塞控制", "top_k": 3},
                    )
                ],
            )
        return LlmResponse(content="建议重点复习 TCP 拥塞控制，它涉及慢启动和拥塞避免。[1]")


class _SkillWorkflow:
    async def try_handle(self, user_message: str, user_id: str):
        return WorkflowResult(
            matched_skill="exam_prep",
            skill_instruction="先看知识掌握，再检索证据，最后给出复习建议。",
            skill_policy=SkillPolicy(
                allowed_tools=["concept_graph_query", "knowledge_query"],
                required_memory=["knowledge_map", "error_memory"],
                allow_autonomous=True,
                max_steps=4,
                output_contract=["review_summary"],
                post_actions=["notes_export"],
            ),
        )


@pytest.mark.asyncio
async def test_autonomous_agent_generates_artifact_and_trace(tmp_path: Path) -> None:
    from src.agent.agent import Agent
    from src.agent.config import AgentConfig
    from src.agent.conversation import ConversationStore
    from src.agent.planner import TaskPlanner
    from src.agent.tools.base import ToolRegistry

    trace_path = tmp_path / "autonomous_agent_traces.jsonl"
    collector = TraceCollector(traces_path=trace_path)
    conversation = Conversation(id="conv_auto", user_id="u1", messages=[])
    store = AsyncMock(spec=ConversationStore)
    store.get.return_value = conversation
    store.create.return_value = conversation
    store.update.return_value = None

    registry = ToolRegistry()
    registry.register(_ConceptGraphTool())
    registry.register(_KnowledgeTool())
    object_store = LocalObjectStore(str(tmp_path / "objects"))

    agent = Agent(
        llm_service=_AutonomousLlm(),
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=6, response_profile="quality_first"),
        task_planner=TaskPlanner(),
        skill_workflow=_SkillWorkflow(),
        post_action_adapter=ArtifactPostActionAdapter(object_store=object_store),
        trace_enabled=True,
        trace_collector=collector,
    )

    events = [event async for event in agent.chat("帮我复习传输层", "u1", "conv_auto")]
    done_event = next(event for event in events if event.type.value == "done")

    assert done_event.metadata["matched_skill"] == "exam_prep"
    assert done_event.metadata["effective_control_mode"] == "autonomous"
    assert done_event.metadata["autonomous_step_count"] == 2
    assert done_event.metadata["tool_path"] == ["concept_graph_query", "knowledge_query"]
    assert len(done_event.metadata["artifacts"]) == 1
    assert done_event.metadata["artifacts"][0]["artifact_type"] == "notes_export"

    trace = TraceService(trace_path).get_trace(done_event.metadata["trace_id"])
    assert trace is not None
    stage_names = [stage["stage"] for stage in trace["stages"]]
    assert "skill_policy_applied" in stage_names
    assert "autonomous_iteration" in stage_names
    assert "autonomous_stop_reason" in stage_names
    assert "post_actions" in stage_names


def test_artifact_download_route_serves_exported_file(tmp_path: Path) -> None:
    from src.server.routes import configure_routes, router

    object_store = LocalObjectStore(str(tmp_path / "objects"))
    adapter = ArtifactPostActionAdapter(object_store=object_store)
    artifacts = asyncio.run(
        adapter.run(
            user_id="u1",
            conversation_id="conv_route",
            matched_skill="chapter_deep_dive",
            post_actions=["notes_export"],
            final_text="这是复习总结。",
            tool_path=["knowledge_query"],
        )
    )
    artifact = artifacts[0]

    app = FastAPI()
    configure_routes(
        chat_handler=SimpleNamespace(handle_stream=None),
        agent=SimpleNamespace(tools=SimpleNamespace(tool_names=[])),
        object_store=object_store,
    )
    app.include_router(router)
    client = TestClient(app)

    response = client.get(artifact["download_url"])
    assert response.status_code == 200
    assert "学习笔记" in response.text
