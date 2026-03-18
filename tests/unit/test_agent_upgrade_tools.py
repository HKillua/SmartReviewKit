from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.skills.registry import SkillRegistry
from src.agent.tools.concept_graph_query import ConceptGraphQueryArgs, ConceptGraphQueryTool
from src.agent.tools.network_calc import NetworkCalcArgs, NetworkCalcTool
from src.agent.tools.protocol_state_simulator import (
    ProtocolStateSimulatorArgs,
    ProtocolStateSimulatorTool,
)
from src.agent.types import ToolContext


class _FakeKnowledgeMap:
    def __init__(self, mastery: dict[str, float]) -> None:
        self._mastery = mastery

    async def get_node(self, user_id: str, concept: str):
        if concept not in self._mastery:
            return None
        return SimpleNamespace(mastery_level=self._mastery[concept])


class _FakeErrorMemory:
    async def get_weak_concepts(self, user_id: str) -> list[str]:
        return ["拥塞控制"]


@pytest.mark.asyncio
async def test_network_calc_subnet_division_returns_structured_result() -> None:
    tool = NetworkCalcTool()
    result = await tool.execute(
        ToolContext(user_id="u1", conversation_id="c1"),
        NetworkCalcArgs(
            type="subnet_division",
            params={"network": "192.168.1.0/24", "num_subnets": 4},
        ),
    )
    assert result.success is True
    assert "可用主机数: 62" in result.result_for_llm
    assert result.metadata["structured_result"]["new_prefix"] == 26
    assert result.metadata["tool_output_kind"] == "analysis_context"


@pytest.mark.asyncio
async def test_concept_graph_query_review_order_uses_mastery_overlay() -> None:
    tool = ConceptGraphQueryTool(
        knowledge_map=_FakeKnowledgeMap({"TCP": 0.8, "拥塞控制": 0.2, "流量控制": 0.4}),
        error_memory=_FakeErrorMemory(),
    )
    result = await tool.execute(
        ToolContext(user_id="u1", conversation_id="c1"),
        ConceptGraphQueryArgs(topic="TCP", query_type="review_order", limit=6),
    )
    assert result.success is True
    rows = result.metadata["graph_rows"]
    assert rows[0]["concept"] in {"拥塞控制", "流量控制"}
    assert any(row["weak"] for row in rows)
    assert result.metadata["tool_output_kind"] == "analysis_context"


@pytest.mark.asyncio
async def test_protocol_state_simulator_supports_fault_injection() -> None:
    tool = ProtocolStateSimulatorTool()
    result = await tool.execute(
        ToolContext(user_id="u1", conversation_id="c1"),
        ProtocolStateSimulatorArgs(
            protocol="tcp_handshake",
            params={"initial_seq": 100},
            fault_injection={"drop_packet": 2},
        ),
    )
    assert result.success is True
    assert "SYN-ACK 丢失" in result.result_for_llm
    assert "sequenceDiagram" in result.metadata["mermaid"]
    assert result.metadata["simulation_protocol"] == "tcp_handshake"


def test_skill_registry_loads_structured_skill_policy() -> None:
    registry = SkillRegistry("src/agent/skills/definitions")
    policy = registry.load_policy("exam_prep")
    assert policy is not None
    assert policy.allow_autonomous is True
    assert "concept_graph_query" in policy.allowed_tools
    assert "schedule_export" in policy.post_actions
