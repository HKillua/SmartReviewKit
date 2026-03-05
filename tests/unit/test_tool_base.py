"""Unit tests for Tool base class and ToolRegistry."""

import asyncio

import pytest
from pydantic import BaseModel

from src.agent.tools.base import Tool, ToolRegistry
from src.agent.types import ToolCallData, ToolContext, ToolResult


class MockArgs(BaseModel):
    query: str
    top_k: int = 5


class MockTool(Tool[MockArgs]):
    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    def get_args_schema(self) -> type[MockArgs]:
        return MockArgs

    async def execute(self, context: ToolContext, args: MockArgs) -> ToolResult:
        return ToolResult(success=True, result_for_llm=f"Found {args.top_k} results for '{args.query}'")


class FailingTool(Tool[MockArgs]):
    @property
    def name(self) -> str:
        return "failing_tool"

    @property
    def description(self) -> str:
        return "Always fails"

    def get_args_schema(self) -> type[MockArgs]:
        return MockArgs

    async def execute(self, context: ToolContext, args: MockArgs) -> ToolResult:
        raise RuntimeError("Boom!")


@pytest.fixture
def ctx():
    return ToolContext(user_id="u1", conversation_id="c1")


class TestTool:
    def test_schema_generation(self):
        tool = MockTool()
        schema = tool.get_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "mock_tool"
        assert "parameters" in schema["function"]


class TestToolRegistry:
    def test_register_and_lookup(self):
        reg = ToolRegistry()
        reg.register(MockTool())
        assert "mock_tool" in reg.tool_names
        tool = reg.get_tool("mock_tool")
        assert tool.name == "mock_tool"

    def test_duplicate_register(self):
        reg = ToolRegistry()
        reg.register(MockTool())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(MockTool())

    def test_get_unknown_tool(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.get_tool("nonexistent")

    def test_get_all_schemas(self):
        reg = ToolRegistry()
        reg.register(MockTool())
        schemas = reg.get_all_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "mock_tool"

    @pytest.mark.asyncio
    async def test_execute_success(self, ctx):
        reg = ToolRegistry()
        reg.register(MockTool())
        tc = ToolCallData(id="tc1", name="mock_tool", arguments={"query": "SQL"})
        result = await reg.execute(tc, ctx)
        assert result.success is True
        assert "SQL" in result.result_for_llm

    @pytest.mark.asyncio
    async def test_execute_invalid_args(self, ctx):
        reg = ToolRegistry()
        reg.register(MockTool())
        tc = ToolCallData(id="tc1", name="mock_tool", arguments={"invalid": True})
        result = await reg.execute(tc, ctx)
        assert result.success is False
        assert "Invalid arguments" in result.error

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, ctx):
        reg = ToolRegistry()
        tc = ToolCallData(id="tc1", name="nope", arguments={})
        result = await reg.execute(tc, ctx)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_exception(self, ctx):
        reg = ToolRegistry()
        reg.register(FailingTool())
        tc = ToolCallData(id="tc1", name="failing_tool", arguments={"query": "x"})
        result = await reg.execute(tc, ctx)
        assert result.success is False
        assert "Boom!" in result.error
