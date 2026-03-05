"""Unit tests for ContextEngineeringFilter."""

import tempfile

import pytest

from src.agent.memory.context_filter import ContextEngineeringFilter
from src.agent.types import Message


def _make_messages(n: int, role: str = "user") -> list[Message]:
    return [Message(role=role, content=f"msg {i}") for i in range(n)]


class TestContextEngineeringFilter:
    def test_no_filtering_under_limit(self):
        f = ContextEngineeringFilter(max_messages=40)
        msgs = _make_messages(10)
        result = f.filter_messages(msgs)
        assert len(result) == 10

    def test_sliding_window(self):
        f = ContextEngineeringFilter(max_messages=10)
        msgs = _make_messages(25)
        result = f.filter_messages(msgs)
        assert len(result) == 10
        assert result[0].content == "msg 15"

    def test_tool_result_offloading(self):
        with tempfile.TemporaryDirectory() as d:
            f = ContextEngineeringFilter(
                max_messages=100,
                tool_result_max_chars=50,
                offload_dir=d,
            )
            msgs = [
                Message(role="user", content="hi"),
                Message(role="tool", content="x" * 200, tool_call_id="tc1"),
            ]
            result = f.filter_messages(msgs)
            assert len(result) == 2
            assert "卸载" in result[1].content
            assert len(result[1].content) < 300

    def test_offload_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            f = ContextEngineeringFilter(
                max_messages=100,
                tool_result_max_chars=10,
                offload_dir=d,
            )
            content = "a" * 100
            ref = f._offload_to_file(content)
            loaded = f.load_offloaded(ref)
            assert loaded == content
