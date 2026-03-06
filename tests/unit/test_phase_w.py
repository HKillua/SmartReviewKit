"""Unit tests for Phase W: Self-Reflection (Anti-Hallucination)."""

import asyncio
import unittest
from unittest.mock import MagicMock


class TestReflectionMiddleware(unittest.TestCase):
    def _make_middleware(self, **kwargs):
        from src.agent.hooks.reflection import ReflectionMiddleware
        return ReflectionMiddleware(**kwargs)

    def _make_request_response(self, response_text, tool_result_text=None, has_tool_calls=False):
        from src.agent.types import LlmRequest, LlmResponse, LlmMessage
        messages = [LlmMessage(role="system", content="You are a helper.")]
        if tool_result_text:
            messages.append(LlmMessage(role="tool", content=tool_result_text))
        messages.append(LlmMessage(role="user", content="explain TCP"))

        req = LlmRequest(messages=messages)
        resp = LlmResponse(
            content=response_text,
            tool_calls=[MagicMock()] if has_tool_calls else None,
        )
        return req, resp

    def test_no_warning_when_well_grounded(self):
        mw = self._make_middleware(groundedness_threshold=0.2)
        rag_ctx = "TCP 是传输控制协议 工作在传输层 提供可靠传输"
        response = "TCP 是传输控制协议，它工作在传输层，提供可靠的数据传输服务。"
        req, resp = self._make_request_response(response, tool_result_text=rag_ctx)

        result = asyncio.get_event_loop().run_until_complete(
            mw.after_llm_response(req, resp)
        )
        self.assertNotIn("⚠️", result.content)

    def test_warning_when_poorly_grounded(self):
        mw = self._make_middleware(groundedness_threshold=0.9)
        rag_ctx = "TCP 是传输控制协议。"
        response = "HTTP 是应用层协议，DNS 用于域名解析，ICMP 用于网络诊断。"
        req, resp = self._make_request_response(response, tool_result_text=rag_ctx)

        result = asyncio.get_event_loop().run_until_complete(
            mw.after_llm_response(req, resp)
        )
        self.assertIn("⚠️", result.content)

    def test_skip_when_no_rag_context(self):
        mw = self._make_middleware(groundedness_threshold=0.5)
        req, resp = self._make_request_response("Some response.", tool_result_text=None)

        result = asyncio.get_event_loop().run_until_complete(
            mw.after_llm_response(req, resp)
        )
        self.assertNotIn("⚠️", result.content)

    def test_skip_when_tool_calls(self):
        from src.agent.types import ToolCallData
        mw = self._make_middleware(groundedness_threshold=0.01)
        from src.agent.types import LlmRequest, LlmResponse, LlmMessage
        req = LlmRequest(messages=[
            LlmMessage(role="system", content="sys"),
            LlmMessage(role="tool", content="context"),
            LlmMessage(role="user", content="q"),
        ])
        resp = LlmResponse(
            content="response",
            tool_calls=[ToolCallData(id="tc1", name="test", arguments={})],
        )
        result = asyncio.get_event_loop().run_until_complete(
            mw.after_llm_response(req, resp)
        )
        self.assertNotIn("⚠️", result.content)

    def test_disabled(self):
        mw = self._make_middleware(enabled=False, groundedness_threshold=0.99)
        req, resp = self._make_request_response("hallucinated", tool_result_text="real context")
        result = asyncio.get_event_loop().run_until_complete(
            mw.after_llm_response(req, resp)
        )
        self.assertNotIn("⚠️", result.content)

    def test_groundedness_score_calculation(self):
        from src.agent.hooks.reflection import ReflectionMiddleware
        score = ReflectionMiddleware._groundedness_score(
            "TCP UDP 协议 传输层",
            "TCP 是传输层协议 UDP 也是传输层协议",
        )
        self.assertGreaterEqual(score, 0.5)

    def test_groundedness_score_no_overlap(self):
        from src.agent.hooks.reflection import ReflectionMiddleware
        score = ReflectionMiddleware._groundedness_score(
            "apple banana cherry",
            "TCP UDP HTTP DNS",
        )
        self.assertLess(score, 0.3)

    def test_empty_response_returns_1(self):
        from src.agent.hooks.reflection import ReflectionMiddleware
        score = ReflectionMiddleware._groundedness_score("", "some context")
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
