"""Self-reflection middleware — detects potential hallucinations in LLM responses.

After the LLM generates a response, this middleware checks whether the answer
is grounded in the RAG context (tool results).  If the overlap between the
response and the retrieved knowledge is below a configurable threshold, a
warning note is appended to the response.
"""

from __future__ import annotations

import logging
import re
from typing import Set

from src.agent.grounding import GroundingEvaluator, build_evidence_bundle
from src.agent.hooks.middleware import LlmMiddleware
from src.agent.types import LlmRequest, LlmResponse

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> Set[str]:
    """Simple whitespace + punctuation tokeniser for overlap calculation."""
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))


class ReflectionMiddleware(LlmMiddleware):
    """Appends a hallucination warning when response is poorly grounded.

    Parameters:
        groundedness_threshold: Minimum token overlap ratio (0-1).
        append_warning: Whether to actually append the warning text.
        enabled: Master switch for the middleware.
    """

    WARNING_TEXT = (
        "\n\n> ⚠️ 注意：以上回答部分内容可能未被课程资料直接支持，建议结合教材验证。"
    )

    def __init__(
        self,
        *,
        groundedness_threshold: float = 0.3,
        append_warning: bool = True,
        enabled: bool = True,
    ) -> None:
        self._threshold = groundedness_threshold
        self._append_warning = append_warning
        self._enabled = enabled
        self._evaluator = GroundingEvaluator(threshold=groundedness_threshold)

    async def after_llm_response(
        self, request: LlmRequest, response: LlmResponse,
    ) -> LlmResponse:
        if not self._enabled:
            return response

        if bool(request.metadata.get("skip_reflection_warning", False)):
            return response

        if not response.content or response.tool_calls:
            return response

        bundle = build_evidence_bundle(
            "reflection",
                {
                    "grounding_capable": bool(request.metadata.get("citations") or request.metadata.get("evidence_summary")),
                    "citations": request.metadata.get("citations", []),
                    "evidence_summary": request.metadata.get("evidence_summary", ""),
                    "source_count": int(request.metadata.get("source_count", len(request.metadata.get("citations", [])))),
                },
            )
        rag_context = self._extract_rag_context(request.messages)
        if bundle is None and not rag_context:
            return response

        if bundle is None and rag_context:
            bundle = build_evidence_bundle(
                "reflection",
                {
                    "grounding_capable": True,
                    "citations": [],
                    "evidence_summary": rag_context,
                    "source_count": 1,
                },
            )
        assessment = self._evaluator.assess(response.content, bundle)
        score = assessment.score
        logger.debug("Reflection groundedness score: %.3f", score)

        if assessment.low_evidence and self._append_warning:
            response = LlmResponse(
                content=response.content + self.WARNING_TEXT,
                tool_calls=response.tool_calls,
                error=response.error,
            )

        return response

    @staticmethod
    def _extract_rag_context(messages: list) -> str:
        """Find the most recent tool-result message in the conversation."""
        for msg in reversed(messages):
            if getattr(msg, "role", None) == "tool" and getattr(msg, "content", None):
                return msg.content
        return ""

    @staticmethod
    def _groundedness_score(response: str, context: str) -> float:
        """Compute token-level overlap between response and RAG context."""
        resp_tokens = _tokenize(response)
        ctx_tokens = _tokenize(context)
        if not resp_tokens:
            return 1.0
        overlap = resp_tokens & ctx_tokens
        return len(overlap) / len(resp_tokens)
