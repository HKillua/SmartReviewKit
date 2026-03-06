"""Security guardrails hook — enhanced input/output filtering.

Runs as the first LifecycleHook in the chain to intercept dangerous inputs
before they reach the LLM, and redact sensitive information from outputs.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.agent.hooks.lifecycle import LifecycleHook
from src.agent.types import Conversation, ToolContext, ToolResult
from src.agent.utils.sanitizer import sanitize_user_input

logger = logging.getLogger(__name__)

_DAN_PATTERNS = [
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"Do\s+Anything\s+Now", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"developer\s+mode\s+(enabled|on|activate)", re.IGNORECASE),
]

_BASE64_INJECT = re.compile(
    r"(?:base64|b64|decode)\s*[:：(（]\s*[A-Za-z0-9+/=]{20,}",
    re.IGNORECASE,
)

_ROLE_CHAIN = re.compile(
    r"(?:from\s+now\s+on|henceforth|以后|从现在起).{0,30}"
    r"(?:you\s+are|你是|act\s+as|扮演|假装)",
    re.IGNORECASE,
)

_SENSITIVE_OUTPUT_PATTERNS = [
    re.compile(r"(?:api[_-]?key|secret|token)\s*[:=]\s*\S{8,}", re.IGNORECASE),
    re.compile(r"(?:password|passwd|密码)\s*[:=：]\s*\S{4,}", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    re.compile(r"\b(?:sk-|pk-|Bearer\s+)[A-Za-z0-9]{20,}\b"),
]


class GuardrailsHook(LifecycleHook):
    """Input validation and output redaction hook.

    Parameters:
        input_filtering: Enable enhanced input injection detection.
        output_redaction: Enable sensitive-info redaction in outputs.
        block_on_high_risk: If True, replace high-risk input with safe message.
        enabled: Master switch.
    """

    BLOCKED_MESSAGE = "检测到潜在安全风险，已过滤该请求。请重新表述您的问题。"

    def __init__(
        self,
        *,
        input_filtering: bool = True,
        output_redaction: bool = True,
        block_on_high_risk: bool = True,
        enabled: bool = True,
    ) -> None:
        self._input_filtering = input_filtering
        self._output_redaction = output_redaction
        self._block_on_high_risk = block_on_high_risk
        self._enabled = enabled

    async def before_message(self, user_id: str, message: str) -> Optional[str]:
        if not self._enabled or not self._input_filtering:
            return None

        risk = self._assess_risk(message)
        if risk == "high" and self._block_on_high_risk:
            logger.warning("Guardrails: HIGH risk input blocked for user %s", user_id)
            return self.BLOCKED_MESSAGE

        return sanitize_user_input(message)

    async def after_message(self, conversation: Conversation) -> None:
        if not self._enabled or not self._output_redaction:
            return
        if not conversation.messages:
            return

        last = conversation.messages[-1]
        if last.role == "assistant" and last.content:
            last.content = self._redact_sensitive(last.content)

    def _assess_risk(self, text: str) -> str:
        """Return ``"high"`` | ``"medium"`` | ``"low"``."""
        for p in _DAN_PATTERNS:
            if p.search(text):
                return "high"
        if _BASE64_INJECT.search(text):
            return "high"
        if _ROLE_CHAIN.search(text):
            return "medium"
        return "low"

    @staticmethod
    def _redact_sensitive(text: str) -> str:
        """Replace sensitive patterns in *text* with ``[REDACTED]``."""
        for p in _SENSITIVE_OUTPUT_PATTERNS:
            text = p.sub("[REDACTED]", text)
        return text
