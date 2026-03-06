"""Input sanitization utilities for prompt injection defense."""

from __future__ import annotations

import re
from pathlib import Path

_INJECTION_PATTERNS = [
    re.compile(r"忽略.{0,10}(以上|之前|上面|前面).{0,10}(指令|规则|提示|命令|要求)", re.IGNORECASE),
    re.compile(r"ignore.{0,15}(previous|above|prior|all).{0,15}(instructions?|rules?|prompts?)", re.IGNORECASE),
    re.compile(r"disregard.{0,15}(previous|above|prior|all).{0,15}(instructions?|rules?)", re.IGNORECASE),
    re.compile(r"forget.{0,15}(previous|above|prior|all|everything)", re.IGNORECASE),
    re.compile(r"(system|系统).{0,10}(prompt|提示词|指令)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"(act|pretend|roleplay)\s+(as|like)", re.IGNORECASE),
    re.compile(r"(输出|打印|显示|泄露).{0,10}(系统|system).{0,10}(提示|prompt)", re.IGNORECASE),
    re.compile(r"DAN\s+mode|Do\s+Anything\s+Now", re.IGNORECASE),
    re.compile(r"jailbreak|越狱|解除限制", re.IGNORECASE),
    re.compile(r"developer\s+mode\s+(enabled|on|activate)", re.IGNORECASE),
    re.compile(r"(?:from\s+now\s+on|以后).{0,20}(?:you\s+are|你是|act\s+as)", re.IGNORECASE),
    re.compile(r"(?:base64|b64)\s*[:：(（]\s*[A-Za-z0-9+/=]{20,}", re.IGNORECASE),
    re.compile(r"repeat\s+(?:the\s+)?(?:above|system|initial)\s+(?:text|prompt|instructions?)", re.IGNORECASE),
]


def sanitize_user_input(text: str, *, max_length: int = 2000) -> str:
    """Remove common prompt injection patterns and enforce length limits.

    Returns the sanitized text. Injection fragments are replaced with
    ``[FILTERED]`` so the LLM still sees a coherent string.
    """
    if not text:
        return text
    text = text[:max_length]
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub("[FILTERED]", text)
    return text


def validate_path_within(path: str | Path, allowed_base: str | Path) -> Path:
    """Resolve *path* and verify it is inside *allowed_base*.

    Raises ``ValueError`` if the resolved path escapes the allowed directory.
    """
    resolved = Path(path).resolve()
    base = Path(allowed_base).resolve()
    if not str(resolved).startswith(str(base) + "/") and resolved != base:
        raise ValueError(f"路径 {resolved} 不在允许范围 {base} 内")
    return resolved
