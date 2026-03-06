"""JSON parsing utilities for LLM response handling."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def safe_parse_json(raw: str, *, fallback: Any = None) -> Any:
    """Parse JSON from LLM output, stripping markdown code fences if present.

    Returns *fallback* (default ``None``) on parse failure instead of raising.
    """
    if not raw:
        return fallback

    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.debug("JSON parse failed for: %s…", text[:100])
        return fallback
