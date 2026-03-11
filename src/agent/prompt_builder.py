"""System prompt builder — assembles the full system prompt with dynamic sections."""

from __future__ import annotations

import logging
import string
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE = (
    "你是一个专业的课程学习助手。请帮助用户进行知识问答、考点复习、习题练习。\n"
    "$tool_descriptions\n$memory_context\n$planner_context\n$active_skill"
)


class SystemPromptBuilder:
    """Builds the Agent system prompt using string.Template.safe_substitute
    to prevent KeyError / injection from user-controlled content."""

    def __init__(self, template_path: str = "config/prompts/system_prompt.txt") -> None:
        self._template_path = Path(template_path)
        self._cached_template: str | None = None

    def _load_template(self) -> str:
        if self._cached_template is not None:
            return self._cached_template
        try:
            raw = self._template_path.read_text(encoding="utf-8")
            raw = raw.replace("{tool_descriptions}", "$tool_descriptions")
            raw = raw.replace("{memory_context}", "$memory_context")
            raw = raw.replace("{planner_context}", "$planner_context")
            raw = raw.replace("{active_skill}", "$active_skill")
            self._cached_template = raw
        except FileNotFoundError:
            logger.warning("System prompt template not found at %s, using default", self._template_path)
            self._cached_template = _DEFAULT_TEMPLATE
        return self._cached_template

    def build(
        self,
        tool_schemas: list[dict] | None = None,
        memory_context: str = "",
        planner_context: str = "",
        active_skill: str = "",
    ) -> str:
        template_str = self._load_template()
        tmpl = string.Template(template_str)

        tool_desc = ""
        if tool_schemas:
            lines = []
            for ts in tool_schemas:
                fn = ts.get("function", {})
                lines.append(f"- **{fn.get('name', '?')}**: {fn.get('description', '')}")
            tool_desc = "\n".join(lines)

        mem_section = ""
        if memory_context:
            mem_section = f"## 学生记忆上下文\n{memory_context}"

        skill_section = ""
        if active_skill:
            skill_section = f"## [Active Skill]\n{active_skill}"

        planner_section = ""
        if planner_context:
            planner_section = f"## [Planner Context]\n{planner_context}"

        return tmpl.safe_substitute(
            tool_descriptions=tool_desc,
            memory_context=mem_section,
            planner_context=planner_section,
            active_skill=skill_section,
        ).strip()
