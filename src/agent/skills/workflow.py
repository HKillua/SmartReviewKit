"""Skill workflow handler — checks user message against skills and injects SOP."""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel

from src.agent.skills.registry import SkillPolicy, SkillRegistry

logger = logging.getLogger(__name__)


class WorkflowResult(BaseModel):
    should_skip_llm: bool = False
    skill_instruction: Optional[str] = None
    matched_skill: Optional[str] = None
    direct_response: Optional[str] = None
    skill_policy: Optional[SkillPolicy] = None


class SkillWorkflowHandler:
    """Integrates skill matching into the Agent message flow."""

    def __init__(self, skill_registry: SkillRegistry) -> None:
        self._registry = skill_registry

    async def try_handle(self, user_message: str, user_id: str) -> WorkflowResult:
        # Handle /help command
        if user_message.strip().lower() in ("/help", "/skills", "帮助"):
            desc = self._registry.get_skill_descriptions_for_prompt()
            help_text = "## 可用技能\n\n" + (desc or "暂无已加载的技能。") + (
                "\n\n直接输入问题即可自动匹配技能，或使用以下关键词触发：\n"
                "- 「考点复习」「期末复习」→ 考点复习\n"
                "- 「出题」「练习」→ 习题训练\n"
                "- 「错题回顾」→ 错题复习\n"
                "- 「章节详解」→ 章节深入\n"
                "- 「掌握度」「知识图谱」→ 知识检查"
            )
            return WorkflowResult(should_skip_llm=True, direct_response=help_text)

        # Try to match a skill
        matched = self._registry.match_skill(user_message)
        if matched is None:
            return WorkflowResult()

        # Load Level 2 instruction
        instruction = self._registry.load_instruction(matched)
        policy = self._registry.load_policy(matched)
        if instruction is None:
            logger.warning("Matched skill '%s' but failed to load instruction", matched)
            return WorkflowResult()

        skill_text = (
            f"你正在执行技能「{matched}」。请严格按照以下步骤操作：\n\n"
            f"{instruction.raw_body}"
        )

        logger.info("Skill matched: %s for user %s", matched, user_id)
        return WorkflowResult(
            skill_instruction=skill_text,
            matched_skill=matched,
            skill_policy=policy,
        )
