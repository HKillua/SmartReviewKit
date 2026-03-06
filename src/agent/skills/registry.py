"""Skill registry — 3-level progressive disclosure loading.

Level 1 (Metadata): name + description + triggers — loaded at startup (~tokens per skill: 50)
Level 2 (Instruction): full SOP steps — loaded on match
Level 3 (Resource): external resources — loaded during execution
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SkillMetadata(BaseModel):
    name: str
    description: str = ""
    trigger_patterns: list[str] = Field(default_factory=list)
    tools_required: list[str] = Field(default_factory=list)
    memory_required: list[str] = Field(default_factory=list)
    estimated_tokens: int = 500
    difficulty: str = "medium"


class SkillInstruction(BaseModel):
    name: str
    steps: list[str] = Field(default_factory=list)
    output_format: str = ""
    quality_checks: list[str] = Field(default_factory=list)
    raw_body: str = ""


class SkillRegistry:
    """Central registry for Agent skills with progressive disclosure."""

    def __init__(self, skills_dir: str = "src/agent/skills/definitions") -> None:
        self._skills_dir = Path(skills_dir)
        self._metadata: dict[str, SkillMetadata] = {}
        self._load_all_metadata()

    def _load_all_metadata(self) -> None:
        if not self._skills_dir.exists():
            logger.warning("Skills directory not found: %s", self._skills_dir)
            return

        for skill_dir in sorted(self._skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            try:
                meta = self._parse_frontmatter(skill_file)
                if meta:
                    self._metadata[meta.name] = meta
                    logger.debug("Loaded skill metadata: %s", meta.name)
            except Exception:
                logger.warning("Failed to parse skill: %s", skill_file)

        logger.info("Loaded %d skill(s) from %s", len(self._metadata), self._skills_dir)

    @staticmethod
    def _parse_frontmatter(path: Path) -> Optional[SkillMetadata]:
        content = path.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not match:
            return None
        fm = yaml.safe_load(match.group(1))
        if not isinstance(fm, dict) or "name" not in fm:
            return None
        return SkillMetadata.model_validate(fm)

    def get_skill_descriptions_for_prompt(self) -> str:
        if not self._metadata:
            return ""
        lines = ["可用学习技能:"]
        for meta in self._metadata.values():
            lines.append(f"- **{meta.name}**: {meta.description}")
        return "\n".join(lines)

    def match_skill(self, user_message: str) -> Optional[str]:
        msg_lower = user_message.lower()
        for name, meta in self._metadata.items():
            for pattern in meta.trigger_patterns:
                if pattern.lower() in msg_lower:
                    return name
        return None

    def load_instruction(self, skill_name: str) -> Optional[SkillInstruction]:
        if skill_name not in self._metadata:
            return None
        skill_file = self._skills_dir / skill_name / "SKILL.md"
        if not skill_file.exists():
            return None

        content = skill_file.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n.*?\n---\s*\n(.*)$", content, re.DOTALL)
        body = match.group(1).strip() if match else content.strip()

        steps: list[str] = []
        for line in body.split("\n"):
            stripped = line.strip()
            if re.match(r"^\d+\.", stripped):
                steps.append(stripped)

        return SkillInstruction(
            name=skill_name,
            steps=steps,
            raw_body=body,
        )

    def load_resource(self, skill_name: str, resource_path: str) -> Optional[str]:
        base = (self._skills_dir / skill_name).resolve()
        path = (base / resource_path).resolve()
        if not str(path).startswith(str(base)):
            logger.warning("Path traversal attempt blocked: %s", resource_path)
            return None
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    @property
    def skill_names(self) -> list[str]:
        return list(self._metadata.keys())

    def get_metadata(self, skill_name: str) -> Optional[SkillMetadata]:
        return self._metadata.get(skill_name)
