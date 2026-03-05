"""Unit tests for SkillRegistry."""

import pytest

from src.agent.skills.registry import SkillRegistry


class TestSkillRegistry:
    def test_load_predefined_skills(self):
        reg = SkillRegistry("src/agent/skills/definitions")
        assert len(reg.skill_names) >= 5
        assert "exam_prep" in reg.skill_names
        assert "quiz_drill" in reg.skill_names

    def test_match_skill(self):
        reg = SkillRegistry("src/agent/skills/definitions")
        assert reg.match_skill("帮我进行考点复习") == "exam_prep"
        assert reg.match_skill("我想做题练习一下") == "quiz_drill"
        assert reg.match_skill("查看我的错题本") == "error_review"

    def test_no_match(self):
        reg = SkillRegistry("src/agent/skills/definitions")
        assert reg.match_skill("今天天气怎么样") is None

    def test_load_instruction(self):
        reg = SkillRegistry("src/agent/skills/definitions")
        inst = reg.load_instruction("exam_prep")
        assert inst is not None
        assert inst.name == "exam_prep"
        assert len(inst.raw_body) > 0

    def test_get_descriptions(self):
        reg = SkillRegistry("src/agent/skills/definitions")
        desc = reg.get_skill_descriptions_for_prompt()
        assert "exam_prep" in desc

    def test_get_metadata(self):
        reg = SkillRegistry("src/agent/skills/definitions")
        meta = reg.get_metadata("quiz_drill")
        assert meta is not None
        assert "quiz_generator" in meta.tools_required
