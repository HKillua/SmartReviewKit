"""Task-level planner for high-level Agent routing decisions."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from src.agent.types import GoalStatus

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.agent.skills.registry import SkillPolicy


class TaskIntent(str, Enum):
    GENERAL_CHAT = "general_chat"
    KNOWLEDGE_QUERY = "knowledge_query"
    REVIEW_SUMMARY = "review_summary"
    QUIZ_GENERATOR = "quiz_generator"
    QUIZ_EVALUATOR = "quiz_evaluator"
    DOCUMENT_INGEST = "document_ingest"


class ControlMode(str, Enum):
    PASS_THROUGH = "pass_through"
    ADVISORY = "advisory"
    FORCE_TOOL = "force_tool"
    AUTONOMOUS = "autonomous"


@dataclass
class PlannedSubtask:
    goal_id: str
    task_intent: TaskIntent
    selected_tool: str
    confidence: float
    source_span: tuple[int, int]
    match_method: str = "rule"
    segment_text: str = ""
    required: bool = True
    depends_on_user_input: bool = False
    status: str = GoalStatus.PENDING.value

    def to_metadata(self) -> Dict[str, object]:
        return {
            "goal_id": self.goal_id,
            "task_intent": self.task_intent.value,
            "selected_tool": self.selected_tool,
            "confidence": round(self.confidence, 3),
            "source_span": [self.source_span[0], self.source_span[1]],
            "match_method": self.match_method,
            "segment_text": self.segment_text,
            "required": self.required,
            "depends_on_user_input": self.depends_on_user_input,
            "status": self.status,
        }


@dataclass
class PlannerDecision:
    task_intent: TaskIntent
    confidence: float = 0.0
    match_method: str = "default"
    control_mode: ControlMode = ControlMode.PASS_THROUGH
    selected_tool: str = ""
    planner_hint: str = ""
    is_composite: bool = False
    subtasks: list[PlannedSubtask] = field(default_factory=list)
    primary_intent: TaskIntent | None = None
    ordering_method: str = ""
    matched_skill: str = ""
    skill_start_index: int = -1
    skill_end_index: int = -1
    planner_execution_model: str = "legacy_single"

    def to_metadata(self) -> Dict[str, object]:
        return {
            "task_intent": self.task_intent.value,
            "confidence": round(self.confidence, 3),
            "match_method": self.match_method,
            "control_mode": self.control_mode.value,
            "selected_tool": self.selected_tool,
            "planner_hint": self.planner_hint,
            "is_composite": self.is_composite,
            "primary_intent": (
                self.primary_intent.value if self.primary_intent is not None else ""
            ),
            "ordering_method": self.ordering_method,
            "matched_skill": self.matched_skill,
            "skill_start_index": self.skill_start_index,
            "skill_end_index": self.skill_end_index,
            "planner_execution_model": self.planner_execution_model,
            "subtasks": [subtask.to_metadata() for subtask in self.subtasks],
        }


_QUIZ_GENERATOR_RULES = re.compile(
    r"出题|做题|练习|习题|刷题|测验|来\s*(?:\d+|[一二两三四五六七八九十])?\s*道(?:选择|填空|简答|SQL)?题|出\s*(?:\d+|[一二两三四五六七八九十])?\s*道(?:选择|填空|简答|SQL)?题|生成.*题|quiz",
    re.IGNORECASE,
)
_QUIZ_EVALUATOR_RULES = re.compile(
    r"判分|评分|批改|评判|评估答案|帮我判|检查答案|我的答案是|请帮我判",
    re.IGNORECASE,
)
_REVIEW_RULES = re.compile(
    r"复习|总结(?:一下)?(?:.*?(?:要点|重点))?|考点|回顾|梳理|复习摘要|复习重点|考试复习|期末复习",
    re.IGNORECASE,
)
_KNOWLEDGE_RULES = re.compile(
    r"解释|讲解|说明|为什么|原理|怎么理解|如何理解|详细介绍|详细解释|介绍(?:一下)?|概述|简述|流程|讲讲|聊聊|区别|对比|比较|分析|是什么",
    re.IGNORECASE,
)
_DOCUMENT_RULES = re.compile(
    r"导入知识库|导入.*知识库|入库|上传.*课件|上传.*资料|ingest|import",
    re.IGNORECASE,
)
_CHAT_RULES = re.compile(
    r"^(你好|hi|hello|谢谢|感谢|再见|bye|帮助|help|/help)$",
    re.IGNORECASE,
)
_EXPLICIT_SEQUENCE_RE = re.compile(r"(再|然后|接着|最后)")
_SEQUENCE_SPLIT_RE = re.compile(r"(再|然后|接着|最后)")

_SKILL_TO_INTENT: Dict[str, PlannerDecision] = {
    "quiz_drill": PlannerDecision(
        task_intent=TaskIntent.QUIZ_GENERATOR,
        confidence=1.0,
        match_method="skill",
        control_mode=ControlMode.AUTONOMOUS,
        selected_tool=TaskIntent.QUIZ_GENERATOR.value,
        planner_hint="这是习题训练任务，允许在受控范围内自主决定出题、判题和针对性解释。",
    ),
    "exam_prep": PlannerDecision(
        task_intent=TaskIntent.REVIEW_SUMMARY,
        confidence=1.0,
        match_method="skill",
        control_mode=ControlMode.AUTONOMOUS,
        selected_tool=TaskIntent.REVIEW_SUMMARY.value,
        planner_hint="这是考试复习任务，允许先看掌握情况，再决定检索、总结和补题顺序。",
    ),
    "chapter_deep_dive": PlannerDecision(
        task_intent=TaskIntent.KNOWLEDGE_QUERY,
        confidence=0.92,
        match_method="skill",
        control_mode=ControlMode.AUTONOMOUS,
        selected_tool=TaskIntent.KNOWLEDGE_QUERY.value,
        planner_hint="这是章节深入讲解任务，允许在知识检索、概念图谱和协议模拟之间自主选择。",
    ),
    "error_review": PlannerDecision(
        task_intent=TaskIntent.KNOWLEDGE_QUERY,
        confidence=0.82,
        match_method="skill",
        control_mode=ControlMode.AUTONOMOUS,
        selected_tool=TaskIntent.KNOWLEDGE_QUERY.value,
        planner_hint="这是错题回顾任务，允许先分析薄弱点，再决定讲解、模拟或补救练习。",
    ),
    "knowledge_check": PlannerDecision(
        task_intent=TaskIntent.KNOWLEDGE_QUERY,
        confidence=0.75,
        match_method="skill",
        control_mode=ControlMode.ADVISORY,
        selected_tool=TaskIntent.KNOWLEDGE_QUERY.value,
        planner_hint="这是知识掌握度检查任务，建议先获取相关知识和学习证据。",
    ),
}

_TASK_PROTOTYPES: Dict[TaskIntent, List[str]] = {
    TaskIntent.KNOWLEDGE_QUERY: [
        "TCP三次握手的过程是什么",
        "详细解释一下DNS解析流程",
        "为什么需要拥塞控制",
        "HTTP和HTTPS有什么区别",
    ],
    TaskIntent.REVIEW_SUMMARY: [
        "帮我总结一下DNS解析的复习要点",
        "给我梳理网络层的考点",
        "期末复习应该怎么复习TCP/IP",
    ],
    TaskIntent.QUIZ_GENERATOR: [
        "围绕UDP出3道选择题",
        "给我来几道关于HTTP的练习题",
        "帮我生成一组网络层测验",
    ],
    TaskIntent.QUIZ_EVALUATOR: [
        "这是我的答案请帮我判分",
        "帮我评估这道题的回答",
        "请批改我的答案",
    ],
    TaskIntent.DOCUMENT_INGEST: [
        "把这个pdf导入知识库",
        "请把课件入库",
        "import this ppt into the knowledge base",
    ],
    TaskIntent.GENERAL_CHAT: [
        "你好",
        "谢谢",
        "你能做什么",
    ],
}

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (norm_a * norm_b)


class TaskPlanner:
    """Rule-first planner with embedding fallback."""

    FORCE_THRESHOLD = 0.88
    ADVISORY_THRESHOLD = 0.72

    def __init__(
        self,
        *,
        embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        skill_match_fn: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self._embedding_fn = embedding_fn
        self._skill_match_fn = skill_match_fn
        self._prototypes: List[tuple[TaskIntent, List[float]]] = []
        self._embedding_ready = False
        if embedding_fn is not None:
            self._precompute_prototypes()

    def _precompute_prototypes(self) -> None:
        texts: List[str] = []
        labels: List[TaskIntent] = []
        for intent, examples in _TASK_PROTOTYPES.items():
            for example in examples:
                texts.append(example)
                labels.append(intent)
        try:
            vectors = self._embedding_fn(texts)
            self._prototypes = list(zip(labels, vectors))
            self._embedding_ready = True
            logger.info(
                "TaskPlanner: pre-computed %d prototype embeddings for %d task intents",
                len(self._prototypes),
                len(_TASK_PROTOTYPES),
            )
        except Exception:
            logger.warning(
                "TaskPlanner prototype embedding failed, using rule-only mode",
                exc_info=True,
            )
            self._embedding_ready = False

    def plan(
        self,
        message: str,
        *,
        matched_skill: str | None = None,
        skill_policy: SkillPolicy | None = None,
    ) -> PlannerDecision:
        composite_decision = self._explicit_sequence_composite_match(
            message,
            matched_skill=matched_skill or "",
            skill_policy=skill_policy,
        )
        if composite_decision is not None:
            return composite_decision

        rule_decision = self._rule_match(message)
        if rule_decision is not None:
            return rule_decision

        if self._embedding_ready and self._embedding_fn is not None:
            embedding_decision = self._embedding_match(message)
            if embedding_decision is not None:
                return embedding_decision

        return self._default_decision(message)

    def _rule_match(self, message: str) -> PlannerDecision | None:
        text = message.strip()
        if _CHAT_RULES.match(text):
            return self._build_decision(
                TaskIntent.GENERAL_CHAT,
                confidence=1.0,
                match_method="rule",
            )
        if _DOCUMENT_RULES.search(text) or self._looks_like_file_ingest(text):
            return self._build_decision(
                TaskIntent.DOCUMENT_INGEST,
                confidence=1.0,
                match_method="rule",
            )
        if _QUIZ_EVALUATOR_RULES.search(text):
            return self._build_decision(
                TaskIntent.QUIZ_EVALUATOR,
                confidence=1.0,
                match_method="rule",
            )
        if _QUIZ_GENERATOR_RULES.search(text):
            return self._build_decision(
                TaskIntent.QUIZ_GENERATOR,
                confidence=1.0,
                match_method="rule",
            )
        if _REVIEW_RULES.search(text):
            return self._build_decision(
                TaskIntent.REVIEW_SUMMARY,
                confidence=1.0,
                match_method="rule",
            )
        if _KNOWLEDGE_RULES.search(text):
            return self._build_decision(
                TaskIntent.KNOWLEDGE_QUERY,
                confidence=0.92,
                match_method="rule",
            )
        return None

    def _embedding_match(self, message: str) -> PlannerDecision | None:
        try:
            query_vector = self._embedding_fn([message])[0]
        except Exception:
            logger.warning("TaskPlanner embedding match failed", exc_info=True)
            return None

        best_intent: TaskIntent | None = None
        best_score = -1.0
        for intent, prototype in self._prototypes:
            score = _cosine_similarity(query_vector, prototype)
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_intent is None or best_score < self.ADVISORY_THRESHOLD:
            return None
        return self._build_decision(
            best_intent,
            confidence=best_score,
            match_method="embedding",
        )

    def _default_decision(self, message: str) -> PlannerDecision:
        if "?" in message or "？" in message:
            return self._build_decision(
                TaskIntent.KNOWLEDGE_QUERY,
                confidence=0.55,
                match_method="default",
            )
        return self._build_decision(
            TaskIntent.GENERAL_CHAT,
            confidence=0.0,
            match_method="default",
        )

    def _plan_segment(self, message: str) -> PlannerDecision:
        rule_decision = self._rule_match(message)
        if rule_decision is not None:
            return rule_decision
        if self._embedding_ready and self._embedding_fn is not None:
            embedding_decision = self._embedding_match(message)
            if embedding_decision is not None:
                return embedding_decision
        return self._default_decision(message)

    def _explicit_sequence_composite_match(
        self,
        message: str,
        *,
        matched_skill: str,
        skill_policy: SkillPolicy | None,
    ) -> PlannerDecision | None:
        segments = self._split_explicit_sequence_segments(message)
        if len(segments) < 2:
            return None

        subtasks: list[PlannedSubtask] = []
        for start, end, segment_text in segments:
            decision = self._plan_segment(segment_text)
            if decision.task_intent == TaskIntent.GENERAL_CHAT and not decision.selected_tool:
                return None
            subtasks.append(
                PlannedSubtask(
                    goal_id=f"goal_{len(subtasks) + 1}",
                    task_intent=decision.task_intent,
                    selected_tool=decision.selected_tool or decision.task_intent.value,
                    confidence=decision.confidence,
                    source_span=(start, end),
                    match_method=decision.match_method,
                    segment_text=segment_text,
                    depends_on_user_input=self._segment_requires_user_input(
                        decision.task_intent,
                        segment_text,
                    ),
                )
            )

        skill_start_index, skill_end_index = self._compute_skill_interval(
            subtasks,
            matched_skill=matched_skill,
            skill_policy=skill_policy,
        )
        primary_intent = subtasks[0].task_intent
        planner_hint = self._build_composite_hint(
            subtasks,
            autonomous=bool(
                skill_policy is not None
                and getattr(skill_policy, "allow_autonomous", False)
                and skill_start_index >= 0
            ),
            matched_skill=matched_skill if skill_start_index >= 0 else "",
            skill_start_index=skill_start_index,
            skill_end_index=skill_end_index,
        )
        return PlannerDecision(
            task_intent=primary_intent,
            confidence=1.0,
            match_method="rule_sequence_composite",
            control_mode=ControlMode.AUTONOMOUS,
            selected_tool=subtasks[0].selected_tool,
            planner_hint=planner_hint,
            is_composite=True,
            subtasks=subtasks,
            primary_intent=primary_intent,
            ordering_method="explicit_sequence_order",
            matched_skill=matched_skill if skill_start_index >= 0 else "",
            skill_start_index=skill_start_index,
            skill_end_index=skill_end_index,
            planner_execution_model=(
                "skill_guided_agenda" if skill_start_index >= 0 else "legacy_composite"
            ),
        )

    def _split_explicit_sequence_segments(
        self,
        message: str,
    ) -> list[tuple[int, int, str]]:
        text = message.strip()
        if not text or not _EXPLICIT_SEQUENCE_RE.search(text):
            return []

        matches: list[re.Match[str]] = []
        for match in _SEQUENCE_SPLIT_RE.finditer(text):
            marker = match.group(1)
            if marker in {"然后", "接着", "最后"}:
                matches.append(match)
                continue
            prefix = text[: match.start()]
            previous_char = prefix.rstrip()[-1:] if prefix.strip() else ""
            if previous_char in {"，", ",", "；", ";", "、"} or "先" in prefix:
                matches.append(match)
        if not matches:
            return []

        segments: list[tuple[int, int, str]] = []
        segment_start = 0
        for match in matches:
            if match.start() <= segment_start:
                continue
            segment_text = text[segment_start:match.start()].strip(" ，。！？；,!?;：:")
            if segment_text:
                segments.append((segment_start, match.start(), segment_text))
            segment_start = match.start()
        tail_text = text[segment_start:].strip(" ，。！？；,!?;：:")
        if tail_text:
            segments.append((segment_start, len(text), tail_text))
        if len(segments) < 2:
            return []
        return segments

    def _compute_skill_interval(
        self,
        subtasks: list[PlannedSubtask],
        *,
        matched_skill: str,
        skill_policy: SkillPolicy | None,
    ) -> tuple[int, int]:
        if not matched_skill or skill_policy is None or not subtasks:
            return -1, -1
        allowed_tools = {
            str(name) for name in getattr(skill_policy, "allowed_tools", []) if str(name).strip()
        }
        if not allowed_tools:
            return -1, -1

        skill_start_index = -1
        for index, subtask in enumerate(subtasks):
            if self._segment_matches_skill(subtask, matched_skill):
                skill_start_index = index
                break
        if skill_start_index < 0:
            return -1, -1

        max_steps = max(1, min(int(getattr(skill_policy, "max_steps", 5) or 5), 5))
        skill_end_index = skill_start_index
        covered_steps = 1
        if subtasks[skill_start_index].selected_tool not in allowed_tools:
            mapped_tool = self._mapped_skill_tool(matched_skill)
            if mapped_tool and mapped_tool in allowed_tools:
                subtasks[skill_start_index].selected_tool = mapped_tool
            else:
                subtasks[skill_start_index].selected_tool = list(allowed_tools)[0]

        for index in range(skill_start_index + 1, len(subtasks)):
            if covered_steps >= max_steps:
                break
            if subtasks[index].selected_tool not in allowed_tools:
                break
            skill_end_index = index
            covered_steps += 1
        return skill_start_index, skill_end_index

    def _segment_matches_skill(self, subtask: PlannedSubtask, matched_skill: str) -> bool:
        if not matched_skill:
            return False
        segment_text = str(subtask.segment_text or "").strip()
        if segment_text and self._skill_match_fn is not None:
            try:
                if self._skill_match_fn(segment_text) == matched_skill:
                    return True
            except Exception:
                logger.warning("TaskPlanner segment skill match failed", exc_info=True)
        mapped_intent = _SKILL_TO_INTENT.get(matched_skill)
        if mapped_intent is not None and subtask.task_intent == mapped_intent.task_intent:
            return True
        return False

    @staticmethod
    def _mapped_skill_tool(matched_skill: str) -> str:
        decision = _SKILL_TO_INTENT.get(matched_skill)
        return decision.selected_tool if decision is not None else ""

    @staticmethod
    def _segment_requires_user_input(intent: TaskIntent, segment_text: str) -> bool:
        if intent != TaskIntent.QUIZ_EVALUATOR:
            return False
        text = segment_text.strip()
        if not text:
            return True
        if re.search(r"(我的答案是|用户答案|正确答案|第\s*\d+\s*题|第[一二两三四五六七八九十]+\s*题)", text):
            return False
        return True

    @staticmethod
    def _looks_like_file_ingest(message: str) -> bool:
        lowered = message.lower()
        return ".pdf" in lowered or ".pptx" in lowered

    def _build_decision(
        self,
        intent: TaskIntent,
        *,
        confidence: float,
        match_method: str,
    ) -> PlannerDecision:
        if intent == TaskIntent.GENERAL_CHAT:
            return PlannerDecision(
                task_intent=intent,
                confidence=confidence,
                match_method=match_method,
                control_mode=ControlMode.PASS_THROUGH,
                planner_hint="这是通用对话任务，不需要强制工具规划。",
            )

        if intent == TaskIntent.KNOWLEDGE_QUERY:
            return PlannerDecision(
                task_intent=intent,
                confidence=confidence,
                match_method=match_method,
                control_mode=ControlMode.ADVISORY,
                selected_tool=intent.value,
                planner_hint="这是开放式知识问答，优先考虑调用 knowledge_query 获取课程证据，但不强制。",
            )

        if confidence >= self.FORCE_THRESHOLD:
            control_mode = ControlMode.FORCE_TOOL
        elif confidence >= self.ADVISORY_THRESHOLD:
            control_mode = ControlMode.ADVISORY
        else:
            control_mode = ControlMode.PASS_THROUGH

        hint_templates = {
            TaskIntent.REVIEW_SUMMARY: "这是复习任务，优先先调用 review_summary 生成结构化复习摘要。",
            TaskIntent.QUIZ_GENERATOR: "这是出题训练任务，优先先调用 quiz_generator 生成练习题。",
            TaskIntent.QUIZ_EVALUATOR: "这是判题任务，优先先调用 quiz_evaluator 评判答案。",
            TaskIntent.DOCUMENT_INGEST: "这是资料入库任务，优先先调用 document_ingest 导入文件。",
        }
        return PlannerDecision(
            task_intent=intent,
            confidence=confidence,
            match_method=match_method,
            control_mode=control_mode,
            selected_tool=intent.value,
            planner_hint=hint_templates.get(intent, ""),
        )

    @staticmethod
    def _build_composite_hint(
        subtasks: list[PlannedSubtask],
        *,
        autonomous: bool = False,
        matched_skill: str = "",
        skill_start_index: int = -1,
        skill_end_index: int = -1,
    ) -> str:
        ordered_tools = " -> ".join(subtask.selected_tool for subtask in subtasks)
        skill_note = ""
        if matched_skill and skill_start_index >= 0:
            skill_note = (
                f" 技能「{matched_skill}」在子任务 {skill_start_index + 1}"
                f" 到 {skill_end_index + 1} 区间内生效。"
            )
        if autonomous:
            return (
                "这是复合学习任务。以下是初始分解建议："
                f"{ordered_tools}。请在观察结果后可控地调整下一步，但不要偏离用户目标。"
                f"{skill_note}"
            )
        return (
            "这是复合学习任务，请严格按用户表达顺序依次执行子任务："
            f"{ordered_tools}。{skill_note}".strip()
        )

    @property
    def embedding_ready(self) -> bool:
        return self._embedding_ready
