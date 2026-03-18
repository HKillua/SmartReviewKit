"""Task-level planner for high-level Agent routing decisions."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


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
    task_intent: TaskIntent
    selected_tool: str
    confidence: float
    source_span: tuple[int, int]
    match_method: str = "rule"
    segment_text: str = ""

    def to_metadata(self) -> Dict[str, object]:
        return {
            "task_intent": self.task_intent.value,
            "selected_tool": self.selected_tool,
            "confidence": round(self.confidence, 3),
            "source_span": [self.source_span[0], self.source_span[1]],
            "match_method": self.match_method,
            "segment_text": self.segment_text,
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

_COMPOSITE_TASK_PATTERNS: list[tuple[TaskIntent, re.Pattern[str]]] = [
    (TaskIntent.KNOWLEDGE_QUERY, _KNOWLEDGE_RULES),
    (TaskIntent.REVIEW_SUMMARY, _REVIEW_RULES),
    (TaskIntent.QUIZ_GENERATOR, _QUIZ_GENERATOR_RULES),
    (TaskIntent.QUIZ_EVALUATOR, _QUIZ_EVALUATOR_RULES),
]


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
    ) -> None:
        self._embedding_fn = embedding_fn
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
    ) -> PlannerDecision:
        if matched_skill and matched_skill in _SKILL_TO_INTENT:
            decision = _SKILL_TO_INTENT[matched_skill]
            return PlannerDecision(
                task_intent=decision.task_intent,
                confidence=decision.confidence,
                match_method=decision.match_method,
                control_mode=decision.control_mode,
                selected_tool=decision.selected_tool,
                planner_hint=decision.planner_hint,
            )

        composite_decision = self._composite_rule_match(message)
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

    def _composite_rule_match(self, message: str) -> PlannerDecision | None:
        text = message.strip()
        if not text:
            return None
        if _CHAT_RULES.match(text):
            return None
        if _DOCUMENT_RULES.search(text) or self._looks_like_file_ingest(text):
            return None
        if _QUIZ_EVALUATOR_RULES.search(text):
            return None

        ordered_hits = self._collect_composite_hits(text)
        if len(ordered_hits) < 2:
            return None
        if ordered_hits[0][0] in {TaskIntent.REVIEW_SUMMARY, TaskIntent.QUIZ_GENERATOR}:
            trailing_intents = {intent for intent, *_ in ordered_hits[1:]}
            if trailing_intents and trailing_intents <= {TaskIntent.KNOWLEDGE_QUERY}:
                return None

        subtasks: list[PlannedSubtask] = []
        for index, (intent, start, end, matched_text) in enumerate(ordered_hits):
            next_start = ordered_hits[index + 1][1] if index + 1 < len(ordered_hits) else len(text)
            segment_text = text[start:next_start].strip()
            if not segment_text:
                segment_text = matched_text
            subtasks.append(
                PlannedSubtask(
                    task_intent=intent,
                    selected_tool=intent.value,
                    confidence=1.0,
                    source_span=(start, end),
                    match_method="rule",
                    segment_text=segment_text,
                )
            )

        primary_intent = subtasks[0].task_intent
        return PlannerDecision(
            task_intent=primary_intent,
            confidence=1.0,
            match_method="rule_composite",
            control_mode=ControlMode.AUTONOMOUS,
            selected_tool=subtasks[0].selected_tool,
            planner_hint=self._build_composite_hint(subtasks, autonomous=True),
            is_composite=True,
            subtasks=subtasks,
            primary_intent=primary_intent,
            ordering_method="rule_span_order",
        )

    def _collect_composite_hits(
        self,
        message: str,
    ) -> list[tuple[TaskIntent, int, int, str]]:
        hits: list[tuple[TaskIntent, int, int, str]] = []
        seen_intents: set[TaskIntent] = set()
        for intent, pattern in _COMPOSITE_TASK_PATTERNS:
            match = pattern.search(message)
            if match is None or intent in seen_intents:
                continue
            seen_intents.add(intent)
            hits.append((intent, match.start(), match.end(), match.group(0)))
        hits.sort(key=lambda item: item[1])
        return hits

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
    def _build_composite_hint(subtasks: list[PlannedSubtask], *, autonomous: bool = False) -> str:
        ordered_tools = " -> ".join(subtask.selected_tool for subtask in subtasks)
        if autonomous:
            return (
                "这是复合学习任务。以下是初始分解建议："
                f"{ordered_tools}。请在观察结果后可控地调整下一步，但不要偏离用户目标。"
            )
        return f"这是复合学习任务，请严格按用户表达顺序依次执行子任务：{ordered_tools}。"

    @property
    def embedding_ready(self) -> bool:
        return self._embedding_ready
