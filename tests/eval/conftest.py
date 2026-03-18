from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from src.agent.agent import Agent
from src.agent.config import AgentConfig
from src.agent.conversation import ConversationStore
from src.agent.hooks.review_schedule import ReviewScheduleHook
from src.agent.memory.knowledge_map import KnowledgeNode
from src.agent.memory.student_profile import StudentProfile
from src.agent.planner import TaskPlanner
from src.agent.skills.registry import SkillPolicy
from src.agent.skills.workflow import WorkflowResult
from src.agent.tools.base import Tool, ToolRegistry
from src.agent.tools.concept_graph_query import ConceptGraphQueryTool
from src.agent.tools.network_calc import NetworkCalcTool
from src.agent.tools.protocol_state_simulator import ProtocolStateSimulatorTool
from src.agent.types import Conversation, LlmMessage, LlmResponse, Message, ToolCallData, ToolContext, ToolResult

_LOW_INFO_RE = re.compile(r"^(你好|您好|hi|hello|hey|在吗)$", re.IGNORECASE)
_NETWORK_HINT_RE = re.compile(r"(子网|掩码|cidr|crc|香农|奈奎斯特|吞吐|时延|窗口|/\d{1,2})", re.IGNORECASE)
_PROTOCOL_HINT_RE = re.compile(r"(三次握手|四次挥手|拥塞控制|syn|ack|fin|丢了|丢包|超时|rip)", re.IGNORECASE)


@dataclass
class EvalCase:
    name: str
    user_message: str
    user_id: str
    mock_state: dict[str, Any]
    expectations: dict[str, Any]
    path: Path


class InMemoryConversationStore(ConversationStore):
    def __init__(self, initial: Conversation | None = None) -> None:
        self._conversations: dict[tuple[str, str], Conversation] = {}
        if initial is not None:
            self._conversations[(initial.id, initial.user_id)] = initial

    async def get(self, conversation_id: str, user_id: str) -> Conversation | None:
        return self._conversations.get((conversation_id, user_id))

    async def create(self, user_id: str) -> Conversation:
        conversation = Conversation(id=uuid.uuid4().hex[:16], user_id=user_id, messages=[])
        self._conversations[(conversation.id, user_id)] = conversation
        return conversation

    async def update(self, conversation: Conversation) -> None:
        self._conversations[(conversation.id, conversation.user_id)] = conversation

    async def list_conversations(self, user_id: str, limit: int = 20) -> list[Conversation]:
        rows = [conversation for (conv_id, owner), conversation in self._conversations.items() if owner == user_id]
        rows.sort(key=lambda item: item.updated_at, reverse=True)
        return rows[:limit]

    async def delete(self, conversation_id: str, user_id: str) -> bool:
        return self._conversations.pop((conversation_id, user_id), None) is not None


class MockKnowledgeMap:
    def __init__(self, nodes: list[dict[str, Any]] | None = None) -> None:
        self._nodes = [self._build_node(node) for node in (nodes or [])]

    @staticmethod
    def _build_node(payload: dict[str, Any]) -> KnowledgeNode:
        last_reviewed = payload.get("last_reviewed")
        if isinstance(payload.get("last_reviewed_days"), (int, float)):
            last_reviewed = datetime.now(timezone.utc) - timedelta(days=float(payload["last_reviewed_days"]))
        return KnowledgeNode(
            concept=str(payload.get("concept") or payload.get("title") or ""),
            chapter=str(payload.get("chapter") or ""),
            mastery_level=float(payload.get("mastery") if payload.get("mastery") is not None else payload.get("mastery_level", 0.0)),
            last_reviewed=last_reviewed,
            review_interval_days=float(payload.get("review_interval_days") or 2.0),
        )

    async def apply_decay(self, user_id: str) -> int:
        now = datetime.now(timezone.utc)
        updated = 0
        for node in self._nodes:
            if node.last_reviewed is None:
                continue
            current = node.last_reviewed
            if current.tzinfo is None:
                current = current.replace(tzinfo=timezone.utc)
            days = max((now - current).days, 0)
            if days <= 0:
                continue
            new_mastery = max(0.0, round(node.mastery_level * (0.92 ** days), 3))
            if new_mastery != node.mastery_level:
                node.mastery_level = new_mastery
                updated += 1
        return updated

    async def get_decayed_nodes(self, user_id: str, *, threshold: float = 0.45, limit: int = 5) -> list[KnowledgeNode]:
        await self.apply_decay(user_id)
        rows = [node for node in self._nodes if node.mastery_level < threshold]
        rows.sort(key=lambda node: (node.mastery_level, node.concept))
        return rows[:limit]

    async def get_due_for_review(self, user_id: str) -> list[KnowledgeNode]:
        return await self.get_decayed_nodes(user_id, threshold=0.6, limit=5)

    async def get_weak_nodes(self, user_id: str, threshold: float = 0.5) -> list[KnowledgeNode]:
        rows = [node for node in self._nodes if node.mastery_level < threshold]
        rows.sort(key=lambda node: (node.mastery_level, node.concept))
        return rows

    async def get_node(self, user_id: str, concept: str) -> KnowledgeNode | None:
        for node in self._nodes:
            if node.concept == concept:
                return node
        return None

    async def _get_all_nodes(self, user_id: str) -> list[KnowledgeNode]:
        return list(self._nodes)


class MockErrorMemory:
    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self._records = list(records or [])

    async def get_weak_concepts(self, user_id: str) -> list[str]:
        ranked = sorted(
            self._records,
            key=lambda row: (-int(row.get("count", 1)), str(row.get("concept", ""))),
        )
        return [str(row.get("concept", "")) for row in ranked if row.get("concept")]

    async def get_errors(self, user_id: str, topic: str | None = None, mastered: bool | None = None, limit: int = 50):
        topic_lower = (topic or "").lower()
        rows = []
        for record in self._records:
            concept = str(record.get("concept", ""))
            if topic_lower and topic_lower not in concept.lower():
                continue
            rows.append(
                type(
                    "MockError",
                    (),
                    {
                        "topic": concept,
                        "question": str(record.get("question") or f"{concept} 相关错题"),
                        "concepts": [concept] if concept else [],
                    },
                )()
            )
        return rows[:limit]


class MockStudentProfileMemory:
    def __init__(self, profile: dict[str, Any] | None = None) -> None:
        payload = dict(profile or {})
        last_active = None
        if isinstance(payload.get("last_active_days"), (int, float)):
            last_active = datetime.now(timezone.utc) - timedelta(days=float(payload["last_active_days"]))
        self._profile = StudentProfile(
            user_id=str(payload.get("user_id") or "eval_user"),
            weak_topics=list(payload.get("weak_topics", [])),
            strong_topics=list(payload.get("strong_topics", [])),
            learning_pace=str(payload.get("learning_pace") or "medium"),
            last_active=last_active,
        )

    async def get_profile(self, user_id: str) -> StudentProfile:
        self._profile.user_id = user_id
        return self._profile

    async def update_profile(self, user_id: str, updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if hasattr(self._profile, key):
                setattr(self._profile, key, value)


class MockSessionMemory:
    async def get_recent_sessions(self, user_id: str, limit: int = 1):
        return []


class EvalKnowledgeQueryArgs(BaseModel):
    query: str = Field(default="")
    top_k: int = Field(default=3)
    collection: str = Field(default="")


class EvalKnowledgeQueryTool(Tool[EvalKnowledgeQueryArgs]):
    def __init__(self, retrieval_state: dict[str, Any] | None = None) -> None:
        self._retrieval = dict(retrieval_state or {})

    @property
    def name(self) -> str:
        return "knowledge_query"

    @property
    def description(self) -> str:
        return "mock knowledge retrieval tool"

    def get_args_schema(self) -> type[EvalKnowledgeQueryArgs]:
        return EvalKnowledgeQueryArgs

    def _lookup(self, query: str) -> list[dict[str, Any]]:
        if not self._retrieval:
            return []
        normalized = query.strip().lower()
        exact = self._retrieval.get(normalized)
        if isinstance(exact, list):
            return exact
        for key, value in self._retrieval.items():
            if key == "default":
                continue
            if str(key).lower() in normalized and isinstance(value, list):
                return value
        default = self._retrieval.get("default")
        return list(default or [])

    async def execute(self, context: ToolContext, args: EvalKnowledgeQueryArgs) -> ToolResult:
        rows = self._lookup(args.query)
        if not rows:
            return ToolResult(
                success=True,
                result_for_llm="未找到与查询相关的知识库内容。请尝试换一种表述重新提问。",
                metadata={
                    "tool_output_kind": "final_answer",
                    "final_response_preferred": True,
                    "source_count": 0,
                    "grounding_capable": True,
                },
            )
        citations = []
        lines = ["以下是与问题最相关的课程证据。", ""]
        for index, row in enumerate(rows, start=1):
            source = str(row.get("source") or f"source_{index}.md")
            text = str(row.get("text") or "")
            citations.append({"index": index, "source": source, "chunk_id": f"mock_{index}"})
            lines.append(f"[{index}] `{source}`")
            lines.append(text)
            lines.append("")
        return ToolResult(
            success=True,
            result_for_llm="\n".join(lines).strip(),
            metadata={
                "tool_output_kind": "evidence_context",
                "grounding_capable": True,
                "citations": citations,
                "evidence_summary": "\n".join(f"[{item['index']}] {item['source']}" for item in citations),
                "source_count": len(citations),
            },
        )


class EvalReviewSummaryArgs(BaseModel):
    topic: str = Field(default="")


class EvalReviewSummaryTool(Tool[EvalReviewSummaryArgs]):
    @property
    def name(self) -> str:
        return "review_summary"

    @property
    def description(self) -> str:
        return "mock review summary tool"

    def get_args_schema(self) -> type[EvalReviewSummaryArgs]:
        return EvalReviewSummaryArgs

    async def execute(self, context: ToolContext, args: EvalReviewSummaryArgs) -> ToolResult:
        text = f"复习摘要：建议先掌握 {args.topic or '当前主题'} 的核心概念，再重点巩固薄弱点。"
        return ToolResult(
            success=True,
            result_for_llm=text,
            metadata={"tool_output_kind": "analysis_context", "source_count": 1},
        )


class EvalQuizGeneratorArgs(BaseModel):
    topic: str = Field(default="")
    count: int = Field(default=3)
    question_type: str = Field(default="选择题")
    difficulty: int = Field(default=3)


class EvalQuizGeneratorTool(Tool[EvalQuizGeneratorArgs]):
    @property
    def name(self) -> str:
        return "quiz_generator"

    @property
    def description(self) -> str:
        return "mock quiz generator"

    def get_args_schema(self) -> type[EvalQuizGeneratorArgs]:
        return EvalQuizGeneratorArgs

    async def execute(self, context: ToolContext, args: EvalQuizGeneratorArgs) -> ToolResult:
        lines = [f"已生成 {args.count} 道{args.question_type}："]
        for index in range(1, args.count + 1):
            lines.append(f"{index}. {args.topic or '计算机网络'} 相关练习题")
        return ToolResult(
            success=True,
            result_for_llm="\n".join(lines),
            metadata={
                "tool_output_kind": "final_answer",
                "final_response_preferred": True,
                "grounding_passthrough": True,
                "source_count": 1,
            },
        )


class EvalQuizEvaluatorArgs(BaseModel):
    question: str = Field(default="")
    user_answer: str = Field(default="")
    correct_answer: str = Field(default="")
    topic: str = Field(default="")
    question_type: str = Field(default="选择题")
    concepts: list[str] = Field(default_factory=list)


class EvalQuizEvaluatorTool(Tool[EvalQuizEvaluatorArgs]):
    @property
    def name(self) -> str:
        return "quiz_evaluator"

    @property
    def description(self) -> str:
        return "mock quiz evaluator"

    def get_args_schema(self) -> type[EvalQuizEvaluatorArgs]:
        return EvalQuizEvaluatorArgs

    async def execute(self, context: ToolContext, args: EvalQuizEvaluatorArgs) -> ToolResult:
        is_correct = bool(args.correct_answer and args.user_answer and args.correct_answer.strip() == args.user_answer.strip())
        verdict = "✅ 正确" if is_correct else "❌ 错误"
        text = f"判定: {verdict}\n题目: {args.question}\n解析: 建议结合 {args.topic or '当前知识点'} 继续复习。"
        return ToolResult(
            success=True,
            result_for_llm=text,
            metadata={"tool_output_kind": "final_answer", "final_response_preferred": True, "source_count": 0},
        )


class EvalDocumentIngestArgs(BaseModel):
    file_path: str = Field(default="")
    collection: str = Field(default="")


class EvalDocumentIngestTool(Tool[EvalDocumentIngestArgs]):
    @property
    def name(self) -> str:
        return "document_ingest"

    @property
    def description(self) -> str:
        return "mock document ingest"

    def get_args_schema(self) -> type[EvalDocumentIngestArgs]:
        return EvalDocumentIngestArgs

    async def execute(self, context: ToolContext, args: EvalDocumentIngestArgs) -> ToolResult:
        return ToolResult(
            success=True,
            result_for_llm=f"已将 {args.file_path} 导入知识库集合 {args.collection or 'default'}。",
            metadata={
                "tool_output_kind": "final_answer",
                "final_response_preferred": True,
                "grounding_passthrough": True,
                "source_count": 0,
            },
        )


class EvalSkillWorkflow:
    async def try_handle(self, user_message: str, user_id: str):
        text = (user_message or "").strip()
        if ("总结" in text or "复习" in text) and ("出题" in text or "道题" in text):
            return WorkflowResult()
        if "复习" in text:
            return WorkflowResult(
                matched_skill="exam_prep",
                skill_instruction="先看图谱，再决定检索和总结顺序。",
                skill_policy=SkillPolicy(
                    allowed_tools=["concept_graph_query", "knowledge_query", "review_summary", "quiz_generator", "network_calc"],
                    required_memory=["knowledge_map", "error_memory", "student_profile"],
                    allow_autonomous=True,
                    max_steps=5,
                    output_contract=["review_summary"],
                ),
            )
        if "错题" in text:
            return WorkflowResult(
                matched_skill="error_review",
                skill_instruction="先分析错因，再做补救讲解。",
                skill_policy=SkillPolicy(
                    allowed_tools=["concept_graph_query", "knowledge_query", "quiz_generator", "protocol_state_simulator"],
                    required_memory=["error_memory", "knowledge_map"],
                    allow_autonomous=True,
                    max_steps=4,
                    output_contract=["error_analysis"],
                ),
            )
        if any(token in text for token in ("章节", "深入", "讲讲")):
            return WorkflowResult(
                matched_skill="chapter_deep_dive",
                skill_instruction="先查依赖，再决定是检索还是模拟。",
                skill_policy=SkillPolicy(
                    allowed_tools=["concept_graph_query", "knowledge_query", "protocol_state_simulator", "network_calc"],
                    required_memory=["knowledge_map"],
                    allow_autonomous=True,
                    max_steps=4,
                    output_contract=["structured_explanation"],
                ),
            )
        if any(token in text for token in ("出题", "练习", "刷题")):
            return WorkflowResult(
                matched_skill="quiz_drill",
                skill_instruction="出题后可继续判题。",
                skill_policy=SkillPolicy(
                    allowed_tools=["quiz_generator", "quiz_evaluator", "knowledge_query", "network_calc"],
                    required_memory=["error_memory", "knowledge_map"],
                    allow_autonomous=True,
                    max_steps=4,
                    output_contract=["quiz"],
                ),
            )
        return WorkflowResult()


class EvalHeuristicLlm:
    def __init__(self, case: EvalCase) -> None:
        self._case = case
        self._tool_counter = 0

    def _available_tools(self, request) -> set[str]:
        return {
            schema.get("function", {}).get("name", "")
            for schema in (request.tools or [])
            if schema.get("function", {}).get("name")
        }

    @staticmethod
    def _latest_user_message(messages: list[LlmMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user" and message.content:
                return message.content
        return ""

    @staticmethod
    def _last_tool(messages: list[LlmMessage]) -> tuple[str, str]:
        mapping: dict[str, str] = {}
        last_name = ""
        last_content = ""
        for message in messages:
            if message.role == "assistant" and message.tool_calls:
                for call in message.tool_calls:
                    mapping[str(call.id)] = str(call.name)
            elif message.role == "tool" and message.tool_call_id:
                last_name = mapping.get(str(message.tool_call_id), "")
                last_content = message.content or ""
        return last_name, last_content

    def _tool_call(self, name: str, arguments: dict[str, Any]) -> LlmResponse:
        self._tool_counter += 1
        return LlmResponse(
            content=f"调用 {name}",
            tool_calls=[ToolCallData(id=f"eval_tool_{self._tool_counter}", name=name, arguments=arguments)],
        )

    def _topic(self, user_text: str) -> str:
        if "TCP" in user_text or "tcp" in user_text:
            return "TCP"
        if "传输层" in user_text:
            return "传输层"
        if "DNS" in user_text or "dns" in user_text:
            return "DNS"
        return "计算机网络"

    def _network_calc_args(self) -> dict[str, Any]:
        payload = dict(self._case.mock_state.get("network_calc_args", {}) or {})
        if payload:
            return payload
        message = self._case.user_message
        if "crc" in message.lower():
            return {"type": "crc", "params": {"data_bits": "1101011011", "generator_bits": "10011"}}
        return {
            "type": "subnet_division",
            "params": {"network": "192.168.1.0/24", "num_subnets": 4},
        }

    def _protocol_args(self) -> dict[str, Any]:
        payload = dict(self._case.mock_state.get("protocol_args", {}) or {})
        if payload:
            return payload
        if "丢" in self._case.user_message:
            return {
                "protocol": "tcp_handshake",
                "params": {"initial_seq": 100, "server_seq": 500},
                "fault_injection": {"drop_packet": 2},
            }
        return {"protocol": "tcp_handshake", "params": {"initial_seq": 100}, "fault_injection": {}}

    async def send_request(self, request):
        system_text = next(
            (message.content or "" for message in request.messages if message.role == "system"),
            "",
        )
        user_text = self._latest_user_message(request.messages)
        available_tools = self._available_tools(request)
        last_tool_name, last_tool_content = self._last_tool(request.messages)

        if last_tool_name:
            if "## [Replan Signal]" in system_text:
                if last_tool_name == "protocol_state_simulator" and "knowledge_query" in available_tools:
                    return self._tool_call(
                        "knowledge_query",
                        {"query": user_text or "TCP 三次握手", "top_k": 3},
                    )
                if last_tool_name == "knowledge_query" and "concept_graph_query" in available_tools:
                    return self._tool_call(
                        "concept_graph_query",
                        {"topic": self._case.mock_state.get("graph_topic", "传输层"), "query_type": "subtopics"},
                    )
            if last_tool_name == "concept_graph_query" and "knowledge_query" in available_tools:
                weak_concepts = self._case.mock_state.get("error_memory", [])
                focus = weak_concepts[0]["concept"] if weak_concepts else self._case.mock_state.get("graph_topic", "传输层")
                return self._tool_call("knowledge_query", {"query": str(focus), "top_k": 3})
            if last_tool_name == "review_summary" and "quiz_generator" in available_tools and ("出" in user_text or "练习" in user_text):
                return self._tool_call(
                    "quiz_generator",
                    {
                        "topic": self._topic(user_text),
                        "count": 2,
                        "question_type": "选择题",
                        "difficulty": 3,
                    },
                )
            return LlmResponse(content=last_tool_content or "已完成当前步骤。")

        if _LOW_INFO_RE.match(user_text or ""):
            return LlmResponse(content="你好，我可以继续帮你复习计算机网络。")
        if "document_ingest" in available_tools and re.search(r"\.(pdf|pptx)\b", user_text or "", re.IGNORECASE):
            return self._tool_call(
                "document_ingest",
                {"file_path": user_text.strip(), "collection": "eval_collection"},
            )
        if "network_calc" in available_tools and _NETWORK_HINT_RE.search(user_text or ""):
            return self._tool_call("network_calc", self._network_calc_args())
        if "protocol_state_simulator" in available_tools and _PROTOCOL_HINT_RE.search(user_text or ""):
            return self._tool_call("protocol_state_simulator", self._protocol_args())
        if "总结" in user_text and "review_summary" in available_tools:
            return self._tool_call("review_summary", {"topic": self._topic(user_text)})
        if "复习" in user_text and "concept_graph_query" in available_tools:
            return self._tool_call(
                "concept_graph_query",
                {"topic": self._case.mock_state.get("graph_topic", "传输层"), "query_type": "subtopics"},
            )
        if any(token in user_text for token in ("出题", "练习", "刷题", "道题")) and "quiz_generator" in available_tools:
            return self._tool_call(
                "quiz_generator",
                {
                    "topic": self._topic(user_text),
                    "count": 2,
                    "question_type": "选择题",
                    "difficulty": 3,
                },
            )
        if any(token in user_text for token in ("判", "批改", "评分")) and "quiz_evaluator" in available_tools:
            return self._tool_call(
                "quiz_evaluator",
                {
                    "question": "示例题",
                    "user_answer": "A",
                    "correct_answer": "A",
                    "topic": self._topic(user_text),
                },
            )
        if "knowledge_query" in available_tools:
            return self._tool_call("knowledge_query", {"query": user_text, "top_k": 3})
        return LlmResponse(content="你好，我可以帮助你进行计算机网络学习。")


def _build_prior_conversation(mock_state: dict[str, Any], user_id: str, conversation_id: str) -> Conversation:
    messages: list[Message] = []
    for index, outcome in enumerate(mock_state.get("quiz_outcomes", []) or [], start=1):
        call_id = f"quiz_history_{index}"
        messages.append(
            Message(
                role="assistant",
                content=None,
                tool_calls=[ToolCallData(id=call_id, name="quiz_evaluator", arguments={})],
            )
        )
        if outcome == "correct":
            content = "判定: ✅ 正确"
        elif outcome == "incorrect":
            content = "判定: ❌ 错误"
        else:
            content = "判定: ⚠️ 部分正确"
        messages.append(Message(role="tool", tool_call_id=call_id, content=content))
    return Conversation(id=conversation_id, user_id=user_id, messages=messages)


def build_agent_for_case(case: EvalCase) -> Agent:
    knowledge_map = MockKnowledgeMap(case.mock_state.get("knowledge_map"))
    error_memory = MockErrorMemory(case.mock_state.get("error_memory"))
    student_profile = MockStudentProfileMemory(case.mock_state.get("profile"))
    session_memory = MockSessionMemory()
    review_hook = ReviewScheduleHook(
        knowledge_map=knowledge_map,
        error_memory=error_memory,
        student_profile=student_profile,
        session_memory=session_memory,
    )

    conversation_id = f"eval_{case.path.stem}"
    store = InMemoryConversationStore(
        _build_prior_conversation(case.mock_state, case.user_id, conversation_id)
    )

    registry = ToolRegistry()
    registry.register(EvalKnowledgeQueryTool(case.mock_state.get("retrieval")))
    registry.register(EvalReviewSummaryTool())
    registry.register(EvalQuizGeneratorTool())
    registry.register(EvalQuizEvaluatorTool())
    registry.register(EvalDocumentIngestTool())
    registry.register(NetworkCalcTool())
    registry.register(
        ConceptGraphQueryTool(
            ontology_path="config/concept_ontology.computer_network.yaml",
            knowledge_map=knowledge_map,
            error_memory=error_memory,
        )
    )
    registry.register(ProtocolStateSimulatorTool())

    return Agent(
        llm_service=EvalHeuristicLlm(case),
        tool_registry=registry,
        conversation_store=store,
        config=AgentConfig(stream_responses=False, max_tool_iterations=5, response_profile="quality_first"),
        task_planner=TaskPlanner(),
        skill_workflow=EvalSkillWorkflow(),
        review_hook=review_hook,
    )


def load_eval_case(path: Path) -> EvalCase:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return EvalCase(
        name=str(payload.get("name") or path.stem),
        user_message=str(payload.get("user_message") or ""),
        user_id=str(payload.get("user_id") or "eval_user"),
        mock_state=dict(payload.get("mock_state") or {}),
        expectations=dict(payload.get("expectations") or {}),
        path=path,
    )
