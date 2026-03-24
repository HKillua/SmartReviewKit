"""Static course ontology + user mastery overlay for graph-style queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from src.agent.tools.base import Tool
from src.agent.types import ToolContext, ToolResult


class ConceptGraphQueryArgs(BaseModel):
    topic: str = Field(..., description="知识点名称或主题")
    query_type: str = Field(default="subtopics", description="查询类型: subtopics/prerequisites/confusable/review_order")
    limit: int = Field(default=8, ge=1, le=20)


class ConceptGraphQueryTool(Tool[ConceptGraphQueryArgs]):
    def __init__(
        self,
        *,
        ontology_path: str = "config/concept_ontology.computer_network.yaml",
        knowledge_map: Any = None,
        error_memory: Any = None,
    ) -> None:
        self._knowledge_map = knowledge_map
        self._error_memory = error_memory
        self._ontology_path = Path(ontology_path)
        self._concepts = self._load_ontology()
        self._alias_index = self._build_alias_index()

    @property
    def name(self) -> str:
        return "concept_graph_query"

    @property
    def description(self) -> str:
        return "查询计算机网络课程知识点的前置依赖、子主题、易混淆概念和复习顺序。"

    def get_args_schema(self) -> type[ConceptGraphQueryArgs]:
        return ConceptGraphQueryArgs

    def _load_ontology(self) -> dict[str, dict[str, Any]]:
        if not self._ontology_path.exists():
            return {}
        payload = yaml.safe_load(self._ontology_path.read_text(encoding="utf-8")) or {}
        concepts = payload.get("concepts", []) or []
        result: dict[str, dict[str, Any]] = {}
        for item in concepts:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or item.get("title") or "").strip()
            if not key:
                continue
            result[key] = {
                "key": key,
                "title": str(item.get("title") or key),
                "aliases": [str(alias) for alias in item.get("aliases", []) or []],
                "chapter": str(item.get("chapter") or ""),
                "summary": str(item.get("summary") or ""),
                "subtopics": [str(v) for v in item.get("subtopics", []) or []],
                "prerequisites": [str(v) for v in item.get("prerequisites", []) or []],
                "confusable": [str(v) for v in item.get("confusable", []) or []],
            }
        return result

    def _build_alias_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for key, node in self._concepts.items():
            aliases = [key, node["title"], *node.get("aliases", [])]
            for alias in aliases:
                normalized = str(alias).strip().lower()
                if normalized:
                    index[normalized] = key
        return index

    def _resolve_topic(self, topic: str) -> dict[str, Any] | None:
        normalized = topic.strip().lower()
        if normalized in self._alias_index:
            return self._concepts.get(self._alias_index[normalized])
        for alias, key in self._alias_index.items():
            if normalized and normalized in alias:
                return self._concepts.get(key)
        return None

    async def _mastery_for(self, user_id: str, concept: str) -> float | None:
        if self._knowledge_map is None or not hasattr(self._knowledge_map, "get_node"):
            return None
        node = await self._knowledge_map.get_node(user_id, concept)
        if node is None:
            return None
        return float(getattr(node, "mastery_level", 0.0))

    async def _weak_concepts(self, user_id: str) -> set[str]:
        if self._error_memory is None or not hasattr(self._error_memory, "get_weak_concepts"):
            return set()
        try:
            values = await self._error_memory.get_weak_concepts(user_id)
        except Exception:
            return set()
        return {str(value) for value in values}

    async def execute(self, context: ToolContext, args: ConceptGraphQueryArgs) -> ToolResult:
        node = self._resolve_topic(args.topic)
        if node is None:
            return ToolResult(success=False, error=f"未找到知识点: {args.topic}")

        weak_from_errors = await self._weak_concepts(context.user_id)
        query_type = str(args.query_type or "subtopics").strip().lower()
        if query_type == "subtopics":
            targets = list(node.get("subtopics", []))
            title = "子知识点"
        elif query_type == "prerequisites":
            targets = list(node.get("prerequisites", []))
            title = "前置依赖"
        elif query_type == "confusable":
            targets = list(node.get("confusable", []))
            title = "易混淆概念"
        elif query_type == "review_order":
            targets = list(dict.fromkeys(node.get("prerequisites", []) + node.get("subtopics", [])))
            title = "推荐复习顺序"
        else:
            return ToolResult(success=False, error=f"不支持的 query_type: {args.query_type}")

        rows: list[dict[str, Any]] = []
        for concept in targets[: args.limit]:
            concept_node = self._resolve_topic(concept) or self._concepts.get(concept) or {"title": concept, "chapter": "", "summary": ""}
            mastery = await self._mastery_for(context.user_id, concept_node.get("key", concept))
            weak = concept in weak_from_errors or (mastery is not None and mastery < 0.5)
            rows.append(
                {
                    "concept": concept_node.get("title", concept),
                    "chapter": concept_node.get("chapter", ""),
                    "summary": concept_node.get("summary", ""),
                    "mastery": mastery,
                    "weak": weak,
                }
            )

        if query_type == "review_order":
            rows.sort(key=lambda item: (item["mastery"] if item["mastery"] is not None else 0.4, 0 if item["weak"] else 1))

        lines = [
            f"主题: {node['title']}",
            f"查询类型: {title}",
        ]
        if node.get("summary"):
            lines.append(f"主题说明: {node['summary']}")
        lines.append("")
        for index, row in enumerate(rows, start=1):
            mastery_text = "掌握度未知" if row["mastery"] is None else f"掌握度 {row['mastery']:.0%}"
            weak_flag = " ⚠️薄弱" if row["weak"] else ""
            chapter = f"（{row['chapter']}）" if row["chapter"] else ""
            summary = f" - {row['summary']}" if row["summary"] else ""
            lines.append(f"{index}. {row['concept']}{chapter}，{mastery_text}{weak_flag}{summary}")

        return ToolResult(
            success=True,
            result_for_llm="\n".join(lines).strip(),
            metadata={
                "tool_output_kind": "analysis_context",
                "completion_hint": "continue",
                "graph_topic": node["title"],
                "graph_query_type": query_type,
                "graph_rows": rows,
            },
        )
