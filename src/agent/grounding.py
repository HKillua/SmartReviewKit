"""Evidence bundling and grounding evaluation helpers for Agent answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_SEGMENT_SPLIT_RE = re.compile(r"[。！？!?\n]+")

DEFAULT_LOW_EVIDENCE_MESSAGE = (
    "根据当前知识库证据，我暂时无法可靠地完整回答这个课程问题。"
    "请缩小范围、指定章节，或先补充相关课件后再试。"
)


class GroundingPolicyAction(str, Enum):
    NORMAL = "normal"
    CONSERVATIVE_REWRITE = "conservative_rewrite"
    LOW_EVIDENCE_WARNING = "low_evidence_warning"


@dataclass
class EvidenceBundle:
    citations: list[dict[str, Any]] = field(default_factory=list)
    evidence_summary: str = ""
    evidence_texts: list[str] = field(default_factory=list)
    source_count: int = 0
    query_trace_ids: list[str] = field(default_factory=list)
    evidence_tool: str = ""
    generation_mode: str = ""
    evaluation_mode: str = ""

    @property
    def has_evidence(self) -> bool:
        return self.source_count > 0 and bool(self.citations or self.evidence_summary)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "citations": list(self.citations),
            "source_count": self.source_count,
            "query_trace_ids": list(self.query_trace_ids),
            "evidence_tool": self.evidence_tool,
            "generation_mode": self.generation_mode,
            "evaluation_mode": self.evaluation_mode,
        }


@dataclass
class GroundingAssessment:
    score: float
    has_evidence: bool
    citation_count: int
    source_count: int
    policy_action: str = GroundingPolicyAction.NORMAL.value
    low_evidence: bool = False
    conservative_rewrite_used: bool = False
    valid_citation_indices: list[int] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "grounding_score": round(self.score, 4),
            "grounding_policy_action": self.policy_action,
            "has_evidence": self.has_evidence,
            "citation_count": self.citation_count,
            "source_count": self.source_count,
            "low_evidence": self.low_evidence,
            "conservative_rewrite_used": self.conservative_rewrite_used,
            "valid_citation_indices": list(self.valid_citation_indices),
        }


def _normalize_citation(citation: Any) -> dict[str, Any]:
    if hasattr(citation, "to_dict"):
        return dict(citation.to_dict())
    if isinstance(citation, dict):
        return dict(citation)
    return {"index": 0, "source": "unknown", "text_snippet": str(citation)}


def build_evidence_summary(citations: list[dict[str, Any]], limit: int = 4) -> str:
    lines: list[str] = []
    for citation in citations[:limit]:
        index = citation.get("index", 0)
        source = citation.get("source", "unknown")
        page = citation.get("page")
        title = citation.get("metadata", {}).get("title", "")
        snippet = str(citation.get("text_snippet", "")).strip()
        source_label = f"`{source}`"
        if page is not None:
            source_label += f" p.{page}"
        if title:
            source_label += f" · {title}"
        lines.append(f"[{index}] {source_label}: {snippet}")
    return "\n".join(lines)


def build_evidence_bundle(tool_name: str, metadata: dict[str, Any] | None) -> EvidenceBundle | None:
    if not metadata:
        return None
    raw_citations = metadata.get("citations") or []
    citations = [_normalize_citation(citation) for citation in raw_citations if citation]
    evidence_summary = str(metadata.get("evidence_summary", "") or "")
    evidence_texts = [
        str(value)
        for value in metadata.get("evidence_texts", [])
        if str(value).strip()
    ]
    source_count = int(metadata.get("source_count", len(citations) or 0) or 0)
    if not evidence_summary and citations:
        evidence_summary = build_evidence_summary(citations)
    if evidence_summary and source_count == 0:
        source_count = max(len(citations), 1)
    query_trace_ids = [
        str(value)
        for value in metadata.get("query_trace_ids", [])
        if str(value)
    ]
    query_trace_id = str(metadata.get("query_trace_id", "") or "")
    if query_trace_id and query_trace_id not in query_trace_ids:
        query_trace_ids.append(query_trace_id)
    if not metadata.get("grounding_capable") and not citations and not evidence_summary:
        return None
    return EvidenceBundle(
        citations=citations,
        evidence_summary=evidence_summary,
        evidence_texts=evidence_texts,
        source_count=source_count,
        query_trace_ids=query_trace_ids,
        evidence_tool=tool_name,
        generation_mode=str(metadata.get("generation_mode", "") or ""),
        evaluation_mode=str(metadata.get("evaluation_mode", "") or ""),
    )


def build_grounding_context(
    bundle: EvidenceBundle | None,
    *,
    course_task: bool,
) -> str:
    if bundle is not None and bundle.has_evidence:
        lines = [
            "你必须优先基于以下课程证据回答，不要编造超出证据范围的课程内容。",
            "若使用来源，请在对应句子后保留 `[1]`、`[2]` 这类引用标记。",
            "可用证据:",
            bundle.evidence_summary,
        ]
        return "\n".join(line for line in lines if line)

    if not course_task:
        return ""

    return (
        "当前没有可用课程证据。不要编造课程知识；如果这是课程型问题，"
        "请明确说明现有知识库证据不足，并引导用户缩小范围或补充资料。"
    )


def extract_citation_indices(text: str) -> list[int]:
    if not text:
        return []
    return [int(match) for match in _CITATION_RE.findall(text)]


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _extract_numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


def _strip_citation_markers(text: str) -> str:
    return _CITATION_RE.sub("", text or "").strip()


def _snippet_match_score(answer: str, snippet: str) -> float:
    normalized_answer = " ".join(_strip_citation_markers(answer).split()).lower()
    normalized_snippet = " ".join((snippet or "").split()).lower()
    if not normalized_answer or not normalized_snippet:
        return 0.0
    if normalized_answer in normalized_snippet or normalized_snippet in normalized_answer:
        return 1.0
    answer_numbers = _extract_numbers(normalized_answer)
    snippet_numbers = _extract_numbers(normalized_snippet)
    if answer_numbers and answer_numbers <= snippet_numbers:
        return 0.85
    answer_tokens = _tokenize(normalized_answer)
    snippet_tokens = _tokenize(normalized_snippet)
    if not answer_tokens or not snippet_tokens:
        return 0.0
    overlap = len(answer_tokens & snippet_tokens)
    return overlap / max(1, min(len(answer_tokens), len(snippet_tokens)))


def _answer_segments(text: str) -> list[str]:
    cleaned = _strip_citation_markers(text or "")
    segments = [segment.strip() for segment in _SEGMENT_SPLIT_RE.split(cleaned) if segment.strip()]
    return segments or ([cleaned] if cleaned else [])


class GroundingEvaluator:
    """Deterministic answer grounding scorer."""

    def __init__(self, threshold: float = 0.4) -> None:
        self.threshold = threshold

    def assess(
        self,
        answer: str,
        bundle: EvidenceBundle | None,
    ) -> GroundingAssessment:
        if not answer:
            return GroundingAssessment(
                score=1.0,
                has_evidence=bool(bundle and bundle.has_evidence),
                citation_count=0,
                source_count=bundle.source_count if bundle else 0,
            )

        if bundle is None or not bundle.has_evidence:
            return GroundingAssessment(
                score=0.0,
                has_evidence=False,
                citation_count=0,
                source_count=0,
                policy_action=GroundingPolicyAction.LOW_EVIDENCE_WARNING.value,
                low_evidence=True,
            )

        answer_core = _strip_citation_markers(answer)
        answer_tokens = _tokenize(answer_core)
        evidence_text = bundle.evidence_summary or " ".join(
            str(citation.get("text_snippet", "")) for citation in bundle.citations
        )
        evidence_tokens = _tokenize(evidence_text)
        overlap = 1.0 if not answer_tokens else len(answer_tokens & evidence_tokens) / len(answer_tokens)

        citation_indices = extract_citation_indices(answer)
        valid_indices = sorted(
            {
                index
                for index in citation_indices
                if 1 <= index <= len(bundle.citations)
            }
        )
        citation_score = 1.0 if valid_indices else 0.0
        cited_snippets = [
            str(
                bundle.citations[index - 1].get("grounding_text")
                or bundle.citations[index - 1].get("text_snippet", "")
            )
            for index in valid_indices
        ] or [
            str(citation.get("grounding_text") or citation.get("text_snippet", ""))
            for citation in bundle.citations
        ]
        if bundle.evidence_texts:
            cited_snippets.extend(bundle.evidence_texts)
        if not cited_snippets and bundle.evidence_summary:
            cited_snippets.append(bundle.evidence_summary)
        cited_snippets = [snippet for snippet in cited_snippets if snippet]
        snippet_match = max((_snippet_match_score(answer_core, snippet) for snippet in cited_snippets), default=0.0)
        segment_match = max(
            (
                _snippet_match_score(segment, snippet)
                for segment in _answer_segments(answer)
                for snippet in cited_snippets
            ),
            default=0.0,
        )
        short_fact_answer = len(answer_tokens) <= 8 or len(answer_core) <= 32
        if citation_score > 0 and short_fact_answer and max(snippet_match, segment_match) >= 0.5:
            score = 0.9
        elif citation_score > 0 and segment_match >= 0.72:
            score = 0.86
        else:
            overlap_weight = 0.35 if short_fact_answer else 0.55
            snippet_weight = 0.45 if short_fact_answer else 0.25
            citation_weight = 0.20
            score = round(
                (overlap_weight * overlap)
                + (snippet_weight * max(snippet_match, segment_match))
                + (citation_weight * citation_score),
                4,
            )
        low_evidence = score < self.threshold
        return GroundingAssessment(
            score=score,
            has_evidence=True,
            citation_count=len(valid_indices),
            source_count=bundle.source_count,
            policy_action=(
                GroundingPolicyAction.NORMAL.value
                if not low_evidence
                else GroundingPolicyAction.LOW_EVIDENCE_WARNING.value
            ),
            low_evidence=low_evidence,
            valid_citation_indices=valid_indices,
        )


def build_conservative_rewrite_messages(
    answer: str,
    bundle: EvidenceBundle,
) -> tuple[str, str]:
    system_prompt = (
        "你是一名严格遵守证据的课程助教。"
        "请将答案改写为保守、可追溯的版本："
        "只能保留被证据支持的结论，证据不足的部分要明确说明，"
        "并尽量保留 `[1]`、`[2]` 这类引用标记。"
    )
    user_prompt = (
        "原始答案:\n"
        f"{answer}\n\n"
        "可用证据:\n"
        f"{bundle.evidence_summary}\n\n"
        "请输出改写后的最终答案，不要补充证据之外的新结论。"
    )
    return system_prompt, user_prompt
