"""Helpers for batch quiz answer alignment and free-text splitting."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from src.agent.types import LlmMessage, LlmRequest
from src.agent.utils.json_helpers import safe_parse_json
from src.agent.utils.sanitizer import sanitize_user_input

_CN_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

_SECTION_MARKER_RE = re.compile(
    r"(?:(?:^|\n)\s*(?:第\s*(?P<cn>[一二两三四五六七八九十\d]+)\s*题|(?P<num>\d+)\s*[、\.\)])\s*)",
    re.IGNORECASE,
)
_ANSWER_ONLY_MARKER_RE = re.compile(
    r"(?:第\s*(?P<cn>[一二两三四五六七八九十\d]+)\s*题|(?P<num>\d+)\s*[、\.\)])",
    re.IGNORECASE,
)
_ANSWER_ONLY_PREFIX_RE = re.compile(
    r"^\s*(?:第\s*(?P<cn>[一二两三四五六七八九十\d]+)\s*题|(?P<num>\d+)\s*[、\.\)])",
    re.IGNORECASE,
)
_QUESTION_FIELD_RE = re.compile(
    r"(?:题目|问题)(?:内容)?(?:是)?[:：]?\s*(.+?)(?=(?:用户答案|我的答案|作答|回答|答案(?:是)?|正确答案|标准答案|$))",
    re.S,
)
_USER_ANSWER_FIELD_RE = re.compile(
    r"(?:用户答案|我的答案|作答|回答|答案(?:是)?)(?!.*(?:正确答案|标准答案))[:：]?\s*(.+?)(?=(?:正确答案|标准答案|$))",
    re.S,
)
_CORRECT_ANSWER_FIELD_RE = re.compile(r"(?:正确答案|标准答案)[:：]?\s*(.+)$", re.S)
_QUESTION_HEADER_RE = re.compile(r"以下是\s*\d+\s*道(?P<qtype>[^：:\n]+)")
_QUIZ_BLOCK_RE = re.compile(
    r"###\s*第\s*(?P<idx>\d+)\s*题(?P<body>.*?)(?=###\s*第\s*\d+\s*题|\Z)",
    re.S,
)
_ANSWER_LINE_RE = re.compile(r"\*\*答案\*\*:\s*(.+?)(?:\n|$)")
_CONCEPTS_LINE_RE = re.compile(r"\*\*涉及知识点\*\*:\s*(.+?)(?:\n|$)")
_DETAILS_SPLIT_RE = re.compile(r"<details>", re.I)
_QUESTION_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")

_FREE_TEXT_SPLIT_PROMPT = """你要做的是把一段自由文本作答拆成逐题结构，不能评分。

用户作答:
{message}

最近题目上下文（如果有）:
{quiz_context}

要求:
1. 如果能识别出多题作答，请输出 items 数组
2. 每个 item 包含:
   - index: 题号；无法确定则填 0
   - question_text: 识别出的题目；如果来自上下文可直接引用对应题目
   - answer_text: 用户对应这题的答案
   - answer_confidence: 0 到 1
   - alignment_notes: 简短说明怎么对齐的
3. 如果有无法归属的片段，放到 unmatched_segments
4. 如果整体仍不足以稳定拆题，clarification_needed 设为 true
5. 只输出合法 JSON，不要有额外文本

输出格式:
{{
  "items": [
    {{
      "index": 1,
      "question_text": "...",
      "answer_text": "...",
      "answer_confidence": 0.9,
      "alignment_notes": "..."
    }}
  ],
  "unmatched_segments": ["..."],
  "clarification_needed": false
}}
"""


@dataclass
class QuizBatchItem:
    question: str = ""
    user_answer: str = ""
    correct_answer: str = ""
    question_type: str = "选择题"
    topic: str = ""
    concepts: list[str] = field(default_factory=list)
    index: int = 0
    alignment_notes: str = ""
    answer_confidence: float = 1.0

    def to_tool_payload(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "user_answer": self.user_answer,
            "correct_answer": self.correct_answer,
            "question_type": self.question_type,
            "topic": self.topic,
            "concepts": list(self.concepts),
            "index": self.index,
            "alignment_notes": self.alignment_notes,
            "answer_confidence": self.answer_confidence,
        }


@dataclass
class QuizBatchAlignment:
    items: list[QuizBatchItem] = field(default_factory=list)
    alignment_mode: str = ""
    alignment_status: str = "clarification_required"
    split_confidence: float = 0.0
    clarification_reason: str = ""

    @property
    def is_aligned(self) -> bool:
        return self.alignment_status == "aligned"


def _parse_cn_number(token: str) -> int:
    cleaned = str(token or "").strip()
    if not cleaned:
        return 0
    if cleaned.isdigit():
        return int(cleaned)
    return int(_CN_NUMBERS.get(cleaned, 0))


def _normalize_space(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _normalize_question(value: str) -> str:
    tokens = _QUESTION_TOKEN_RE.findall((value or "").lower())
    return "".join(tokens)


def _question_similarity(a: str, b: str) -> float:
    norm_a = _normalize_question(a)
    norm_b = _normalize_question(b)
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0
    if norm_a in norm_b or norm_b in norm_a:
        return min(len(norm_a), len(norm_b)) / max(len(norm_a), len(norm_b))
    set_a = set(_QUESTION_TOKEN_RE.findall(norm_a))
    set_b = set(_QUESTION_TOKEN_RE.findall(norm_b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


def _default_topic(question: str, concepts: list[str]) -> str:
    if concepts:
        return concepts[0]
    return _normalize_space(question)[:80]


def _complete_item(item: QuizBatchItem) -> bool:
    return bool(_normalize_space(item.question) and _normalize_space(item.user_answer))


def _split_marked_sections(message: str) -> list[tuple[int, str]]:
    text = message or ""
    matches = list(_SECTION_MARKER_RE.finditer(text))
    if not matches:
        return []
    sections: list[tuple[int, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        raw = text[start:end].strip()
        index = _parse_cn_number(match.group("num") or match.group("cn") or "")
        sections.append((index, raw))
    return sections


def _extract_fields_from_section(section: str, *, default_index: int = 0) -> QuizBatchItem | None:
    text = section.strip()
    if not text:
        return None
    marker = _ANSWER_ONLY_PREFIX_RE.search(text)
    index = default_index
    if marker is not None:
        index = _parse_cn_number(marker.group("num") or marker.group("cn") or "") or default_index
        text = text[marker.end() :].strip(" ：:.-\n")

    question_match = _QUESTION_FIELD_RE.search(text)
    answer_match = _USER_ANSWER_FIELD_RE.search(text)
    correct_match = _CORRECT_ANSWER_FIELD_RE.search(text)

    question = _normalize_space(question_match.group(1)) if question_match else ""
    user_answer = _normalize_space(answer_match.group(1)) if answer_match else ""
    correct_answer = _normalize_space(correct_match.group(1)) if correct_match else ""

    if not user_answer and not question and re.search(r"(我觉得|我认为|答案是|应该是|因为)", text):
        user_answer = _normalize_space(text)

    if not question and "题目是" in text:
        prefix, _, rest = text.partition("题目是")
        question_guess = rest
        for splitter in ("我的答案是", "用户答案", "答案是", "回答是", "作答是"):
            if splitter in question_guess:
                question_guess, _, answer_guess = question_guess.partition(splitter)
                question = _normalize_space(question_guess)
                if not user_answer:
                    user_answer = _normalize_space(answer_guess)
                break
        else:
            question = _normalize_space(question_guess)

    if not question and not user_answer and not correct_answer:
        return None

    return QuizBatchItem(
        question=question,
        user_answer=user_answer,
        correct_answer=correct_answer,
        index=index,
    )


def extract_explicit_quiz_items(message: str) -> list[QuizBatchItem]:
    sections = _split_marked_sections(message)
    items: list[QuizBatchItem] = []
    if sections:
        for index, section in sections:
            parsed = _extract_fields_from_section(section, default_index=index)
            if parsed is not None:
                items.append(parsed)
        if items:
            return items

    if sum(1 for _ in re.finditer(r"题目[:：]", message or "")) > 1:
        raw_sections = re.split(r"(?=题目[:：])", message or "")
        for section in raw_sections:
            parsed = _extract_fields_from_section(section)
            if parsed is not None:
                items.append(parsed)
        if items:
            return items

    single = _extract_fields_from_section(message or "")
    return [single] if single is not None else []


def extract_numbered_answer_blocks(message: str) -> list[QuizBatchItem]:
    items: list[QuizBatchItem] = []
    for index, section in _split_marked_sections(message):
        parsed = _extract_fields_from_section(section, default_index=index)
        body = section
        marker = _ANSWER_ONLY_MARKER_RE.search(body)
        if marker is not None:
            body = body[marker.end() :]
        normalized_body = _normalize_space(body)
        if parsed is None:
            if not normalized_body:
                continue
            parsed = QuizBatchItem(index=index, user_answer=normalized_body)
        elif not parsed.user_answer:
            parsed.user_answer = normalized_body
        items.append(parsed)
    return items


def extract_recent_quiz_bundle(recent_messages: list[dict[str, Any]]) -> list[QuizBatchItem]:
    assistant_text = ""
    for message in reversed(recent_messages or []):
        if str(message.get("role", "")) != "assistant":
            continue
        content = str(message.get("content", "") or "")
        if "### 第" in content and "**答案**:" in content:
            assistant_text = content
            break
    if not assistant_text:
        return []

    qtype_match = _QUESTION_HEADER_RE.search(assistant_text)
    question_type = _normalize_space(qtype_match.group("qtype")) if qtype_match else "选择题"
    bundle: list[QuizBatchItem] = []
    for match in _QUIZ_BLOCK_RE.finditer(assistant_text):
        body = match.group("body") or ""
        details_split = _DETAILS_SPLIT_RE.split(body, maxsplit=1)
        question_text = _normalize_space(details_split[0])
        answer_match = _ANSWER_LINE_RE.search(body)
        correct_answer = _normalize_space(answer_match.group(1)) if answer_match else ""
        concepts_match = _CONCEPTS_LINE_RE.search(body)
        concepts = []
        if concepts_match:
            concepts = [
                _normalize_space(part)
                for part in re.split(r"[，,、]", concepts_match.group(1))
                if _normalize_space(part)
            ]
        index = int(match.group("idx"))
        bundle.append(
            QuizBatchItem(
                question=question_text,
                correct_answer=correct_answer,
                question_type=question_type or "选择题",
                topic=_default_topic(question_text, concepts),
                concepts=concepts,
                index=index,
            )
        )
    return bundle


def _align_to_bundle(items: list[QuizBatchItem], bundle: list[QuizBatchItem]) -> tuple[list[QuizBatchItem], bool]:
    if not items or not bundle:
        return items, False
    aligned_any = False
    used_indices: set[int] = set()
    aligned: list[QuizBatchItem] = []
    for item in items:
        candidate: QuizBatchItem | None = None
        if item.index and 1 <= item.index <= len(bundle):
            candidate = bundle[item.index - 1]
        elif item.question:
            best_score = 0.0
            best_item: QuizBatchItem | None = None
            for source in bundle:
                if source.index in used_indices:
                    continue
                score = _question_similarity(item.question, source.question)
                if score > best_score:
                    best_score = score
                    best_item = source
            if best_item is not None and best_score >= 0.45:
                candidate = best_item
                item.answer_confidence = max(item.answer_confidence, round(best_score, 2))
        if candidate is not None:
            aligned_any = True
            used_indices.add(candidate.index)
            if not item.index:
                item.index = candidate.index
            if not item.question:
                item.question = candidate.question
            if not item.correct_answer:
                item.correct_answer = candidate.correct_answer
            if not item.question_type:
                item.question_type = candidate.question_type
            if not item.topic:
                item.topic = candidate.topic
            if not item.concepts:
                item.concepts = list(candidate.concepts)
            if not item.alignment_notes:
                item.alignment_notes = (
                    f"根据最近一轮题目上下文对齐到第 {candidate.index} 题。"
                )
        aligned.append(item)
    return aligned, aligned_any


def _items_complete(items: list[QuizBatchItem]) -> bool:
    return bool(items) and all(_complete_item(item) for item in items)


def _build_quiz_context(bundle: list[QuizBatchItem]) -> str:
    if not bundle:
        return "无最近题目上下文"
    lines = []
    for item in bundle[:5]:
        lines.append(
            f"{item.index}. 题目: {sanitize_user_input(item.question, max_length=160)} | "
            f"标准答案: {sanitize_user_input(item.correct_answer, max_length=80)} | "
            f"知识点: {', '.join(item.concepts[:3])}"
        )
    return "\n".join(lines)


async def _llm_split_free_text(
    *,
    llm_service: Any,
    message: str,
    bundle: list[QuizBatchItem],
) -> tuple[list[QuizBatchItem], float, bool]:
    if llm_service is None or not _normalize_space(message):
        return [], 0.0, True
    prompt = _FREE_TEXT_SPLIT_PROMPT.format(
        message=sanitize_user_input(message, max_length=1800),
        quiz_context=_build_quiz_context(bundle),
    )
    request = LlmRequest(
        messages=[
            LlmMessage(role="system", content="你是一名严谨的题答结构化助手，只输出合法 JSON。"),
            LlmMessage(role="user", content=prompt),
        ],
        temperature=0.0,
        max_tokens=700,
        stream=False,
    )
    response = await llm_service.send_request(request)
    parsed = safe_parse_json(response.content or "", fallback={}) if response and not response.error else {}
    raw_items = parsed.get("items", []) if isinstance(parsed, dict) else []
    clarification_needed = bool(parsed.get("clarification_needed", False)) if isinstance(parsed, dict) else True
    items: list[QuizBatchItem] = []
    confidences: list[float] = []
    for raw in raw_items if isinstance(raw_items, list) else []:
        if not isinstance(raw, dict):
            continue
        confidence = float(raw.get("answer_confidence", 0.0) or 0.0)
        items.append(
            QuizBatchItem(
                index=int(raw.get("index", 0) or 0),
                question=_normalize_space(str(raw.get("question_text", "") or "")),
                user_answer=_normalize_space(str(raw.get("answer_text", "") or "")),
                answer_confidence=confidence,
                alignment_notes=_normalize_space(str(raw.get("alignment_notes", "") or "")),
            )
        )
        confidences.append(confidence)
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    return items, avg_confidence, clarification_needed


async def build_quiz_batch_alignment(
    *,
    message: str,
    recent_messages: list[dict[str, Any]] | None = None,
    llm_service: Any = None,
    max_items: int = 5,
) -> QuizBatchAlignment:
    raw_text = (message or "").strip()
    text = _normalize_space(message)
    bundle = extract_recent_quiz_bundle(recent_messages or [])

    explicit_items = extract_explicit_quiz_items(raw_text)
    if explicit_items:
        explicit_items, used_bundle = _align_to_bundle(explicit_items, bundle)
        if len(explicit_items) > max_items:
            return QuizBatchAlignment(
                items=explicit_items[:max_items],
                alignment_mode="explicit_message" if not used_bundle else "hybrid_split",
                alignment_status="clarification_required",
                split_confidence=1.0 if _items_complete(explicit_items[:max_items]) else 0.75,
                clarification_reason=f"本次最多支持 {max_items} 题，请拆分后再批改。",
            )
        if _items_complete(explicit_items):
            return QuizBatchAlignment(
                items=explicit_items,
                alignment_mode="explicit_message" if not used_bundle else "hybrid_split",
                alignment_status="aligned",
                split_confidence=min(1.0, max(item.answer_confidence for item in explicit_items)),
            )

    answer_items = extract_numbered_answer_blocks(raw_text)
    if answer_items and bundle:
        answer_items, _ = _align_to_bundle(answer_items, bundle)
        if len(answer_items) > max_items:
            return QuizBatchAlignment(
                items=answer_items[:max_items],
                alignment_mode="recent_quiz_context",
                alignment_status="clarification_required",
                split_confidence=0.8,
                clarification_reason=f"本次最多支持 {max_items} 题，请拆分后再批改。",
            )
        if _items_complete(answer_items):
            return QuizBatchAlignment(
                items=answer_items,
                alignment_mode="recent_quiz_context",
                alignment_status="aligned",
                split_confidence=min(1.0, max(item.answer_confidence for item in answer_items)),
            )

    llm_items: list[QuizBatchItem] = []
    llm_confidence = 0.0
    llm_clarification = True
    if llm_service is not None and text:
        llm_items, llm_confidence, llm_clarification = await _llm_split_free_text(
            llm_service=llm_service,
            message=text,
            bundle=bundle,
        )
        if llm_items:
            llm_items, used_bundle = _align_to_bundle(llm_items, bundle)
            if len(llm_items) > max_items:
                return QuizBatchAlignment(
                    items=llm_items[:max_items],
                    alignment_mode="llm_free_text_split" if not used_bundle else "hybrid_split",
                    alignment_status="clarification_required",
                    split_confidence=llm_confidence,
                    clarification_reason=f"本次最多支持 {max_items} 题，请拆分后再批改。",
                )
            if _items_complete(llm_items) and not llm_clarification:
                return QuizBatchAlignment(
                    items=llm_items,
                    alignment_mode="llm_free_text_split" if not used_bundle else "hybrid_split",
                    alignment_status="aligned",
                    split_confidence=llm_confidence,
                )

    partial_items = explicit_items or answer_items or llm_items
    reason = "未能稳定识别每道题的题目与答案对应关系，请按题号逐条作答，或补充原题内容。"
    if not bundle and not explicit_items and not answer_items:
        reason = "当前消息里缺少足够的题目边界信息，也没有可用的上一轮题目上下文，请按题号逐条作答。"
    return QuizBatchAlignment(
        items=partial_items[:max_items],
        alignment_mode="llm_free_text_split" if llm_items else ("recent_quiz_context" if bundle else "explicit_message"),
        alignment_status="clarification_required" if not partial_items else "partial_alignment",
        split_confidence=llm_confidence,
        clarification_reason=reason,
    )
