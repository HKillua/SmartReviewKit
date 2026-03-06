"""Question bank parser — extracts individual questions as atomic chunks.

Recognises common Chinese and English question boundaries (e.g. "第1题",
"1.", "1)", "(1)") and splits question bank documents into one Chunk per
question.  Structured fields (options, answer, explanation) are stored in
chunk metadata so they can be reused at quiz time without LLM generation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.core.types import Chunk

_BOUNDARY_PATTERN = re.compile(
    r"(?:^|\n)\s*"
    r"(?:"
    r"第?\s*[一二三四五六七八九十百\d]+\s*[题.|、：:]"  # 第1题 / 一、
    r"|[\d]+\s*[.)．、]"                               # 1. / 1) / 1、
    r"|\(\s*[\d]+\s*\)"                                # (1)
    r"|\[\s*[\d]+\s*\]"                                # [1]
    r"|Question\s+\d+"                                 # Question 1
    r")",
    re.IGNORECASE,
)

_ANSWER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:答案|Answer|正确答案|参考答案)\s*[:：]\s*(.*)",
    re.IGNORECASE,
)

_EXPLANATION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:解析|解答|Explanation|详解|分析)\s*[:：]\s*([\s\S]*?)(?=\n\s*(?:答案|解析|第|[\d]+[.)．]|\Z))",
    re.IGNORECASE,
)

_OPTION_PATTERN = re.compile(
    r"^\s*([A-Fa-f])\s*[.．)）、:：]\s*(.+)$",
    re.MULTILINE,
)

_DIFFICULTY_KEYWORDS: Dict[int, List[str]] = {
    1: ["基础", "简单", "basic", "easy"],
    2: ["理解", "掌握", "understand"],
    3: ["应用", "分析", "综合", "apply", "analyze"],
    4: ["设计", "优化", "design", "advanced"],
    5: ["拓展", "研究", "前沿", "research"],
}


@dataclass
class ParsedQuestion:
    """A single parsed question with structured fields."""

    raw_text: str
    question_text: str = ""
    options: Dict[str, str] = field(default_factory=dict)
    answer: str = ""
    explanation: str = ""
    difficulty: int = 3
    question_type: str = "unknown"
    index: int = 0


class QuestionParser:
    """Parses question bank documents into structured :class:`ParsedQuestion` objects."""

    def parse(self, text: str) -> List[ParsedQuestion]:
        """Split *text* into individual questions."""
        boundaries = list(_BOUNDARY_PATTERN.finditer(text))
        if not boundaries:
            return []

        raw_blocks: List[str] = []
        for i, m in enumerate(boundaries):
            start = m.start()
            end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
            block = text[start:end].strip()
            if block:
                raw_blocks.append(block)

        questions: List[ParsedQuestion] = []
        for idx, block in enumerate(raw_blocks):
            pq = self._parse_single(block, idx)
            if pq.question_text.strip():
                questions.append(pq)

        return questions

    def to_chunks(
        self,
        questions: List[ParsedQuestion],
        source_path: str,
        doc_id: str = "",
    ) -> List[Chunk]:
        """Convert parsed questions into :class:`Chunk` objects."""
        chunks: List[Chunk] = []
        for q in questions:
            content_hash = hashlib.sha256(q.raw_text.encode("utf-8")).hexdigest()[:8]
            chunk_id = f"{doc_id}_qb_{q.index:04d}_{content_hash}"

            metadata: Dict = {
                "source_path": source_path,
                "source_type": "question_bank",
                "chunk_index": q.index,
                "question_type": q.question_type,
                "difficulty": q.difficulty,
                "source_ref": doc_id,
            }
            if q.answer:
                metadata["answer"] = q.answer
            if q.options:
                metadata["options"] = q.options
            if q.explanation:
                metadata["explanation"] = q.explanation

            chunks.append(Chunk(id=chunk_id, text=q.raw_text, metadata=metadata))

        return chunks

    # ------------------------------------------------------------------

    def _parse_single(self, block: str, index: int) -> ParsedQuestion:
        pq = ParsedQuestion(raw_text=block, index=index)

        answer_match = _ANSWER_PATTERN.search(block)
        if answer_match:
            pq.answer = answer_match.group(1).strip()

        explanation_match = _EXPLANATION_PATTERN.search(block)
        if explanation_match:
            pq.explanation = explanation_match.group(1).strip()

        options = dict(_OPTION_PATTERN.findall(block))
        if options:
            pq.options = {k.upper(): v.strip() for k, v in options.items()}
            pq.question_type = "choice"
        elif re.search(r"_{3,}|（\s*）|_+\s*_+", block):
            pq.question_type = "fill_blank"
        elif re.search(r"(?:简答|论述|分析|说明|阐述)", block):
            pq.question_type = "short_answer"
        else:
            pq.question_type = "unknown"

        body_end = answer_match.start() if answer_match else len(block)
        body = block[:body_end]
        body = _OPTION_PATTERN.sub("", body)
        body = _BOUNDARY_PATTERN.sub("", body, count=1).strip()
        pq.question_text = body

        pq.difficulty = self._infer_difficulty(block)
        return pq

    @staticmethod
    def _infer_difficulty(text: str) -> int:
        text_lower = text.lower()
        for level in (5, 4, 3, 2, 1):
            for kw in _DIFFICULTY_KEYWORDS[level]:
                if kw in text_lower:
                    return level
        return 3
