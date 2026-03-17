"""Adaptive retrieval-policy router for ``knowledge_query``.

**Layer 1 (Rule):** Fast keyword regex matching for retrieval preferences.
**Layer 2 (Embedding):** Cosine similarity against pre-computed prototype
embeddings for ambiguous knowledge-query variants.

The router does **not** own top-level task selection; that remains the job of
``TaskPlanner``. This module only derives retrieval policy for knowledge
queries, such as preferred ``source_type`` collections and fallback strategy.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Internal retrieval-policy labels.

    These labels are used only to derive retrieval policy inside
    ``knowledge_query`` and are not a second top-level user intent system.
    """

    QUIZ_REQUEST = "quiz_request"
    CONCEPT_REVIEW = "concept_review"
    DEEP_UNDERSTANDING = "deep_understanding"
    GENERAL_CHAT = "general_chat"


@dataclass
class RoutingDecision:
    """Output of the retrieval-policy router."""

    intent: QueryIntent
    need_rag: bool = True
    preferred_sources: List[str] = field(default_factory=lambda: ["slide", "textbook"])
    retrieval_strategy: str = "full"  # full | dense_only | sparse_only
    fallback_to_llm: bool = True
    source_filter: Optional[Dict] = None
    source_weights: Dict[str, float] = field(default_factory=dict)
    source_raw_overfetch: Dict[str, int] = field(default_factory=dict)
    normalization_profile: str = "source_aware_units"
    evidence_profile: str = "default"
    confidence: float = 1.0
    match_method: str = "rule"  # rule | embedding | default

    def to_metadata_filter(self) -> Optional[Dict]:
        """Build a ChromaDB-compatible metadata filter for ``source_type``."""
        if not self.preferred_sources:
            return self.source_filter
        if len(self.preferred_sources) == 1:
            return {"source_type": self.preferred_sources[0]}
        return {"source_type": {"$in": self.preferred_sources}}

    def compute_source_unit_budgets(self, top_k: int) -> Dict[str, int]:
        """Allocate per-source answer-unit budget using largest remainder."""
        if top_k <= 0:
            return {}
        positive = {
            source: float(weight)
            for source, weight in (self.source_weights or {}).items()
            if float(weight) > 0.0
        }
        if not positive:
            ordered = list(self.preferred_sources)
            if not ordered:
                return {}
            return {source: int(index < top_k) for index, source in enumerate(ordered)}

        total_weight = sum(positive.values()) or 1.0
        normalized = {
            source: weight / total_weight for source, weight in positive.items()
        }

        base_allocations: Dict[str, int] = {}
        remainders: list[tuple[float, str]] = []
        reserved = 0
        min_slots = min(top_k, len(normalized))
        ordered_sources = self.preferred_sources or sorted(
            normalized,
            key=lambda key: normalized[key],
            reverse=True,
        )

        for source in ordered_sources:
            if source not in normalized:
                continue
            quota = normalized[source] * top_k
            floor_value = int(math.floor(quota))
            if min_slots >= len(normalized):
                floor_value = max(floor_value, 1)
            base_allocations[source] = floor_value
            reserved += floor_value
            remainders.append((quota - math.floor(quota), source))

        while reserved < top_k:
            remainders.sort(key=lambda item: (-item[0], ordered_sources.index(item[1])))
            remainder, source = remainders[0]
            base_allocations[source] = base_allocations.get(source, 0) + 1
            reserved += 1
            remainders[0] = (0.0 if remainder > 0 else remainder, source)

        while reserved > top_k:
            removable = [
                source
                for source in ordered_sources
                if base_allocations.get(source, 0) > 0
                and not (min_slots >= len(normalized) and base_allocations[source] <= 1)
            ]
            if not removable:
                break
            source = removable[-1]
            base_allocations[source] -= 1
            reserved -= 1

        return {source: value for source, value in base_allocations.items() if value > 0}


# ── Intent prototype queries for embedding classification ──────────────

_INTENT_PROTOTYPES: Dict[QueryIntent, List[str]] = {
    QueryIntent.QUIZ_REQUEST: [
        "出三道关于TCP的选择题",
        "帮我生成一些练习题",
        "来几道关于路由算法的填空题",
        "做一组HTTP协议的测试题",
        "出一套网络层的模拟考试",
        "给我出几道关于子网划分的题",
        "模拟一下期末考试的题目",
        "quiz me on DNS",
    ],
    QueryIntent.CONCEPT_REVIEW: [
        "帮我复习TCP/IP协议栈",
        "总结一下DNS的考点",
        "归纳网络层的重点知识",
        "梳理传输层的核心内容",
        "回顾一下OSI七层模型",
        "列出路由协议的知识点",
        "帮我整理HTTP的要点",
        "review the key concepts of IP addressing",
    ],
    QueryIntent.DEEP_UNDERSTANDING: [
        "TCP的拥塞控制原理是什么",
        "为什么需要三次握手而不是两次",
        "详细解释ARP协议的工作过程",
        "HTTP和HTTPS有什么区别",
        "子网划分是如何计算的",
        "滑动窗口机制是怎么保证可靠传输的",
        "OSPF和RIP的区别在哪里",
        "explain how NAT works in detail",
    ],
    QueryIntent.GENERAL_CHAT: [
        "你好",
        "谢谢你的帮助",
        "再见",
        "你能做什么",
        "帮助",
        "hello",
        "thanks",
        "what can you do",
    ],
}

# ── Intent → RoutingDecision template mapping ──────────────────────────

_INTENT_DECISION_MAP: Dict[QueryIntent, Dict] = {
    QueryIntent.GENERAL_CHAT: dict(
        need_rag=False,
        preferred_sources=[],
        fallback_to_llm=True,
    ),
    QueryIntent.QUIZ_REQUEST: dict(
        need_rag=True,
        preferred_sources=["question_bank", "slide", "textbook"],
    ),
    QueryIntent.CONCEPT_REVIEW: dict(
        need_rag=True,
        preferred_sources=["slide", "textbook"],
    ),
    QueryIntent.DEEP_UNDERSTANDING: dict(
        need_rag=True,
        preferred_sources=["textbook", "slide"],
    ),
}

_DEFAULT_SOURCE_RAW_OVERFETCH: Dict[str, int] = {
    "question_bank": 2,
    "slide": 2,
    "textbook": 4,
}

_TASK_SOURCE_PROFILES: Dict[str, Dict[str, object]] = {
    "knowledge_query": {
        "source_weights": {"textbook": 0.55, "slide": 0.30, "question_bank": 0.15},
        "evidence_profile": "explanatory",
    },
    "review_summary": {
        "source_weights": {"slide": 0.45, "textbook": 0.40, "question_bank": 0.15},
        "evidence_profile": "summary",
    },
    "quiz_generator": {
        "source_weights": {"question_bank": 0.60, "textbook": 0.25, "slide": 0.15},
        "evidence_profile": "practice_generation",
    },
    "quiz_evaluator": {
        "source_weights": {"question_bank": 0.50, "textbook": 0.30, "slide": 0.20},
        "evidence_profile": "answer_grading",
    },
}

_TASK_DEFAULT_INTENT: Dict[str, QueryIntent] = {
    "knowledge_query": QueryIntent.DEEP_UNDERSTANDING,
    "review_summary": QueryIntent.CONCEPT_REVIEW,
    "quiz_generator": QueryIntent.QUIZ_REQUEST,
    "quiz_evaluator": QueryIntent.QUIZ_REQUEST,
}

# ── Regex patterns for Layer 1 (fast path) ─────────────────────────────

_QUIZ_KEYWORDS = re.compile(
    r"出题|练习|测[试验]|quiz|做题|习题|考[试题]|模拟|生成.*题|来.*道题|出.*道",
    re.IGNORECASE,
)
_REVIEW_KEYWORDS = re.compile(
    r"复习|总结|考点|review|归纳|梳理|重点|回顾|知识点",
    re.IGNORECASE,
)
_DEEP_KEYWORDS = re.compile(
    r"解释|为什么|原理|详细|如何|怎[么样]|区别|对比|比较|举例|深入",
    re.IGNORECASE,
)
_CHAT_KEYWORDS = re.compile(
    r"^(你好|hi|hello|谢谢|感谢|再见|bye|帮助|help|/help)$",
    re.IGNORECASE,
)

_RULE_MATCHERS: List[Tuple[re.Pattern, QueryIntent, bool]] = [
    (_CHAT_KEYWORDS, QueryIntent.GENERAL_CHAT, True),   # fullmatch
    (_QUIZ_KEYWORDS, QueryIntent.QUIZ_REQUEST, False),
    (_REVIEW_KEYWORDS, QueryIntent.CONCEPT_REVIEW, False),
    (_DEEP_KEYWORDS, QueryIntent.DEEP_UNDERSTANDING, False),
]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


class QueryRouter:
    """Dual-layer intent classifier: rule fast-path + embedding vector classification.

    Parameters:
        fallback_to_llm: Whether RAG-miss results should fall back to LLM.
        embedding_fn: ``(texts: list[str]) -> list[list[float]]`` — optional
            embedding function (typically ``CachedEmbedding.embed``).  When
            provided, enables Layer 2 embedding classification.
        similarity_threshold: Minimum cosine similarity for an embedding
            match to be accepted (default 0.75).
    """

    def __init__(
        self,
        *,
        fallback_to_llm: bool = True,
        embedding_fn: Optional[Callable] = None,
        similarity_threshold: float = 0.75,
    ) -> None:
        self._fallback = fallback_to_llm
        self._embed_fn = embedding_fn
        self._sim_threshold = similarity_threshold

        # Pre-computed prototype embeddings: list of (intent, embedding)
        self._prototypes: List[Tuple[QueryIntent, List[float]]] = []
        self._embedding_ready = False

        if self._embed_fn is not None:
            self._precompute_prototypes()

    # ── Prototype pre-computation ──────────────────────────────────────

    def _precompute_prototypes(self) -> None:
        """Embed all intent prototypes at init time (one-off cost)."""
        all_texts: List[str] = []
        intent_labels: List[QueryIntent] = []
        for intent, examples in _INTENT_PROTOTYPES.items():
            for text in examples:
                all_texts.append(text)
                intent_labels.append(intent)

        try:
            vectors = self._embed_fn(all_texts)
            self._prototypes = list(zip(intent_labels, vectors))
            self._embedding_ready = True
            logger.info(
                "QueryRouter: pre-computed %d prototype embeddings for %d intents",
                len(self._prototypes),
                len(_INTENT_PROTOTYPES),
            )
        except Exception:
            logger.warning(
                "QueryRouter: prototype embedding failed, falling back to rule-only mode",
                exc_info=True,
            )
            self._embedding_ready = False

    # ── Public API ─────────────────────────────────────────────────────

    def route(
        self,
        query: str,
        context: Optional[List[Dict]] = None,
        *,
        planner_task_intent: str | None = None,
        matched_skill: str | None = None,
    ) -> RoutingDecision:
        """Derive retrieval policy for *query* and return a routing decision.

        Layer 1 (rule) is tried first. On miss, Layer 2 (embedding) runs
        if an ``embedding_fn`` was provided. When upstream planner context is
        provided and is not ``knowledge_query``, the router becomes a no-op
        retrieval-policy passthrough instead of acting like a second task
        classifier.
        """
        q = query.strip()
        planner_intent = str(planner_task_intent or "").strip()
        if planner_intent and planner_intent not in _TASK_SOURCE_PROFILES:
            return self._build_passthrough_decision(method="planner_context")

        if planner_intent and planner_intent != "knowledge_query":
            decision = self._build_decision(
                _TASK_DEFAULT_INTENT[planner_intent],
                confidence=1.0,
                method="planner_context",
            )
            return self._apply_planner_context(decision, planner_intent)

        # ── Layer 1: rule fast-path ──
        rule_result = self._rule_match(q)
        if rule_result is not None:
            return self._apply_planner_context(rule_result, planner_intent)

        if planner_intent == "knowledge_query":
            return self._apply_planner_context(
                self._build_decision(
                    QueryIntent.DEEP_UNDERSTANDING,
                    confidence=0.0,
                    method="planner_context_default",
                ),
                planner_intent,
            )

        # ── Layer 2: embedding classification ──
        if self._embedding_ready and self._embed_fn is not None:
            embed_result = self._embedding_match(q)
            if embed_result is not None:
                return self._apply_planner_context(embed_result, planner_intent)

        # ── Default fallback ──
        return self._apply_planner_context(
            self._build_decision(
                QueryIntent.DEEP_UNDERSTANDING,
                confidence=0.0,
                method="default",
            ),
            planner_intent,
        )

    # ── Layer 1: Rule matching ─────────────────────────────────────────

    def _rule_match(self, query: str) -> Optional[RoutingDecision]:
        for pattern, intent, is_fullmatch in _RULE_MATCHERS:
            if is_fullmatch:
                if pattern.match(query):
                    return self._build_decision(intent, confidence=1.0, method="rule")
            else:
                if pattern.search(query):
                    return self._build_decision(intent, confidence=1.0, method="rule")
        return None

    # ── Layer 2: Embedding matching ────────────────────────────────────

    def _embedding_match(self, query: str) -> Optional[RoutingDecision]:
        try:
            query_vec = self._embed_fn([query])[0]
        except Exception:
            logger.debug("QueryRouter embedding call failed", exc_info=True)
            return None

        best_intent: Optional[QueryIntent] = None
        best_sim = -1.0

        for intent, proto_vec in self._prototypes:
            sim = _cosine_similarity(query_vec, proto_vec)
            if sim > best_sim:
                best_sim = sim
                best_intent = intent

        if best_intent is not None and best_sim >= self._sim_threshold:
            logger.debug(
                "QueryRouter embedding match: intent=%s sim=%.3f query='%s'",
                best_intent.value,
                best_sim,
                query[:60],
            )
            return self._build_decision(
                best_intent, confidence=best_sim, method="embedding",
            )

        logger.debug(
            "QueryRouter embedding miss: best_sim=%.3f (threshold=%.3f) query='%s'",
            best_sim,
            self._sim_threshold,
            query[:60],
        )
        return None

    # ── Decision builder ───────────────────────────────────────────────

    def _build_decision(
        self,
        intent: QueryIntent,
        confidence: float = 1.0,
        method: str = "rule",
    ) -> RoutingDecision:
        template = _INTENT_DECISION_MAP.get(intent, {})
        return RoutingDecision(
            intent=intent,
            need_rag=template.get("need_rag", True),
            preferred_sources=list(template.get("preferred_sources", ["slide", "textbook"])),
            fallback_to_llm=template.get("fallback_to_llm", self._fallback),
            source_raw_overfetch=dict(_DEFAULT_SOURCE_RAW_OVERFETCH),
            confidence=confidence,
            match_method=method,
        )

    def _build_passthrough_decision(self, *, method: str) -> RoutingDecision:
        return RoutingDecision(
            intent=QueryIntent.DEEP_UNDERSTANDING,
            need_rag=False,
            preferred_sources=[],
            retrieval_strategy="full",
            fallback_to_llm=self._fallback,
            source_raw_overfetch={},
            confidence=1.0,
            match_method=method,
        )

    def _apply_planner_context(
        self,
        decision: RoutingDecision,
        planner_task_intent: str,
    ) -> RoutingDecision:
        if planner_task_intent and planner_task_intent != "knowledge_query":
            if planner_task_intent not in _TASK_SOURCE_PROFILES:
                return self._build_passthrough_decision(method="planner_context")

        if planner_task_intent == "knowledge_query" and not decision.need_rag:
            decision = RoutingDecision(
                intent=QueryIntent.DEEP_UNDERSTANDING,
                need_rag=True,
                preferred_sources=["textbook", "slide"],
                retrieval_strategy=decision.retrieval_strategy,
                fallback_to_llm=decision.fallback_to_llm,
                source_filter=decision.source_filter,
                source_raw_overfetch=dict(_DEFAULT_SOURCE_RAW_OVERFETCH),
                confidence=decision.confidence,
                match_method=f"{decision.match_method}+planner_context",
            )
        return self._apply_task_profile(decision, planner_task_intent)

    def _apply_task_profile(
        self,
        decision: RoutingDecision,
        planner_task_intent: str,
    ) -> RoutingDecision:
        profile = _TASK_SOURCE_PROFILES.get(planner_task_intent or "")
        if not profile:
            return decision

        source_weights = {
            source: float(weight)
            for source, weight in dict(profile.get("source_weights", {})).items()
            if float(weight) > 0.0
        }
        preferred = list(decision.preferred_sources)
        if planner_task_intent != "knowledge_query":
            preferred = sorted(source_weights, key=lambda key: source_weights[key], reverse=True)
        elif not preferred:
            preferred = sorted(source_weights, key=lambda key: source_weights[key], reverse=True)

        return RoutingDecision(
            intent=decision.intent,
            need_rag=decision.need_rag,
            preferred_sources=preferred,
            retrieval_strategy=decision.retrieval_strategy,
            fallback_to_llm=decision.fallback_to_llm,
            source_filter=decision.source_filter,
            source_weights=source_weights,
            source_raw_overfetch=dict(_DEFAULT_SOURCE_RAW_OVERFETCH),
            normalization_profile="source_aware_units",
            evidence_profile=str(profile.get("evidence_profile", "default")),
            confidence=decision.confidence,
            match_method=decision.match_method,
        )

    # ── Introspection ──────────────────────────────────────────────────

    @property
    def embedding_ready(self) -> bool:
        return self._embedding_ready

    @property
    def prototype_count(self) -> int:
        return len(self._prototypes)
