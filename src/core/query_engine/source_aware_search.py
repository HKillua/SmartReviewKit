"""Source-aware retrieval and answer-unit normalization.

This layer sits above ``HybridSearch``. It keeps each source on its native
chunking strategy during recall, then normalizes results into comparable
answer units before doing global rerank / MMR / dedup.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.core.types import AnswerUnit, RetrievalResult, SourceAwareSearchResult

logger = logging.getLogger(__name__)


_TASK_SOURCE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "knowledge_query": {"textbook": 0.55, "slide": 0.30, "question_bank": 0.15},
    "review_summary": {"slide": 0.45, "textbook": 0.40, "question_bank": 0.15},
    "quiz_generator": {"question_bank": 0.60, "textbook": 0.25, "slide": 0.15},
    "quiz_evaluator": {"question_bank": 0.50, "textbook": 0.30, "slide": 0.20},
}

_DEFAULT_OVERFETCH: Dict[str, int] = {
    "question_bank": 2,
    "slide": 2,
    "textbook": 4,
}

_EVIDENCE_SOURCE_ORDER: Dict[str, List[str]] = {
    "explanatory": ["textbook", "slide", "question_bank"],
    "summary": ["slide", "textbook", "question_bank"],
    "practice_generation": ["question_bank", "textbook", "slide"],
    "answer_grading": ["question_bank", "textbook", "slide"],
    "default": ["textbook", "slide", "question_bank"],
}


class SourceAwareSearch:
    """Shared retrieval layer for mixed-granularity multi-source search."""

    def __init__(self, hybrid_search: Any, query_router: Any = None) -> None:
        self._hybrid_search = hybrid_search
        self._query_router = query_router

    def search(
        self,
        *,
        query: str,
        task_intent: str,
        top_k: int,
        trace: Optional[Any] = None,
        filters: Optional[Dict[str, Any]] = None,
        route_decision: Any = None,
        query_vector: Optional[List[float]] = None,
        allowed_sources: Optional[Iterable[str]] = None,
    ) -> SourceAwareSearchResult:
        routing = route_decision or self._build_route(query=query, task_intent=task_intent)
        source_weights, preferred_sources = self._select_sources(routing, allowed_sources)
        source_budgets = self._allocate_budgets(
            source_weights=source_weights,
            preferred_sources=preferred_sources,
            top_k=top_k,
        )
        overfetch = self._select_overfetch(routing, preferred_sources)

        if trace is not None:
            trace.record_stage(
                "source_allocation",
                {
                    "task_intent": task_intent,
                    "preferred_sources": preferred_sources,
                    "source_weights": source_weights,
                    "source_unit_budgets": source_budgets,
                    "source_raw_overfetch": overfetch,
                },
            )

        all_units: list[AnswerUnit] = []
        raw_candidate_counts: Dict[str, int] = {}
        normalized_unit_counts: Dict[str, int] = {}
        textbook_stats = {
            "child_hit_count": 0,
            "parent_promotions": 0,
            "collapsed_parent_count": 0,
        }

        for source_type in preferred_sources:
            unit_budget = source_budgets.get(source_type, 0)
            if unit_budget <= 0:
                continue
            raw_k = max(unit_budget, overfetch.get(source_type, 1) * unit_budget)
            source_filters = self._merge_filters(filters, {"source_type": source_type})
            raw_results = self._run_hybrid_search(
                query=query,
                top_k=raw_k,
                filters=source_filters,
                query_vector=query_vector,
                trace=trace,
            )
            raw_candidate_counts[source_type] = len(raw_results)
            normalized_units, stats = self._normalize_source_results(source_type, raw_results)
            normalized_units.sort(key=lambda unit: unit.score, reverse=True)
            normalized_units = normalized_units[:unit_budget]
            normalized_unit_counts[source_type] = len(normalized_units)
            if source_type == "textbook":
                for key in textbook_stats:
                    textbook_stats[key] += int(stats.get(key, 0))
            all_units.extend(normalized_units)

        fallback_global_used = False
        if len(all_units) < top_k and allowed_sources is None:
            fallback_results = self._run_hybrid_search(
                query=query,
                top_k=max(top_k, max(source_budgets.values(), default=top_k)),
                filters=filters,
                query_vector=query_vector,
                trace=trace,
            )
            if fallback_results:
                fallback_global_used = True
                raw_candidate_counts["fallback_all"] = len(fallback_results)
                fallback_units, fallback_stats = self._normalize_mixed_results(fallback_results)
                existing_unit_ids = {unit.unit_id for unit in all_units}
                fallback_units = [
                    unit for unit in fallback_units
                    if unit.unit_id not in existing_unit_ids
                ]
                normalized_unit_counts["fallback_all"] = len(fallback_units)
                for key in textbook_stats:
                    textbook_stats[key] += int(fallback_stats.get(key, 0))
                all_units.extend(fallback_units[: max(top_k - len(all_units), 0)])

        all_units = self._coalesce_duplicate_units(all_units)
        raw_unit_results = [unit.to_retrieval_result() for unit in all_units]
        raw_unit_results.sort(key=lambda result: result.score, reverse=True)

        if trace is not None:
            trace.record_stage(
                "answer_unit_normalization",
                {
                    "normalization_profile": getattr(
                        routing, "normalization_profile", "source_aware_units"
                    ),
                    "source_raw_candidate_counts": raw_candidate_counts,
                    "source_normalized_unit_counts": normalized_unit_counts,
                    **textbook_stats,
                },
            )

        reranked = self._apply_unit_reranker(query=query, results=raw_unit_results, trace=trace)
        reranked = self._apply_unit_mmr(query=query, results=reranked, top_k=top_k, trace=trace)
        reranked = self._apply_unit_dedup(results=reranked, trace=trace)
        final_results = reranked[:top_k]

        unit_lookup = {unit.unit_id: unit for unit in all_units}
        final_units: list[AnswerUnit] = []
        for result in final_results:
            final_units.append(unit_lookup.get(result.chunk_id) or self._unit_from_result(result))

        evidence_profile = getattr(routing, "evidence_profile", "default")
        final_units = self._apply_presentation_order(
            final_units,
            evidence_profile=str(evidence_profile or "default"),
        )
        final_results = [unit.to_retrieval_result() for unit in final_units]

        unit_kind_distribution = Counter(unit.unit_kind for unit in final_units)
        routing_metadata = {
            "preferred_sources": preferred_sources,
            "source_weights": source_weights,
            "source_unit_budgets": source_budgets,
            "source_raw_overfetch": overfetch,
            "source_raw_candidate_counts": raw_candidate_counts,
            "source_normalized_unit_counts": normalized_unit_counts,
            "normalization_profile": getattr(routing, "normalization_profile", "source_aware_units"),
            "evidence_profile": evidence_profile,
            "evidence_source_order": list(
                _EVIDENCE_SOURCE_ORDER.get(str(evidence_profile or "default"), _EVIDENCE_SOURCE_ORDER["default"])
            ),
            "unit_kind_distribution": dict(unit_kind_distribution),
            "fallback_global_used": fallback_global_used,
            **textbook_stats,
        }
        return SourceAwareSearchResult(
            answer_units=final_units,
            results=final_results,
            routing_metadata=routing_metadata,
        )

    def _build_route(self, *, query: str, task_intent: str) -> Any:
        if self._query_router is not None:
            try:
                return self._query_router.route(query, planner_task_intent=task_intent)
            except Exception:
                logger.warning("SourceAwareSearch route fallback triggered", exc_info=True)

        weights = dict(_TASK_SOURCE_WEIGHTS.get(task_intent, _TASK_SOURCE_WEIGHTS["knowledge_query"]))
        evidence_profile = {
            "knowledge_query": "explanatory",
            "review_summary": "summary",
            "quiz_generator": "practice_generation",
            "quiz_evaluator": "answer_grading",
        }.get(task_intent, "default")
        return SimpleNamespace(
            preferred_sources=sorted(weights, key=lambda key: weights[key], reverse=True),
            source_weights=weights,
            source_raw_overfetch=dict(_DEFAULT_OVERFETCH),
            normalization_profile="source_aware_units",
            evidence_profile=evidence_profile,
        )

    def _select_sources(
        self,
        routing: Any,
        allowed_sources: Optional[Iterable[str]],
    ) -> tuple[Dict[str, float], List[str]]:
        route_weights = {
            str(source): float(weight)
            for source, weight in dict(getattr(routing, "source_weights", {}) or {}).items()
            if float(weight) > 0.0
        }
        if not route_weights:
            preferred = list(getattr(routing, "preferred_sources", []) or [])
            if not preferred:
                preferred = ["textbook", "slide"]
            descending = list(reversed(range(1, len(preferred) + 1)))
            route_weights = {
                source: float(weight)
                for source, weight in zip(preferred, descending, strict=False)
            }

        selected = set(route_weights)
        if allowed_sources is not None:
            selected &= {str(source) for source in allowed_sources}

        if not selected:
            return {}, []

        filtered_weights = {
            source: weight for source, weight in route_weights.items() if source in selected
        }
        total = sum(filtered_weights.values()) or 1.0
        filtered_weights = {
            source: round(weight / total, 6) for source, weight in filtered_weights.items()
        }

        preferred_sources = [
            source
            for source in list(getattr(routing, "preferred_sources", []) or [])
            if source in filtered_weights
        ]
        for source in sorted(filtered_weights, key=lambda key: filtered_weights[key], reverse=True):
            if source not in preferred_sources:
                preferred_sources.append(source)
        return filtered_weights, preferred_sources

    def _allocate_budgets(
        self,
        *,
        source_weights: Dict[str, float],
        preferred_sources: List[str],
        top_k: int,
    ) -> Dict[str, int]:
        if top_k <= 0 or not source_weights:
            return {}

        quotas = {
            source: float(weight) * top_k for source, weight in source_weights.items()
        }
        budgets = {
            source: int(quotas[source]) for source in source_weights
        }

        if top_k >= len(source_weights):
            for source in source_weights:
                budgets[source] = max(budgets[source], 1)

        reserved = sum(budgets.values())
        remainders = [
            (quotas[source] - int(quotas[source]), source)
            for source in preferred_sources
            if source in quotas
        ]
        while reserved < top_k and remainders:
            remainders.sort(key=lambda item: (-item[0], preferred_sources.index(item[1])))
            _, source = remainders[0]
            budgets[source] = budgets.get(source, 0) + 1
            reserved += 1
            remainders[0] = (0.0, source)

        while reserved > top_k:
            removable = [
                source
                for source in reversed(preferred_sources)
                if budgets.get(source, 0) > 1 or (top_k < len(source_weights) and budgets.get(source, 0) > 0)
            ]
            if not removable:
                break
            source = removable[0]
            budgets[source] -= 1
            reserved -= 1

        return {source: count for source, count in budgets.items() if count > 0}

    def _select_overfetch(self, routing: Any, preferred_sources: List[str]) -> Dict[str, int]:
        configured = dict(getattr(routing, "source_raw_overfetch", {}) or {})
        return {
            source: int(configured.get(source, _DEFAULT_OVERFETCH.get(source, 1)))
            for source in preferred_sources
        }

    def _run_hybrid_search(
        self,
        *,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        query_vector: Optional[List[float]],
        trace: Optional[Any],
    ) -> list[RetrievalResult]:
        try:
            results = self._hybrid_search.search(
                query=query,
                top_k=top_k,
                filters=filters,
                query_vector=query_vector,
                trace=trace,
            )
        except TypeError:
            try:
                results = self._hybrid_search.search(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    trace=trace,
                )
            except TypeError:
                results = self._hybrid_search.search(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                )
        return results if isinstance(results, list) else []

    def _normalize_source_results(
        self,
        source_type: str,
        results: List[RetrievalResult],
    ) -> tuple[list[AnswerUnit], Dict[str, int]]:
        if source_type == "textbook":
            return self._normalize_textbook_results(results)
        if source_type == "question_bank":
            return self._normalize_flat_results(results, source_type="question_bank", unit_kind="question_item"), {}
        return self._normalize_flat_results(results, source_type="slide", unit_kind="slide_chunk"), {}

    def _normalize_flat_results(
        self,
        results: List[RetrievalResult],
        *,
        source_type: str,
        unit_kind: str,
    ) -> list[AnswerUnit]:
        units: list[AnswerUnit] = []
        for result in results:
            metadata = dict(result.metadata or {})
            metadata.setdefault("source_type", source_type)
            units.append(
                AnswerUnit(
                    unit_id=result.chunk_id,
                    source_type=source_type,
                    unit_kind=unit_kind,
                    retrieval_text=result.text,
                    display_text=result.text,
                    backing_chunk_ids=[result.chunk_id],
                    metadata=metadata,
                    raw_scores=[result.score],
                    support_count=1,
                )
            )
        return units

    def _normalize_textbook_results(
        self,
        results: List[RetrievalResult],
    ) -> tuple[list[AnswerUnit], Dict[str, int]]:
        grouped: Dict[str, List[RetrievalResult]] = {}
        child_hit_count = 0
        for result in results:
            metadata = dict(result.metadata or {})
            is_parent = bool(metadata.get("is_parent", False))
            parent_id = str(metadata.get("parent_chunk_id") or result.chunk_id)
            if is_parent:
                parent_id = result.chunk_id
            else:
                child_hit_count += 1
            grouped.setdefault(parent_id, []).append(result)

        parent_records = self._fetch_parent_records(grouped.keys())
        units: list[AnswerUnit] = []
        parent_promotions = 0
        for parent_id, members in grouped.items():
            parent_record = parent_records.get(parent_id)
            parent_hit = next((item for item in members if bool(item.metadata.get("is_parent", False))), None)
            representative = parent_hit or members[0]
            if parent_record is not None:
                parent_text = str(parent_record.get("text", "") or representative.text)
                parent_metadata = dict(parent_record.get("metadata", {}) or {})
                parent_metadata.setdefault("source_type", "textbook")
                if parent_hit is None:
                    parent_promotions += 1
            else:
                parent_text = representative.text
                parent_metadata = dict(representative.metadata or {})
                parent_metadata.setdefault("source_type", "textbook")
                parent_metadata["parent_resolution_failed"] = True
            parent_metadata["collapsed_from_chunk_ids"] = [member.chunk_id for member in members]
            units.append(
                AnswerUnit(
                    unit_id=parent_id,
                    source_type="textbook",
                    unit_kind="textbook_parent",
                    retrieval_text=parent_text,
                    display_text=parent_text,
                    backing_chunk_ids=[member.chunk_id for member in members],
                    metadata=parent_metadata,
                    raw_scores=[member.score for member in members],
                    support_count=len(members),
                )
            )

        stats = {
            "child_hit_count": child_hit_count,
            "parent_promotions": parent_promotions,
            "collapsed_parent_count": len(units),
        }
        return units, stats

    def _normalize_mixed_results(
        self,
        results: List[RetrievalResult],
    ) -> tuple[list[AnswerUnit], Dict[str, int]]:
        buckets: Dict[str, List[RetrievalResult]] = {
            "question_bank": [],
            "slide": [],
            "textbook": [],
        }
        for result in results:
            inferred = self._infer_source_type(result.metadata)
            buckets.setdefault(inferred, []).append(result)

        all_units: list[AnswerUnit] = []
        textbook_stats = {"child_hit_count": 0, "parent_promotions": 0, "collapsed_parent_count": 0}
        for source_type, bucket in buckets.items():
            if not bucket:
                continue
            units, stats = self._normalize_source_results(source_type, bucket)
            all_units.extend(units)
            if source_type == "textbook":
                for key in textbook_stats:
                    textbook_stats[key] += int(stats.get(key, 0))
        return all_units, textbook_stats

    def _coalesce_duplicate_units(self, units: List[AnswerUnit]) -> list[AnswerUnit]:
        if len(units) <= 1:
            return units
        merged: Dict[tuple[tuple[str, ...], str], AnswerUnit] = {}
        for unit in units:
            key = (
                tuple(sorted(str(chunk_id) for chunk_id in unit.backing_chunk_ids)),
                unit.display_text.strip(),
            )
            existing = merged.get(key)
            if existing is None:
                merged[key] = unit
                continue
            merged[key] = AnswerUnit(
                unit_id=existing.unit_id,
                source_type=existing.source_type,
                unit_kind=existing.unit_kind,
                retrieval_text=existing.retrieval_text,
                display_text=existing.display_text,
                backing_chunk_ids=sorted(set(existing.backing_chunk_ids + unit.backing_chunk_ids)),
                metadata=dict(existing.metadata),
                raw_scores=list(existing.raw_scores) + list(unit.raw_scores),
                support_count=max(existing.support_count, unit.support_count),
            )
        return list(merged.values())

    def _fetch_parent_records(self, parent_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        vector_store = self._resolve_vector_store()
        if vector_store is None:
            return {}
        ids = [parent_id for parent_id in parent_ids if parent_id]
        if not ids:
            return {}
        try:
            records = vector_store.get_by_ids(ids)
        except Exception:
            logger.debug("Parent record fetch failed", exc_info=True)
            return {}
        return {
            str(record.get("id", "")): dict(record)
            for record in records or []
            if str(record.get("id", ""))
        }

    def _resolve_vector_store(self) -> Any | None:
        dense = getattr(self._hybrid_search, "dense_retriever", None)
        if dense is not None and getattr(dense, "vector_store", None) is not None:
            return dense.vector_store
        sparse = getattr(self._hybrid_search, "sparse_retriever", None)
        if sparse is not None and getattr(sparse, "vector_store", None) is not None:
            return sparse.vector_store
        return None

    def _apply_unit_reranker(
        self,
        *,
        query: str,
        results: List[RetrievalResult],
        trace: Optional[Any],
    ) -> List[RetrievalResult]:
        reranker = getattr(self._hybrid_search, "reranker", None)
        config = getattr(self._hybrid_search, "config", None)
        rerank_enabled = self._flag_enabled(config, "rerank_enabled")
        rerank_top_k = int(getattr(config, "rerank_top_k", len(results) or 1))
        if not rerank_enabled or reranker is None or not hasattr(reranker, "rerank") or not results:
            return results
        try:
            start = time.monotonic()
            candidates = [
                {
                    "chunk_id": result.chunk_id,
                    "text": result.text,
                    "metadata": dict(result.metadata or {}),
                }
                for result in results
            ]
            reranked = reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=max(rerank_top_k, min(len(candidates), len(results))),
            )
            elapsed_ms = (time.monotonic() - start) * 1000.0
            reranked_results = [
                RetrievalResult(
                    chunk_id=candidate.get("chunk_id") or candidate.get("id") or "",
                    score=float(candidate.get("rerank_score", candidate.get("score", 0.0))),
                    text=str(candidate.get("text", "")),
                    metadata=dict(candidate.get("metadata", {}) or {}),
                )
                for candidate in reranked
                if candidate.get("chunk_id") or candidate.get("id")
            ]
            if trace is not None:
                trace.record_stage(
                    "answer_unit_rerank",
                    {
                        "input_count": len(results),
                        "output_count": len(reranked_results),
                        "method": type(reranker).__name__,
                    },
                    elapsed_ms=elapsed_ms,
                )
            return reranked_results or results
        except Exception as exc:
            logger.warning("Answer-unit rerank failed, using pre-rerank order: %s", exc)
            if trace is not None:
                trace.record_stage(
                    "answer_unit_rerank",
                    {
                        "input_count": len(results),
                        "output_count": len(results),
                        "method": type(reranker).__name__,
                        "used_fallback": True,
                        "error": str(exc)[:300],
                    },
                )
            return results

    def _apply_unit_mmr(
        self,
        *,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
        trace: Optional[Any],
    ) -> List[RetrievalResult]:
        config = getattr(self._hybrid_search, "config", None)
        if not self._flag_enabled(config, "mmr_enabled") or not results:
            return results
        mmr_fn = getattr(self._hybrid_search, "_apply_mmr", None)
        if mmr_fn is None or not callable(mmr_fn):
            return results
        try:
            return mmr_fn(query, results, top_k, trace)
        except Exception:
            logger.debug("Answer-unit MMR failed", exc_info=True)
            return results

    def _apply_unit_dedup(
        self,
        *,
        results: List[RetrievalResult],
        trace: Optional[Any],
    ) -> List[RetrievalResult]:
        config = getattr(self._hybrid_search, "config", None)
        if not self._flag_enabled(config, "post_dedup_enabled") or len(results) <= 1:
            return results
        dedup_fn = getattr(self._hybrid_search, "_apply_post_dedup", None)
        if dedup_fn is None or not callable(dedup_fn):
            return results
        try:
            start = time.monotonic()
            deduped = dedup_fn(results)
            if trace is not None:
                trace.record_stage(
                    "answer_unit_dedup",
                    {
                        "input_count": len(results),
                        "output_count": len(deduped),
                    },
                    elapsed_ms=(time.monotonic() - start) * 1000.0,
                )
            return deduped
        except Exception:
            logger.debug("Answer-unit dedup failed", exc_info=True)
            return results

    @staticmethod
    def _apply_presentation_order(
        units: List[AnswerUnit],
        *,
        evidence_profile: str,
    ) -> List[AnswerUnit]:
        if len(units) <= 1:
            return units
        source_order = _EVIDENCE_SOURCE_ORDER.get(
            evidence_profile,
            _EVIDENCE_SOURCE_ORDER["default"],
        )
        order_index = {source: index for index, source in enumerate(source_order)}
        return sorted(
            units,
            key=lambda unit: (
                order_index.get(unit.source_type, len(order_index)),
                -unit.score,
                unit.unit_id,
            ),
        )

    @staticmethod
    def _merge_filters(
        base_filters: Optional[Dict[str, Any]],
        extra_filters: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not base_filters and not extra_filters:
            return None
        merged = dict(base_filters or {})
        merged.update(extra_filters or {})
        return merged

    @staticmethod
    def _unit_from_result(result: RetrievalResult) -> AnswerUnit:
        metadata = dict(result.metadata or {})
        return AnswerUnit(
            unit_id=result.chunk_id,
            source_type=str(metadata.get("source_type", "unknown") or "unknown"),
            unit_kind=str(metadata.get("unit_kind", "normalized_chunk") or "normalized_chunk"),
            retrieval_text=result.text,
            display_text=result.text,
            backing_chunk_ids=list(metadata.get("backing_chunk_ids", [result.chunk_id])),
            metadata=metadata,
            raw_scores=list(metadata.get("raw_scores", [result.score])),
            support_count=int(metadata.get("support_count", 1) or 1),
        )

    @staticmethod
    def _flag_enabled(config: Any, field_name: str) -> bool:
        if config is None:
            return False
        value = getattr(config, field_name, False)
        return value is True

    @staticmethod
    def _infer_source_type(metadata: Dict[str, Any] | None) -> str:
        metadata = metadata or {}
        source_type = str(metadata.get("source_type", "") or "").strip().lower()
        if source_type in {"question_bank", "slide", "textbook"}:
            return source_type

        haystack = " ".join(
            str(metadata.get(key, "") or "").lower()
            for key in ("source_path", "source_label", "original_filename", "title")
        )
        if any(token in haystack for token in ("习题", "question", "quiz", "exam", "题库")):
            return "question_bank"
        if any(token in haystack for token in (".ppt", ".pptx", "slide", "lecture", "课件")):
            return "slide"
        return "textbook"
