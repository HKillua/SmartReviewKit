"""Custom evaluator implementation for lightweight retrieval metrics.

This evaluator computes deterministic retrieval metrics for either chunk IDs
or source labels. The source-level path is intentionally supported because
chunk IDs may drift after re-ingestion while source filenames stay stable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Custom evaluator for lightweight metrics.

    The evaluator expects retrieved chunks to contain an identifier field.
    Supported ID fields: ``id``, ``chunk_id``, ``document_id``, ``doc_id``.
    Supported source fields: ``source_label``, ``original_filename``,
    ``source``, ``source_path``.
    """

    SUPPORTED_METRICS = {"hit_rate", "mrr", "source_hit_rate", "source_mrr"}
    _ID_FIELDS = ("id", "chunk_id", "document_id", "doc_id")
    _SOURCE_FIELDS = ("source_label", "original_filename", "source", "source_path")

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalized = [str(metric).strip().lower() for metric in (metrics or [])]
        if not normalized:
            normalized = ["hit_rate", "mrr"]

        unsupported = [metric for metric in normalized if metric not in self.SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                "Unsupported custom metrics: "
                f"{', '.join(unsupported)}. Supported: {', '.join(sorted(self.SUPPORTED_METRICS))}"
            )

        self.metrics = normalized

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Compute requested metrics for the given retrieval results.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks or records.
            generated_answer: Optional generated answer (unused).
            ground_truth: Ground truth ids or structure.
            trace: Optional TraceContext (unused).
            **kwargs: Additional parameters (unused).

        Returns:
            Dictionary of metric name to float value.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        retrieved_ids = self._extract_ids(retrieved_chunks, label="retrieved_chunks")
        retrieved_sources: List[str] = []
        if "source_hit_rate" in self.metrics or "source_mrr" in self.metrics:
            retrieved_sources = self._extract_sources(retrieved_chunks, label="retrieved_chunks")
        ground_truth_ids = self._extract_ground_truth_ids(ground_truth)
        ground_truth_sources = self._extract_ground_truth_sources(ground_truth)

        results: Dict[str, float] = {}

        if "hit_rate" in self.metrics:
            results["hit_rate"] = self._compute_hit_rate(retrieved_ids, ground_truth_ids)
        if "mrr" in self.metrics:
            results["mrr"] = self._compute_mrr(retrieved_ids, ground_truth_ids)
        if "source_hit_rate" in self.metrics:
            results["source_hit_rate"] = self._compute_hit_rate(retrieved_sources, ground_truth_sources)
        if "source_mrr" in self.metrics:
            results["source_mrr"] = self._compute_mrr(retrieved_sources, ground_truth_sources)

        return results

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        metrics = getattr(getattr(settings, "evaluation", None), "metrics", None)
        if metrics is None:
            return []
        return [str(metric) for metric in metrics]

    def _extract_ground_truth_ids(self, ground_truth: Optional[Any]) -> List[str]:
        """Extract ground truth ids from various input shapes."""
        if ground_truth is None:
            return []
        if isinstance(ground_truth, str):
            return [ground_truth]
        if isinstance(ground_truth, dict):
            if "ids" in ground_truth and isinstance(ground_truth["ids"], list):
                return self._extract_ids(ground_truth["ids"], label="ground_truth.ids")
            if "ids" not in ground_truth:
                return []
            return self._extract_ids([ground_truth], label="ground_truth")
        if isinstance(ground_truth, list):
            return self._extract_ids(ground_truth, label="ground_truth")

        raise ValueError(
            f"Unsupported ground_truth type: {type(ground_truth).__name__}. "
            "Expected str, dict, list, or None."
        )

    def _extract_ground_truth_sources(self, ground_truth: Optional[Any]) -> List[str]:
        """Extract ground-truth source labels from various input shapes."""
        if ground_truth is None:
            return []
        if isinstance(ground_truth, dict):
            raw_sources = ground_truth.get("sources")
            if isinstance(raw_sources, list):
                return [self._normalize_source_label(item) for item in raw_sources if str(item).strip()]
            if isinstance(raw_sources, str) and raw_sources.strip():
                return [self._normalize_source_label(raw_sources)]
            return []
        return []

    def _extract_ids(self, items: Iterable[Any], label: str) -> List[str]:
        """Extract ids from a list of items."""
        ids: List[str] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                ids.append(item)
                continue
            if isinstance(item, dict):
                for field in self._ID_FIELDS:
                    if field in item:
                        ids.append(str(item[field]))
                        break
                else:
                    raise ValueError(
                        f"Missing id field in {label}[{index}]. "
                        f"Expected one of {', '.join(self._ID_FIELDS)}"
                    )
                continue
            if hasattr(item, "id"):
                ids.append(str(getattr(item, "id")))
                continue
            if hasattr(item, "chunk_id"):
                ids.append(str(getattr(item, "chunk_id")))
                continue

            raise ValueError(
                f"Unable to extract id from {label}[{index}] of type "
                f"{type(item).__name__}"
            )

        return ids

    def _extract_sources(self, items: Iterable[Any], label: str) -> List[str]:
        """Extract normalized source labels from a list of items."""
        sources: List[str] = []
        for index, item in enumerate(items):
            source_value: Optional[str] = None
            if isinstance(item, dict):
                source_value = self._source_from_mapping(item)
            elif hasattr(item, "metadata"):
                metadata = getattr(item, "metadata", None)
                if isinstance(metadata, dict):
                    source_value = self._source_from_mapping(metadata)
            if source_value is None:
                continue
            normalized = self._normalize_source_label(source_value)
            if normalized:
                sources.append(normalized)
        return self._dedupe_preserve_order(sources)

    def _source_from_mapping(self, payload: Dict[str, Any]) -> Optional[str]:
        for field in self._SOURCE_FIELDS:
            value = payload.get(field)
            if value:
                return str(value)
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for field in self._SOURCE_FIELDS:
                value = metadata.get(field)
                if value:
                    return str(value)
        return None

    def _normalize_source_label(self, source: Any) -> str:
        text = str(source or "").strip()
        if not text:
            return ""
        if "/" in text or "\\" in text:
            text = Path(text).name
        return text.casefold()

    def _dedupe_preserve_order(self, values: Sequence[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _compute_hit_rate(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """Compute hit rate (binary)."""
        if not ground_truth_ids:
            return 0.0
        return 1.0 if any(item in ground_truth_ids for item in retrieved_ids) else 0.0

    def _compute_mrr(self, retrieved_ids: Sequence[str], ground_truth_ids: Sequence[str]) -> float:
        """Compute Mean Reciprocal Rank (MRR)."""
        if not ground_truth_ids:
            return 0.0
        for rank, item in enumerate(retrieved_ids, start=1):
            if item in ground_truth_ids:
                return 1.0 / rank
        return 0.0
