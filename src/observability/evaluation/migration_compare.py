"""Compare retrieval quality between legacy and production retrieval stacks."""

from __future__ import annotations

import json
import copy
from dataclasses import replace
from dataclasses import is_dataclass
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.eval_runner import EvalRunner, EvalReport


@dataclass
class CaseDiff:
    query: str
    legacy_ids: List[str] = field(default_factory=list)
    prod_ids: List[str] = field(default_factory=list)
    expected_ids: List[str] = field(default_factory=list)
    overlap_at_k: float = 0.0
    missing_expected_ids: List[str] = field(default_factory=list)
    new_only_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    failed: bool = False


@dataclass
class MigrationCompareReport:
    legacy_metrics: Dict[str, float]
    prod_metrics: Dict[str, float]
    pairwise_summary: Dict[str, Any]
    per_case: List[CaseDiff]
    hard_failures: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "legacy_metrics": self.legacy_metrics,
            "prod_metrics": self.prod_metrics,
            "pairwise_summary": self.pairwise_summary,
            "per_case": [
                {
                    "query": case.query,
                    "legacy_ids": case.legacy_ids,
                    "prod_ids": case.prod_ids,
                    "expected_ids": case.expected_ids,
                    "overlap_at_k": round(case.overlap_at_k, 4),
                    "missing_expected_ids": case.missing_expected_ids,
                    "new_only_ids": case.new_only_ids,
                    "warnings": case.warnings,
                    "failed": case.failed,
                }
                for case in self.per_case
            ],
            "hard_failures": self.hard_failures,
            "warnings": self.warnings,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class MigrationCompareRunner:
    """Run retrieval evaluation on both stacks and compare results case-by-case."""

    def __init__(
        self,
        *,
        legacy_settings_path: str | Path,
        production_settings_path: str | Path,
    ) -> None:
        self.legacy_settings_path = str(legacy_settings_path)
        self.production_settings_path = str(production_settings_path)

    def run(
        self,
        test_set_path: str | Path,
        *,
        top_k: int = 10,
        collection: Optional[str] = None,
    ) -> MigrationCompareReport:
        legacy_report = self._run_eval(self.legacy_settings_path, test_set_path, top_k=top_k, collection=collection)
        prod_report = self._run_eval(self.production_settings_path, test_set_path, top_k=top_k, collection=collection)
        pairwise = self._build_pairwise_report(test_set_path, legacy_report, prod_report, top_k=top_k)
        return pairwise

    def _run_eval(
        self,
        settings_path: str | Path,
        test_set_path: str | Path,
        *,
        top_k: int,
        collection: Optional[str],
    ) -> EvalReport:
        from src.core.settings import load_settings
        from src.server.app import _build_hybrid_search

        settings = load_settings(settings_path)
        compare_settings = self._with_compare_evaluation(settings)
        evaluator = EvaluatorFactory.create(compare_settings)
        hybrid_search, _, _ = _build_hybrid_search(collection or settings.vector_store.collection_name, settings_path=str(settings_path))
        runner = EvalRunner(settings=compare_settings, hybrid_search=hybrid_search, evaluator=evaluator)
        return runner.run(test_set_path=test_set_path, top_k=top_k, collection=collection)

    @staticmethod
    def _with_compare_evaluation(settings: Any) -> Any:
        if is_dataclass(settings) and is_dataclass(getattr(settings, "evaluation", None)):
            return replace(
                settings,
                evaluation=replace(settings.evaluation, enabled=True, provider="custom", metrics=["hit_rate", "mrr"]),
            )

        compare_settings = copy.deepcopy(settings)
        evaluation = copy.deepcopy(getattr(settings, "evaluation", None))
        if evaluation is None:
            raise ValueError("settings.evaluation is required for migration compare")
        setattr(evaluation, "enabled", True)
        setattr(evaluation, "provider", "custom")
        setattr(evaluation, "metrics", ["hit_rate", "mrr"])
        setattr(compare_settings, "evaluation", evaluation)
        return compare_settings

    def _build_pairwise_report(
        self,
        test_set_path: str | Path,
        legacy_report: EvalReport,
        prod_report: EvalReport,
        *,
        top_k: int,
    ) -> MigrationCompareReport:
        from src.observability.evaluation.eval_runner import load_test_set

        cases = load_test_set(test_set_path)
        legacy_by_query = {result.query: result for result in legacy_report.query_results}
        prod_by_query = {result.query: result for result in prod_report.query_results}

        hard_failures: List[str] = []
        warnings: List[str] = []
        per_case: List[CaseDiff] = []
        overlap_values: List[float] = []

        for case in cases:
            legacy_ids = legacy_by_query.get(case.query).retrieved_chunk_ids if case.query in legacy_by_query else []
            prod_ids = prod_by_query.get(case.query).retrieved_chunk_ids if case.query in prod_by_query else []
            legacy_set = set(legacy_ids[:top_k])
            prod_set = set(prod_ids[:top_k])
            union = legacy_set | prod_set
            overlap_at_k = (len(legacy_set & prod_set) / len(union)) if union else 1.0
            overlap_values.append(overlap_at_k)
            missing_expected_ids = [cid for cid in case.expected_chunk_ids if cid not in prod_ids[:top_k]]
            new_only_ids = [cid for cid in prod_ids[:top_k] if cid not in legacy_ids[:top_k]]
            case_warnings: List[str] = []
            failed = False

            if legacy_ids and not prod_ids:
                msg = f"hard_fail:no_prod_results:{case.query}"
                hard_failures.append(msg)
                failed = True
            if overlap_at_k < 0.4:
                case_warnings.append("low_overlap")
                warnings.append(f"warning:low_overlap:{case.query}")
            if missing_expected_ids:
                failed = True
                hard_failures.append(f"hard_fail:missing_expected:{case.query}")

            per_case.append(
                CaseDiff(
                    query=case.query,
                    legacy_ids=legacy_ids[:top_k],
                    prod_ids=prod_ids[:top_k],
                    expected_ids=case.expected_chunk_ids,
                    overlap_at_k=overlap_at_k,
                    missing_expected_ids=missing_expected_ids,
                    new_only_ids=new_only_ids,
                    warnings=case_warnings,
                    failed=failed,
                )
            )

        for metric_name in ("recall", "mrr", "recall_at_k", "mrr_at_k"):
            legacy_value = legacy_report.aggregate_metrics.get(metric_name)
            prod_value = prod_report.aggregate_metrics.get(metric_name)
            if legacy_value is None or prod_value is None:
                continue
            if float(prod_value) < float(legacy_value) - 0.02:
                hard_failures.append(f"hard_fail:{metric_name}_regressed")

        pairwise_summary = {
            "case_count": len(per_case),
            "avg_overlap_at_k": round(sum(overlap_values) / len(overlap_values), 4) if overlap_values else 0.0,
            "hard_failure_count": len(hard_failures),
            "warning_count": len(warnings),
        }

        return MigrationCompareReport(
            legacy_metrics=legacy_report.aggregate_metrics,
            prod_metrics=prod_report.aggregate_metrics,
            pairwise_summary=pairwise_summary,
            per_case=per_case,
            hard_failures=hard_failures,
            warnings=warnings,
        )
