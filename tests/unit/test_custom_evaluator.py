"""Unit tests for CustomEvaluator and EvaluatorFactory."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.libs.evaluator.base_evaluator import NoneEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory


class TestCustomEvaluator:
    """Tests for CustomEvaluator metrics computation."""

    def test_hit_rate_and_mrr_success(self) -> None:
        evaluator = CustomEvaluator(metrics=["hit_rate", "mrr"])
        retrieved = [
            {"id": "c1"},
            {"id": "c2"},
            {"id": "c3"},
        ]
        ground_truth = ["c2", "c9"]

        metrics = evaluator.evaluate("query", retrieved, ground_truth=ground_truth)

        assert metrics["hit_rate"] == 1.0
        assert metrics["mrr"] == 0.5

    def test_hit_rate_and_mrr_no_hit(self) -> None:
        evaluator = CustomEvaluator(metrics=["hit_rate", "mrr"])
        retrieved = [{"id": "a"}, {"id": "b"}]
        ground_truth = ["x", "y"]

        metrics = evaluator.evaluate("query", retrieved, ground_truth=ground_truth)

        assert metrics["hit_rate"] == 0.0
        assert metrics["mrr"] == 0.0

    def test_validate_query_and_retrieved(self) -> None:
        evaluator = CustomEvaluator(metrics=["hit_rate"])

        with pytest.raises(ValueError, match="Query cannot be empty"):
            evaluator.evaluate("  ", [{"id": "x"}], ground_truth=["x"])

        with pytest.raises(ValueError, match="retrieved_chunks cannot be empty"):
            evaluator.evaluate("query", [], ground_truth=["x"])

    def test_unsupported_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported custom metrics"):
            CustomEvaluator(metrics=["faithfulness"])  # not supported in custom evaluator

    def test_chunk_id_attribute_is_accepted(self) -> None:
        evaluator = CustomEvaluator(metrics=["hit_rate", "mrr"])
        retrieved = [SimpleNamespace(chunk_id="c1"), SimpleNamespace(chunk_id="c2")]
        metrics = evaluator.evaluate("query", retrieved, ground_truth=["c2"])

        assert metrics["hit_rate"] == 1.0
        assert metrics["mrr"] == 0.5

    def test_source_metrics_use_source_label(self) -> None:
        evaluator = CustomEvaluator(metrics=["source_hit_rate", "source_mrr"])
        retrieved = [
            {"id": "c1", "metadata": {"source_label": "第1章 计算机网络与因特网.pptx"}},
            {"id": "c2", "metadata": {"source_label": "第3章 运输层.pptx"}},
        ]

        metrics = evaluator.evaluate(
            "query",
            retrieved,
            ground_truth={"sources": ["第3章 运输层.pptx"]},
        )

        assert metrics["source_hit_rate"] == 1.0
        assert metrics["source_mrr"] == 0.5

    def test_source_metrics_fall_back_to_source_path_basename(self) -> None:
        evaluator = CustomEvaluator(metrics=["source_hit_rate", "source_mrr"])
        retrieved = [
            SimpleNamespace(
                chunk_id="c1",
                metadata={"source_path": "/tmp/docs/第2章 应用层.pptx"},
            ),
        ]

        metrics = evaluator.evaluate(
            "query",
            retrieved,
            ground_truth={"sources": ["第2章 应用层.pptx"]},
        )

        assert metrics["source_hit_rate"] == 1.0
        assert metrics["source_mrr"] == 1.0

    def test_source_metrics_skip_results_without_resolvable_source(self) -> None:
        evaluator = CustomEvaluator(metrics=["source_hit_rate", "source_mrr"])
        retrieved = [
            SimpleNamespace(chunk_id="c1", metadata={}),
            SimpleNamespace(
                chunk_id="c2",
                metadata={"source_label": "第2章 应用层.pptx"},
            ),
        ]

        metrics = evaluator.evaluate(
            "query",
            retrieved,
            ground_truth={"sources": ["第2章 应用层.pptx"]},
        )

        assert metrics["source_hit_rate"] == 1.0
        assert metrics["source_mrr"] == 1.0

    def test_source_metrics_dedupe_duplicate_sources(self) -> None:
        evaluator = CustomEvaluator(metrics=["source_hit_rate", "source_mrr"])
        retrieved = [
            SimpleNamespace(chunk_id="c1", metadata={"source_label": "第3章 运输层.pptx"}),
            SimpleNamespace(chunk_id="c2", metadata={"source_label": "第3章 运输层.pptx"}),
            SimpleNamespace(chunk_id="c3", metadata={"source_label": "第1章 计算机网络与因特网.pptx"}),
        ]

        metrics = evaluator.evaluate(
            "query",
            retrieved,
            ground_truth={"sources": ["第1章 计算机网络与因特网.pptx"]},
        )

        assert metrics["source_hit_rate"] == 1.0
        assert metrics["source_mrr"] == 0.5


class TestEvaluatorFactory:
    """Tests for EvaluatorFactory."""

    def setup_method(self) -> None:
        EvaluatorFactory._PROVIDERS = {"custom": CustomEvaluator}

    def test_create_custom_evaluator(self) -> None:
        settings = MagicMock()
        settings.evaluation.enabled = True
        settings.evaluation.provider = "custom"
        settings.evaluation.metrics = ["hit_rate", "mrr"]

        evaluator = EvaluatorFactory.create(settings)

        assert isinstance(evaluator, CustomEvaluator)

    def test_create_disabled_returns_none_evaluator(self) -> None:
        settings = MagicMock()
        settings.evaluation.enabled = False
        settings.evaluation.provider = "custom"
        settings.evaluation.metrics = ["hit_rate"]

        evaluator = EvaluatorFactory.create(settings)

        assert isinstance(evaluator, NoneEvaluator)

    def test_create_unknown_provider_raises(self) -> None:
        settings = MagicMock()
        settings.evaluation.enabled = True
        settings.evaluation.provider = "unknown"
        settings.evaluation.metrics = ["hit_rate"]

        with pytest.raises(ValueError, match="Unsupported Evaluator provider"):
            EvaluatorFactory.create(settings)

    def test_register_provider_success(self) -> None:
        class FakeEvaluator(CustomEvaluator):
            pass

        EvaluatorFactory.register_provider("fake", FakeEvaluator)

        assert "fake" in EvaluatorFactory._PROVIDERS
        assert EvaluatorFactory._PROVIDERS["fake"] is FakeEvaluator

    def test_list_providers_sorted(self) -> None:
        class AlphaEvaluator(CustomEvaluator):
            pass

        class BetaEvaluator(CustomEvaluator):
            pass

        EvaluatorFactory.register_provider("beta", BetaEvaluator)
        EvaluatorFactory.register_provider("alpha", AlphaEvaluator)

        assert EvaluatorFactory.list_providers() == ["alpha", "beta", "custom"]
