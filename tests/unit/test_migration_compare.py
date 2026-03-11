from __future__ import annotations

import json
from pathlib import Path

from src.observability.evaluation.eval_runner import EvalReport, QueryResult
from src.observability.evaluation.migration_compare import MigrationCompareRunner


def test_migration_compare_runner_detects_hard_failures(tmp_path, monkeypatch) -> None:
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {"query": "q1", "expected_chunk_ids": ["c1"]},
                    {"query": "q2", "expected_chunk_ids": ["c2"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    legacy_report = EvalReport(
        query_results=[
            QueryResult(query="q1", retrieved_chunk_ids=["c1", "x1"], metrics={"mrr": 0.9}),
            QueryResult(query="q2", retrieved_chunk_ids=["c2"], metrics={"mrr": 0.8}),
        ],
        aggregate_metrics={"mrr": 0.85, "recall": 1.0},
    )
    prod_report = EvalReport(
        query_results=[
            QueryResult(query="q1", retrieved_chunk_ids=["z1"], metrics={"mrr": 0.1}),
            QueryResult(query="q2", retrieved_chunk_ids=[], metrics={"mrr": 0.0}),
        ],
        aggregate_metrics={"mrr": 0.1, "recall": 0.4},
    )

    runner = MigrationCompareRunner(
        legacy_settings_path="config/settings.yaml",
        production_settings_path="config/settings.storage_stack.yaml",
    )

    calls = {"count": 0}

    def _fake_run_eval(settings_path, test_set_path, top_k, collection):
        calls["count"] += 1
        return legacy_report if calls["count"] == 1 else prod_report

    monkeypatch.setattr(runner, "_run_eval", _fake_run_eval)

    report = runner.run(golden_path, top_k=2, collection="demo")

    assert len(report.hard_failures) >= 2
    assert any("missing_expected" in item for item in report.hard_failures)
    assert report.pairwise_summary["case_count"] == 2
    assert report.per_case[0].failed is True


def test_migration_compare_run_eval_uses_full_settings(monkeypatch) -> None:
    runner = MigrationCompareRunner(
        legacy_settings_path="config/settings.yaml",
        production_settings_path="config/settings.storage_stack.yaml",
    )

    captured = {}

    class _FakeEvalRunner:
        def __init__(self, settings, hybrid_search, evaluator) -> None:
            captured["settings"] = settings
            captured["hybrid_search"] = hybrid_search
            captured["evaluator"] = evaluator

        def run(self, test_set_path, top_k, collection):
            return EvalReport(query_results=[], aggregate_metrics={})

    fake_settings = type(
        "FakeSettings",
        (),
        {
            "evaluation": type("EvalCfg", (), {"provider": "custom", "enabled": False})(),
            "vector_store": type("VS", (), {"collection_name": "demo"})(),
        },
    )()

    monkeypatch.setattr("src.core.settings.load_settings", lambda path: fake_settings)
    monkeypatch.setattr("src.observability.evaluation.migration_compare.EvaluatorFactory.create", lambda settings: ("evaluator", settings))
    monkeypatch.setattr("src.observability.evaluation.migration_compare.EvalRunner", _FakeEvalRunner)
    monkeypatch.setattr("src.server.app._build_hybrid_search", lambda collection, settings_path: ("hybrid", None, None))

    runner._run_eval("config/settings.yaml", "tests/fixtures/golden_test_set.json", top_k=5, collection="demo")

    assert captured["settings"] is not fake_settings
    assert captured["settings"].evaluation.enabled is True
    assert captured["settings"].evaluation.provider == "custom"
    assert captured["evaluator"][1].evaluation.provider == "custom"
