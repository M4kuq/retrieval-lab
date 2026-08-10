from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from retrieval_lab import (
    ComparisonTolerance,
    ConfigurationError,
    EvaluationError,
    EvaluationResult,
    IncomparableRunError,
    LatencyStats,
    QualityGateCheck,
    QualityGateConfig,
    QualityGateReport,
    QualityGateResult,
    QueryEvaluation,
    RetrieverMetrics,
    evaluate_quality_gates,
)


def _result(
    value: float,
    *,
    run_id: str = "run",
    latency: float | None = None,
    manifest: Mapping[str, object] | None = None,
) -> EvaluationResult:
    base_manifest: dict[str, object] = {
        "dataset_hash": "dataset",
        "query_ids": ["q1", "q2"],
        "relevance_level": "document",
        "metric_version": 1,
        "top_k": [1],
    }
    if manifest is not None:
        base_manifest.update(manifest)
    queries = tuple(
        QueryEvaluation(
            query_id=query_id,
            retrieved_ids=(),
            metrics_by_cutoff={1: {"recall": value}},
        )
        for query_id in ("q1", "q2")
    )
    latency_values = None
    if latency is not None:
        latency_values = {
            "bm25": LatencyStats(
                mean_ms=latency,
                p50_ms=latency,
                p95_ms=latency,
                max_ms=latency,
                sample_count=2,
            )
        }
    return EvaluationResult(
        run_id=run_id,
        metrics={"bm25": RetrieverMetrics({1: {"recall": value}})},
        query_results={"bm25": queries},
        manifest=base_manifest,
        latency=latency_values,
    )


def test_min_and_max_constraints_are_evaluated() -> None:
    candidate = _result(0.6)
    report = evaluate_quality_gates(
        candidate,
        (
            QualityGateConfig(retriever="bm25", metric="recall@1", min_value=0.7),
            QualityGateConfig(retriever="bm25", metric="recall@1", max_value=0.7),
        ),
    )
    assert not report.passed
    assert len(report.results) == 2
    assert report.results[0].checks[0].reason == "candidate is below minimum"
    assert report.results[1].passed
    assert len(report.failed) == 1


def test_absolute_and_relative_retrieval_drop_use_higher_is_better() -> None:
    baseline = _result(0.8, run_id="baseline")
    candidate = _result(0.6, run_id="candidate")
    report = evaluate_quality_gates(
        candidate,
        (
            QualityGateConfig(
                retriever="bm25",
                metric="recall@1",
                max_absolute_drop=0.2,
                max_relative_drop=0.25,
            ),
        ),
        baseline=baseline,
    )
    assert report.passed
    assert [check.actual for check in report.results[0].checks] == pytest.approx(
        [0.2, 0.25]
    )
    failing = evaluate_quality_gates(
        candidate,
        (
            QualityGateConfig(
                retriever="bm25", metric="recall@1", max_absolute_drop=0.1
            ),
        ),
        baseline=baseline,
    )
    assert not failing.passed


def test_improvement_has_zero_drop() -> None:
    report = evaluate_quality_gates(
        _result(0.9, run_id="candidate"),
        (
            QualityGateConfig(
                retriever="bm25", metric="recall@1", max_relative_drop=0.0
            ),
        ),
        baseline=_result(0.5, run_id="baseline"),
    )
    check = report.results[0].checks[0]
    assert report.passed and check.actual == 0.0


def test_latency_constraints_are_lower_is_better() -> None:
    baseline = _result(0.5, run_id="baseline", latency=10.0)
    slower = _result(0.5, run_id="candidate", latency=15.0)
    report = evaluate_quality_gates(
        slower,
        (
            QualityGateConfig(
                retriever="bm25", metric="latency_p95_ms", max_value=12.0
            ),
            QualityGateConfig(
                retriever="bm25", metric="latency_p95_ms", max_absolute_drop=5.0
            ),
        ),
        baseline=baseline,
    )
    assert not report.results[0].passed
    assert report.results[1].passed
    faster = evaluate_quality_gates(
        _result(0.5, run_id="candidate", latency=5.0),
        (
            QualityGateConfig(
                retriever="bm25", metric="latency_p95_ms", max_absolute_drop=0.0
            ),
        ),
        baseline=baseline,
    )
    assert faster.passed and faster.results[0].checks[0].actual == 0.0
    relative = evaluate_quality_gates(
        slower,
        (
            QualityGateConfig(
                retriever="bm25", metric="latency_p95_ms", max_relative_drop=0.5
            ),
        ),
        baseline=baseline,
    )
    assert relative.passed
    assert relative.results[0].checks[0].actual == pytest.approx(0.5)


def test_baseline_zero_relative_cases_are_explicit() -> None:
    baseline = _result(0.0, run_id="baseline")
    both_zero = evaluate_quality_gates(
        _result(0.0, run_id="candidate"),
        (
            QualityGateConfig(
                retriever="bm25", metric="recall@1", max_relative_drop=0.0
            ),
        ),
        baseline=baseline,
    )
    assert both_zero.passed
    assert both_zero.results[0].checks[0].actual == 0.0
    improved = evaluate_quality_gates(
        _result(0.2, run_id="candidate"),
        (
            QualityGateConfig(
                retriever="bm25", metric="recall@1", max_relative_drop=0.0
            ),
        ),
        baseline=baseline,
    )
    assert improved.passed
    assert improved.results[0].checks[0].actual == 0.0
    regressed = evaluate_quality_gates(
        _result(-0.2, run_id="candidate"),
        (
            QualityGateConfig(
                retriever="bm25", metric="recall@1", max_relative_drop=0.0
            ),
        ),
        baseline=baseline,
    )
    assert not regressed.passed
    assert regressed.results[0].checks[0].actual is None
    assert "undefined" in regressed.results[0].checks[0].reason


def test_tolerance_applies_at_constraint_boundary() -> None:
    report = evaluate_quality_gates(
        _result(0.7 + 5e-13),
        (QualityGateConfig(retriever="bm25", metric="recall@1", min_value=0.7),),
        tolerance=ComparisonTolerance(absolute=1e-12, relative=0.0),
    )
    assert report.passed


def test_all_valid_gates_are_evaluated_without_short_circuit() -> None:
    report = evaluate_quality_gates(
        _result(0.5),
        (
            QualityGateConfig(retriever="bm25", metric="recall@1", min_value=0.9),
            QualityGateConfig(retriever="bm25", metric="recall@1", max_value=0.1),
        ),
    )
    assert len(report.results) == 2
    assert len(report.failed) == 2


def test_duplicate_targets_keep_distinct_gate_indexes_and_round_trip() -> None:
    candidate = _result(0.5, run_id="candidate")
    report = evaluate_quality_gates(
        candidate,
        (
            QualityGateConfig(retriever="bm25", metric="recall@1", min_value=0.4),
            QualityGateConfig(retriever="bm25", metric="recall@1", min_value=0.6),
        ),
    )
    assert [result.gate_index for result in report.results] == [0, 1]
    assert [result.passed for result in report.results] == [True, False]
    loaded = EvaluationResult.from_json(candidate.with_quality_gates(report).to_json())
    assert [result.gate_index for result in loaded.quality_gates] == [0, 1]


def test_empty_quality_gate_report_is_a_passing_report() -> None:
    report = evaluate_quality_gates(_result(0.5), ())
    assert report.passed
    assert report.results == ()
    assert report.failed == ()


def test_invalid_reference_and_missing_baseline_are_configuration_errors() -> None:
    candidate = _result(0.5)
    with pytest.raises(ConfigurationError):
        evaluate_quality_gates(
            candidate,
            (QualityGateConfig(retriever="missing", metric="recall@1", min_value=0.1),),
        )


def test_quality_engine_rejects_invalid_inputs_and_nonfinite_values() -> None:
    candidate = _result(0.5)
    with pytest.raises(EvaluationError):
        evaluate_quality_gates(object(), ())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        evaluate_quality_gates(candidate, "gates")  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        evaluate_quality_gates(candidate, (), baseline=object())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        evaluate_quality_gates(candidate, (), tolerance=object())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        evaluate_quality_gates(candidate, (object(),))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        evaluate_quality_gates(
            candidate,
            (
                QualityGateConfig(
                    retriever="bm25", metric="latency_p95_ms", max_value=10.0
                ),
            ),
        )
    with pytest.raises(ConfigurationError):
        evaluate_quality_gates(
            candidate,
            (QualityGateConfig(retriever="bm25", metric="mrr@1", min_value=0.1),),
        )
    metrics = candidate.metrics["bm25"]
    object.__setattr__(
        metrics,
        "metrics_by_cutoff",
        {1: {"recall": float("nan")}},
    )
    with pytest.raises(EvaluationError):
        evaluate_quality_gates(
            candidate,
            (QualityGateConfig(retriever="bm25", metric="recall@1", min_value=0.1),),
        )
    with pytest.raises(ConfigurationError):
        evaluate_quality_gates(
            candidate,
            (
                QualityGateConfig(
                    retriever="bm25", metric="recall@1", max_absolute_drop=0.1
                ),
            ),
        )
    with pytest.raises(ConfigurationError):
        evaluate_quality_gates(
            candidate,
            (QualityGateConfig(retriever="bm25", metric="mrr@1", min_value=0.1),),
        )


def test_incomparable_baseline_propagates() -> None:
    with pytest.raises(IncomparableRunError):
        evaluate_quality_gates(
            _result(0.5, run_id="candidate"),
            (
                QualityGateConfig(
                    retriever="bm25", metric="recall@1", max_absolute_drop=0.1
                ),
            ),
            baseline=_result(
                0.6,
                run_id="baseline",
                manifest={"dataset_hash": "different"},
            ),
        )


def test_gate_results_attach_and_round_trip() -> None:
    candidate = _result(0.5, run_id="candidate")
    report = evaluate_quality_gates(
        candidate,
        (QualityGateConfig(retriever="bm25", metric="recall@1", min_value=0.5),),
        tolerance=ComparisonTolerance(absolute=0.01, relative=0.02),
    )
    attached = candidate.with_quality_gates(report)
    loaded = EvaluationResult.from_json(attached.to_json())
    assert loaded.quality_gates == attached.quality_gates
    assert loaded.quality_gates[0].checks[0].absolute_tolerance == 0.01
    assert loaded.quality_gates[0].checks[0].relative_tolerance == 0.02
    assert loaded.to_json() == attached.to_json()
    assert "query_id" not in report.to_json()
    assert "retrieved_ids" not in report.to_json()


def test_malformed_gate_json_is_rejected() -> None:
    candidate = _result(0.5, run_id="candidate")
    report = evaluate_quality_gates(
        candidate,
        (QualityGateConfig(retriever="bm25", metric="recall@1", min_value=0.5),),
    )
    payload = json.loads(candidate.with_quality_gates(report).to_json())
    payload["quality_gates"][0]["passed"] = False
    with pytest.raises(EvaluationError):
        EvaluationResult.from_dict(payload)
    payload = json.loads(candidate.with_quality_gates(report).to_json())
    payload["quality_gates"].append(payload["quality_gates"][0])
    with pytest.raises(EvaluationError):
        EvaluationResult.from_dict(payload)
    payload = json.loads(candidate.with_quality_gates(report).to_json())
    payload["quality_gates"][0]["checks"][0]["actual"] = float("nan")
    with pytest.raises(EvaluationError):
        EvaluationResult.from_dict(payload)
    for field, value in (
        ("threshold", 1.0),
        ("passed", False),
        ("reason", "tampered"),
        ("absolute_tolerance", -1.0),
        ("relative_tolerance", True),
        ("status", "tampered"),
    ):
        payload = json.loads(candidate.with_quality_gates(report).to_json())
        payload["quality_gates"][0]["checks"][0][field] = value
        with pytest.raises(EvaluationError):
            EvaluationResult.from_dict(payload)
    for mutate in ("gate_index", "retriever", "metric", "candidate_actual"):
        payload = json.loads(candidate.with_quality_gates(report).to_json())
        gate = payload["quality_gates"][0]
        check = gate["checks"][0]
        if mutate == "gate_index":
            gate["gate_index"] = 2
        elif mutate == "retriever":
            gate["retriever"] = "missing"
            check.pop("retriever")
        elif mutate == "metric":
            gate["metric"] = "mrr@1"
            check.pop("metric")
        else:
            check["actual"] = 0.51
        with pytest.raises(EvaluationError):
            EvaluationResult.from_dict(payload)


@pytest.mark.parametrize(
    "bad",
    [
        {"retriever": "bm25", "metric": "recall@1"},
        {
            "retriever": "bm25",
            "metric": "recall@1",
            "max_absolute_drop": -0.1,
        },
        {
            "retriever": "bm25",
            "metric": "recall@1",
            "max_relative_drop": float("inf"),
        },
        {
            "retriever": "bm25",
            "metric": "recall@1",
            "min_value": True,
        },
        {
            "retriever": "bm25",
            "metric": "recall@1",
            "min_value": 0.8,
            "max_value": 0.2,
        },
    ],
)
def test_quality_gate_config_rejects_invalid_constraints(
    bad: Mapping[str, object],
) -> None:
    with pytest.raises(ConfigurationError):
        QualityGateConfig(**bad)  # type: ignore[arg-type]


def test_quality_gate_models_reject_nonfinite_and_internal_contradictions() -> None:
    with pytest.raises(EvaluationError):
        evaluate_quality_gates(
            _result(0.5),
            (QualityGateConfig(retriever="bm25", metric="recall@1", min_value=0.1),),
            tolerance=ComparisonTolerance(),
        ).results[0].checks[0].__class__(
            retriever="bm25",
            metric="recall@1",
            constraint="min_value",
            actual=float("nan"),
            threshold=0.1,
            passed=True,
            reason="ok",
            candidate_run_id="run",
        )
    drop = QualityGateCheck(
        retriever="bm25",
        metric="recall@1",
        constraint="max_absolute_drop",
        actual=0.0,
        threshold=0.0,
        passed=True,
        reason="regression is within threshold",
        candidate_run_id="candidate",
    )
    with pytest.raises(EvaluationError):
        QualityGateResult(
            retriever="bm25",
            metric="recall@1",
            checks=(drop,),
            passed=True,
            candidate_run_id="candidate",
        )
    absolute = evaluate_quality_gates(
        _result(0.5),
        (QualityGateConfig(retriever="bm25", metric="recall@1", min_value=0.1),),
    ).results[0]
    with pytest.raises(EvaluationError):
        QualityGateReport(
            candidate_run_id="run",
            baseline_run_id=None,
            results=(
                QualityGateResult(
                    retriever=absolute.retriever,
                    metric=absolute.metric,
                    checks=absolute.checks,
                    passed=absolute.passed,
                    candidate_run_id=absolute.candidate_run_id,
                    gate_index=2,
                ),
            ),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"constraint": "invalid"},
        {"actual": None},
        {
            "constraint": "max_absolute_drop",
            "threshold": -0.1,
            "reason": "regression is within threshold",
        },
        {"absolute_tolerance": -1.0},
        {"status": "invalid"},
        {"status": "undefined_baseline_zero_regression"},
        {"passed": 1},
        {"reason": "wrong"},
    ],
)
def test_quality_gate_check_rejects_invalid_contracts(
    changes: Mapping[str, object],
) -> None:
    values: dict[str, object] = {
        "retriever": "bm25",
        "metric": "recall@1",
        "constraint": "min_value",
        "actual": 0.5,
        "threshold": 0.4,
        "passed": True,
        "reason": "candidate meets minimum",
        "candidate_run_id": "candidate",
    }
    values.update(changes)
    with pytest.raises(EvaluationError):
        QualityGateCheck(**values)  # type: ignore[arg-type]


def test_quality_gate_result_and_report_reject_invalid_nested_contracts() -> None:
    check = QualityGateCheck(
        retriever="bm25",
        metric="recall@1",
        constraint="min_value",
        actual=0.5,
        threshold=0.4,
        passed=True,
        reason="candidate meets minimum",
        candidate_run_id="candidate",
    )
    valid = QualityGateResult(
        retriever="bm25",
        metric="recall@1",
        checks=(check,),
        passed=True,
        candidate_run_id="candidate",
    )
    with pytest.raises(EvaluationError):
        QualityGateResult("bm25", "recall@1", (), True, "candidate")
    with pytest.raises(EvaluationError):
        QualityGateResult(
            "bm25",
            "recall@1",
            (object(),),
            True,
            "candidate",  # type: ignore[arg-type]
        )
    with pytest.raises(EvaluationError):
        QualityGateResult("bm25", "recall@1", (check, check), True, "candidate")
    with pytest.raises(EvaluationError):
        QualityGateResult("dense", "recall@1", (check,), True, "candidate")
    with pytest.raises(EvaluationError):
        QualityGateResult("bm25", "recall@1", (check,), True, "other")
    with pytest.raises(EvaluationError):
        QualityGateResult("bm25", "recall@1", (check,), True, "candidate", "baseline")
    with pytest.raises(EvaluationError):
        QualityGateResult(
            "bm25",
            "recall@1",
            (check,),
            1,
            "candidate",  # type: ignore[arg-type]
        )
    with pytest.raises(EvaluationError):
        QualityGateResult(
            "bm25", "recall@1", (check,), True, "candidate", gate_index=True
        )
    with pytest.raises(EvaluationError):
        QualityGateReport("candidate", None, (object(),))  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        QualityGateReport("other", None, (valid,))
    with pytest.raises(EvaluationError):
        QualityGateReport("candidate", "baseline", (valid,))
