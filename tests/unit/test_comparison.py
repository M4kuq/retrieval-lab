from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import pytest

from retrieval_lab import (
    ComparabilityIssue,
    ComparabilityReport,
    ComparisonTolerance,
    EvaluationResult,
    IncomparableRunError,
    LatencyStats,
    MetricComparison,
    MetricDelta,
    QueryDeltaExtreme,
    QueryEvaluation,
    RetrieverMetrics,
    RunComparison,
    check_comparability,
    compare_runs,
)
from retrieval_lab.exceptions import EvaluationError


def _result(
    values: Mapping[str, Sequence[float]],
    *,
    retrievers: Sequence[str] = ("bm25",),
    manifest: Mapping[str, object] | None = None,
    latency: bool = False,
    query_order: Sequence[str] = ("q1", "q2", "q3"),
) -> EvaluationResult:
    base_manifest: dict[str, object] = {
        "dataset_hash": "dataset-hash",
        "query_ids": list(query_order),
        "relevance_level": "document",
        "metric_version": 1,
        "top_k": [1, 3],
    }
    if manifest is not None:
        base_manifest.update(manifest)

    metrics: dict[str, RetrieverMetrics] = {}
    query_results: dict[str, tuple[QueryEvaluation, ...]] = {}
    latency_values: dict[str, LatencyStats] = {}
    for index, name in enumerate(retrievers):
        query_values = values[name]
        queries = tuple(
            QueryEvaluation(
                query_id=query_id,
                retrieved_ids=(),
                metrics_by_cutoff={
                    1: {"recall": query_values[position]},
                    3: {"recall": query_values[position]},
                },
            )
            for position, query_id in enumerate(query_order)
        )
        query_results[name] = queries
        metrics[name] = RetrieverMetrics(
            metrics_by_cutoff={
                1: {"recall": sum(query_values) / len(query_values)},
                3: {"recall": sum(query_values) / len(query_values)},
            }
        )
        if latency:
            latency_values[name] = LatencyStats(
                mean_ms=10.0 + index,
                p50_ms=9.0 + index,
                p95_ms=20.0 + index,
                max_ms=30.0 + index,
                sample_count=3,
            )
    return EvaluationResult(
        run_id="run-id",
        metrics=metrics,
        query_results=query_results,
        manifest=cast(Mapping[str, object], base_manifest),
        latency=latency_values if latency else None,
    )


def test_complete_comparison_is_deterministic_and_reports_query_extremes() -> None:
    baseline = _result({"bm25": (0.2, 0.4, 0.6)})
    candidate = _result(
        {"bm25": (0.4, 0.4, 0.2)},
        query_order=("q3", "q2", "q1"),
    )

    report = check_comparability(baseline, candidate)
    comparison = compare_runs(baseline, candidate)

    assert report.comparable
    assert set(comparison.metrics) == {"bm25"}
    recall = comparison.metrics["bm25"][0]
    assert isinstance(recall, MetricComparison)
    assert recall.aggregate.absolute_delta == pytest.approx(-0.0666666666667)
    assert [delta.query_id for delta in recall.query_deltas] == ["q1", "q2", "q3"]
    assert recall.best is not None and recall.best.query_id == "q1"
    assert recall.worst is not None and recall.worst.query_id == "q3"


def test_query_extreme_ties_use_query_id_order() -> None:
    baseline = _result({"bm25": (0.2, 0.2, 0.2)})
    candidate = _result({"bm25": (0.3, 0.3, 0.3)})

    recall = compare_runs(baseline, candidate).metrics["bm25"][0]

    assert [delta.query_id for delta in recall.query_deltas] == ["q1", "q2", "q3"]
    assert recall.best is not None and recall.best.query_id == "q1"
    assert recall.worst is not None and recall.worst.query_id == "q1"


def test_added_and_removed_retrievers_are_non_blocking() -> None:
    baseline = _result({"bm25": (0.2, 0.4, 0.6)})
    candidate = _result(
        {"bm25": (0.2, 0.4, 0.6), "dense": (0.1, 0.2, 0.3)},
        retrievers=("bm25", "dense"),
    )

    report = check_comparability(baseline, candidate)
    assert report.comparable
    assert report.common_retrievers == ("bm25",)
    assert report.added_retrievers == ("dense",)
    assert set(compare_runs(baseline, candidate).comparisons) == {"bm25"}


def test_multiple_strict_issues_are_collected() -> None:
    baseline = _result({"bm25": (0.2, 0.4, 0.6)})
    candidate = _result(
        {"bm25": (0.2, 0.4, 0.6)},
        manifest={
            "dataset_hash": "other",
            "relevance_level": "chunk",
            "metric_version": 2,
            "top_k": [5],
            "query_ids": ["q1", "q2"],
        },
    )

    report = check_comparability(baseline, candidate)
    fields = {issue.field for issue in report.issues}
    assert {
        "manifest.dataset_hash",
        "manifest.relevance_level",
        "manifest.metric_version",
        "manifest.top_k",
        "manifest.query_ids",
    } <= fields
    with pytest.raises(IncomparableRunError) as captured:
        compare_runs(baseline, candidate)
    assert len(captured.value.issues) >= 5


def test_missing_strict_fields_are_all_blocking() -> None:
    baseline = _result({"bm25": (0.2, 0.4, 0.6)})
    candidate = _result({"bm25": (0.2, 0.4, 0.6)})
    candidate_manifest = dict(candidate.manifest)
    for field in (
        "dataset_hash",
        "relevance_level",
        "metric_version",
        "top_k",
        "query_ids",
    ):
        candidate_manifest.pop(field)
    candidate = EvaluationResult(
        run_id=candidate.run_id,
        metrics=candidate.metrics,
        query_results=candidate.query_results,
        manifest=candidate_manifest,
    )

    report = check_comparability(baseline, candidate)
    assert {issue.field for issue in report.issues} >= {
        "manifest.dataset_hash",
        "manifest.relevance_level",
        "manifest.metric_version",
        "manifest.top_k",
        "manifest.query_ids",
    }


def test_baseline_zero_relative_delta_is_explicit() -> None:
    baseline = _result({"bm25": (0.0, 0.0, 0.0)})
    candidate = _result({"bm25": (0.0, 0.2, 0.4)})
    comparison = compare_runs(baseline, candidate)

    aggregate = comparison.metrics["bm25"][0].aggregate
    assert aggregate.relative_delta is None
    assert aggregate.relative_status == "baseline_zero"
    assert aggregate.classification == "improved"

    equal = compare_runs(baseline, _result({"bm25": (0.0, 0.0, 0.0)}))
    assert equal.metrics["bm25"][0].aggregate.relative_delta == 0.0
    assert (
        equal.metrics["bm25"][0].aggregate.relative_status == "baseline_zero_both_zero"
    )


def test_tolerance_boundary_and_invalid_tolerance() -> None:
    baseline = _result({"bm25": (0.2, 0.2, 0.2)})
    candidate = _result({"bm25": (0.2 + 1e-12, 0.2, 0.2)})
    tolerance = ComparisonTolerance(absolute=1e-9, relative=0.0)
    delta = compare_runs(baseline, candidate, tolerance=tolerance).metrics["bm25"][0]
    assert delta.aggregate.classification == "unchanged"
    with pytest.raises(EvaluationError):
        ComparisonTolerance(absolute=float("inf"))
    with pytest.raises(EvaluationError):
        ComparisonTolerance(relative=True)  # type: ignore[arg-type]


def test_latency_is_lower_is_better_and_missing_latency_is_diagnostic() -> None:
    baseline = _result({"bm25": (0.2, 0.4, 0.6)}, latency=True)
    candidate_base = _result({"bm25": (0.2, 0.4, 0.6)}, latency=True)
    candidate = EvaluationResult(
        run_id=candidate_base.run_id,
        metrics=candidate_base.metrics,
        query_results=candidate_base.query_results,
        manifest=candidate_base.manifest,
        latency={
            "bm25": LatencyStats(
                mean_ms=8.0,
                p50_ms=7.0,
                p95_ms=18.0,
                max_ms=28.0,
                sample_count=3,
            )
        },
    )
    comparison = compare_runs(baseline, candidate)
    assert {
        item.metric
        for item in comparison.metrics["bm25"]
        if item.metric.startswith("latency_")
    } == {
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_max_ms",
    }
    latency = next(
        item for item in comparison.metrics["bm25"] if item.metric == "latency_p95_ms"
    )
    assert latency.aggregate.absolute_delta == -2.0
    assert latency.aggregate.direction == "lower_is_better"
    assert latency.aggregate.classification == "improved"

    missing = _result({"bm25": (0.2, 0.4, 0.6)})
    report = check_comparability(baseline, missing)
    assert report.comparable
    assert report.diagnostics[0].field == "latency"
    assert not report.diagnostics[0].blocking
    assert all(
        item.metric == "recall"
        for item in compare_runs(baseline, missing).metrics["bm25"]
    )


def test_metric_shape_mismatch_is_blocking() -> None:
    baseline = _result({"bm25": (0.2, 0.4, 0.6)})
    altered_query = QueryEvaluation(
        query_id="q1",
        retrieved_ids=(),
        metrics_by_cutoff={1: {"recall": 0.2}},
    )
    candidate = _result({"bm25": (0.2, 0.4, 0.6)})
    candidate = EvaluationResult(
        run_id=candidate.run_id,
        metrics=candidate.metrics,
        query_results={"bm25": (altered_query, *candidate.query_results["bm25"][1:])},
        manifest=candidate.manifest,
    )
    report = check_comparability(baseline, candidate)
    assert any("metric_shape" in issue.field for issue in report.issues)


def test_matching_runs_with_internal_aggregate_shape_mismatch_are_blocked() -> None:
    source = _result({"bm25": (0.2, 0.4, 0.6)})
    metrics = {"bm25": RetrieverMetrics({1: {"recall": 0.4}})}
    baseline = EvaluationResult(
        run_id=source.run_id,
        metrics=metrics,
        query_results=source.query_results,
        manifest=source.manifest,
    )
    candidate = EvaluationResult(
        run_id=source.run_id,
        metrics=metrics,
        query_results=source.query_results,
        manifest=source.manifest,
    )

    report = check_comparability(baseline, candidate)

    assert not report.comparable
    assert any(
        "does not match baseline aggregate" in issue.reason for issue in report.issues
    )


def test_added_retriever_shape_is_not_a_comparability_blocker() -> None:
    baseline = _result({"bm25": (0.2, 0.4, 0.6)})
    candidate_base = _result(
        {"bm25": (0.2, 0.4, 0.6), "dense": (0.1, 0.2, 0.3)},
        retrievers=("bm25", "dense"),
    )
    dense_queries = tuple(
        QueryEvaluation(
            query_id=query_id,
            retrieved_ids=(),
            metrics_by_cutoff={2: {"mrr": 0.1 + index / 10}},
        )
        for index, query_id in enumerate(("q1", "q2", "q3"))
    )
    candidate = EvaluationResult(
        run_id=candidate_base.run_id,
        metrics={
            **candidate_base.metrics,
            "dense": RetrieverMetrics({2: {"mrr": 0.2}}),
        },
        query_results={
            **candidate_base.query_results,
            "dense": dense_queries,
        },
        manifest=candidate_base.manifest,
    )

    report = check_comparability(baseline, candidate)
    assert report.comparable
    assert report.added_retrievers == ("dense",)


def test_metric_shapes_are_checked_per_common_retriever() -> None:
    baseline_base = _result(
        {"bm25": (0.2, 0.4, 0.6), "keyword": (0.2, 0.4, 0.6)},
        retrievers=("bm25", "keyword"),
    )
    candidate_base = _result(
        {"bm25": (0.2, 0.4, 0.6), "keyword": (0.2, 0.4, 0.6)},
        retrievers=("bm25", "keyword"),
    )

    def reshaped(
        source: EvaluationResult,
        name: str,
        cutoff: int,
        metric: str,
    ) -> EvaluationResult:
        query_results = {
            retriever: tuple(values)
            for retriever, values in source.query_results.items()
        }
        values = tuple(
            QueryEvaluation(
                query_id=query.query_id,
                retrieved_ids=(),
                metrics_by_cutoff={cutoff: {metric: 0.5}},
            )
            for query in query_results[name]
        )
        query_results[name] = values
        metrics = dict(source.metrics)
        metrics[name] = RetrieverMetrics({cutoff: {metric: 0.5}})
        return EvaluationResult(
            run_id=source.run_id,
            metrics=metrics,
            query_results=query_results,
            manifest=source.manifest,
        )

    baseline = reshaped(baseline_base, "keyword", 2, "mrr")
    candidate = reshaped(candidate_base, "bm25", 2, "mrr")
    report = check_comparability(baseline, candidate)
    shape_fields = [
        issue.field for issue in report.issues if "metric_shape" in issue.field
    ]
    assert shape_fields.count("query_results[bm25].metric_shape") == 3
    assert shape_fields.count("query_results[keyword].metric_shape") == 3
    assert shape_fields.count("retrievers['bm25'].metric_shape") == 1
    assert shape_fields.count("retrievers['keyword'].metric_shape") == 1


def test_query_ids_are_not_interpolated_into_incomparable_error_message() -> None:
    baseline = _result(
        {"bm25": (0.2, 0.4, 0.6)},
        manifest={"query_ids": ["/secret/path"]},
    )
    candidate = _result(
        {"bm25": (0.2, 0.4, 0.6)},
        manifest={"query_ids": ["/other/path"]},
    )
    with pytest.raises(IncomparableRunError) as captured:
        compare_runs(baseline, candidate)
    message = str(captured.value)
    assert "/secret/path" not in message
    assert "/other/path" not in message
    manifest_issue = next(
        issue for issue in captured.value.issues if issue.field == "manifest.query_ids"
    )
    assert manifest_issue.baseline_value == ["/secret/path"]
    assert manifest_issue.candidate_value == ["/other/path"]


def test_retriever_names_are_not_interpolated_into_incomparable_error_message() -> None:
    baseline = _result(
        {"/secret/retriever": (0.2, 0.4, 0.6)},
        retrievers=("/secret/retriever",),
    )
    candidate = _result(
        {"/other/retriever": (0.2, 0.4, 0.6)},
        retrievers=("/other/retriever",),
    )
    with pytest.raises(IncomparableRunError) as captured:
        compare_runs(baseline, candidate)
    message = str(captured.value)
    assert "/secret/retriever" not in message
    assert "/other/retriever" not in message


def test_common_retriever_query_ids_must_match_manifest() -> None:
    baseline = _result({"bm25": (0.2, 0.4, 0.6)})
    candidate_base = _result({"bm25": (0.2, 0.4, 0.6)})
    candidate = EvaluationResult(
        run_id=candidate_base.run_id,
        metrics=candidate_base.metrics,
        query_results={"bm25": candidate_base.query_results["bm25"][:2]},
        manifest=candidate_base.manifest,
    )
    report = check_comparability(baseline, candidate)
    assert any(
        issue.field == "query_results[bm25].query_ids" for issue in report.issues
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_metric_is_rejected_during_comparison(
    bad_value: float,
) -> None:
    baseline = _result({"bm25": (0.2, 0.4, 0.6)})
    metrics = baseline.metrics["bm25"]
    object.__setattr__(
        metrics,
        "metrics_by_cutoff",
        {1: {"recall": bad_value}, 3: {"recall": bad_value}},
    )
    candidate = _result({"bm25": (0.2, 0.4, 0.6)})
    with pytest.raises(EvaluationError):
        compare_runs(baseline, candidate)


def test_json_round_trip_result_can_be_compared() -> None:
    baseline = _result({"bm25": (0.2, 0.4, 0.6)}, latency=True)
    candidate = EvaluationResult.from_json(baseline.to_json())
    comparison = compare_runs(baseline, candidate)
    assert comparison.metrics["bm25"][0].aggregate.absolute_delta == 0.0


def test_tiny_nonzero_baseline_has_defined_relative_delta() -> None:
    baseline = _result({"bm25": (1e-13, 1e-13, 1e-13)})
    candidate = _result({"bm25": (2e-13, 2e-13, 2e-13)})
    delta = compare_runs(baseline, candidate).metrics["bm25"][0].aggregate
    assert delta.relative_status == "defined"
    assert delta.relative_delta == pytest.approx(1.0)


def test_runtime_changes_are_non_blocking_experimental_differences() -> None:
    baseline = _result(
        {"bm25": (0.2, 0.4, 0.6)},
        manifest={"runtime": {"started_at_utc": "first"}},
    )
    candidate = _result(
        {"bm25": (0.2, 0.4, 0.6)},
        manifest={"runtime": {"started_at_utc": "second"}},
    )

    report = check_comparability(baseline, candidate)

    assert report.comparable
    runtime = next(
        issue for issue in report.variable_differences if issue.field == "runtime"
    )
    assert not runtime.blocking


def test_comparison_models_reject_invalid_values() -> None:
    with pytest.raises(EvaluationError):
        ComparisonTolerance(absolute=-1.0)
    with pytest.raises(EvaluationError):
        ComparisonTolerance(relative="bad")  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        ComparabilityIssue(field="", reason="bad")
    with pytest.raises(EvaluationError):
        ComparabilityIssue(field="x", reason="", blocking=1)  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        ComparabilityReport(issues=(object(),))  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        ComparabilityReport(
            issues=(ComparabilityIssue("x", "diagnostic", blocking=False),)
        )
    with pytest.raises(EvaluationError):
        ComparabilityReport(diagnostics=(ComparabilityIssue("x", "blocking"),))
    with pytest.raises(EvaluationError):
        ComparabilityReport(common_retrievers=("",))
    with pytest.raises(EvaluationError):
        QueryDeltaExtreme("", 0.0, 0.0)
    with pytest.raises(EvaluationError):
        QueryDeltaExtreme("q", float("inf"), 0.0)


def _valid_delta(**changes: object) -> MetricDelta:
    values: dict[str, object] = {
        "retriever": "bm25",
        "metric": "recall",
        "cutoff": 1,
        "baseline": 0.5,
        "candidate": 0.6,
        "absolute_delta": 0.1,
        "relative_delta": 0.2,
        "relative_status": "defined",
        "classification": "improved",
        "direction": "higher_is_better",
    }
    values.update(changes)
    return MetricDelta(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"retriever": ""},
        {"metric": ""},
        {"cutoff": 0},
        {"baseline": float("nan")},
        {"relative_delta": float("inf")},
        {"relative_status": "other"},
        {"classification": "other"},
        {"direction": "other"},
        {"query_id": ""},
    ],
)
def test_metric_delta_rejects_invalid_values(changes: dict[str, object]) -> None:
    with pytest.raises(EvaluationError):
        _valid_delta(**changes)


def test_comparison_models_reject_invalid_nested_values() -> None:
    delta = _valid_delta()
    extreme = QueryDeltaExtreme("q1", 0.1, 0.1)
    with pytest.raises(EvaluationError):
        MetricComparison("bm25", "recall", 1, object())  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        MetricComparison("dense", "recall", 1, delta)
    with pytest.raises(EvaluationError):
        MetricComparison("bm25", "recall", 1, delta, query_deltas=(object(),))  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        MetricComparison("bm25", "recall", 1, delta, query_deltas=(delta,))
    with pytest.raises(EvaluationError):
        MetricComparison(
            "bm25",
            "recall",
            1,
            delta,
            query_deltas=(_valid_delta(retriever="dense", query_id="q1"),),
        )
    with pytest.raises(EvaluationError):
        MetricComparison("bm25", "recall", 1, delta, best=object())  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        MetricComparison("bm25", "recall", 1, delta, worst=object())  # type: ignore[arg-type]
    assert (
        MetricComparison(
            "bm25",
            "recall",
            1,
            delta,
            query_deltas=(_valid_delta(query_id="q1"),),
            best=extreme,
            worst=extreme,
        ).delta
        == delta
    )
    with pytest.raises(EvaluationError):
        RunComparison("", "candidate", ComparabilityReport(), {})
    with pytest.raises(EvaluationError):
        RunComparison("baseline", "candidate", object(), {})  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        RunComparison("baseline", "candidate", ComparabilityReport(), [])  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        RunComparison("baseline", "candidate", ComparabilityReport(), {"": ()})
    with pytest.raises(EvaluationError):
        RunComparison(
            "baseline",
            "candidate",
            ComparabilityReport(),
            {"bm25": (object(),)},  # type: ignore[arg-type]
        )


def test_public_comparison_input_validation() -> None:
    result = _result({"bm25": (0.2, 0.4, 0.6)})
    with pytest.raises(EvaluationError):
        check_comparability(cast(EvaluationResult, object()), result)
    with pytest.raises(EvaluationError):
        check_comparability(result, cast(EvaluationResult, object()))
    with pytest.raises(EvaluationError):
        compare_runs(result, result, tolerance=cast(ComparisonTolerance, object()))
