"""Deterministic comparison of two saved evaluation results."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

from retrieval_lab.domain import EvaluationResult, JSONValue, QueryEvaluation
from retrieval_lab.exceptions import EvaluationError, IncomparableRunError

MetricDirection = Literal["higher_is_better", "lower_is_better"]
RelativeStatus = Literal["defined", "baseline_zero_both_zero", "baseline_zero"]
DeltaClassification = Literal["improved", "regressed", "unchanged"]

_STRICT_FIELDS = (
    "dataset_hash",
    "relevance_level",
    "metric_version",
    "top_k",
)
_LATENCY_METRICS = (
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_max_ms",
)


def _finite_non_negative(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{field_name} must be a finite non-negative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise EvaluationError(f"{field_name} must be a finite non-negative number")
    return normalized


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise EvaluationError(f"{field_name} must be a finite number")
    return normalized


def _require_tolerance(value: object) -> ComparisonTolerance:
    if not isinstance(value, ComparisonTolerance):
        raise EvaluationError("tolerance must be a ComparisonTolerance")
    return value


@dataclass(frozen=True)
class ComparisonTolerance:
    """Absolute and relative tolerances used for equality and classifications."""

    absolute: float = 1e-12
    relative: float = 1e-9

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "absolute",
            _finite_non_negative(self.absolute, "tolerance.absolute"),
        )
        object.__setattr__(
            self,
            "relative",
            _finite_non_negative(self.relative, "tolerance.relative"),
        )

    def close(self, left: float, right: float) -> bool:
        """Return whether two finite values are equal within this tolerance."""
        return math.isclose(
            left,
            right,
            rel_tol=self.relative,
            abs_tol=self.absolute,
        )


_DEFAULT_TOLERANCE = ComparisonTolerance()


@dataclass(frozen=True)
class ComparabilityIssue:
    """One blocking or diagnostic difference between two result manifests."""

    field: str
    reason: str
    baseline_value: JSONValue | None = None
    candidate_value: JSONValue | None = None
    blocking: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise EvaluationError("ComparabilityIssue.field must be non-empty")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise EvaluationError("ComparabilityIssue.reason must be non-empty")
        if not isinstance(self.blocking, bool):
            raise EvaluationError("ComparabilityIssue.blocking must be boolean")


@dataclass(frozen=True)
class ComparabilityReport:
    """Complete comparability diagnostics, including non-blocking run changes."""

    issues: tuple[ComparabilityIssue, ...] = ()
    diagnostics: tuple[ComparabilityIssue, ...] = ()
    variable_differences: tuple[ComparabilityIssue, ...] = ()
    common_retrievers: tuple[str, ...] = ()
    added_retrievers: tuple[str, ...] = ()
    removed_retrievers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "issues",
            "diagnostics",
            "variable_differences",
        ):
            values = tuple(getattr(self, field_name))
            if not all(isinstance(value, ComparabilityIssue) for value in values):
                raise EvaluationError(
                    f"ComparabilityReport.{field_name} must contain issues"
                )
            if field_name == "issues" and any(not value.blocking for value in values):
                raise EvaluationError(
                    "ComparabilityReport.issues must contain only blocking issues"
                )
            if field_name != "issues" and any(value.blocking for value in values):
                raise EvaluationError(
                    f"ComparabilityReport.{field_name} must be non-blocking"
                )
            object.__setattr__(self, field_name, values)
        for field_name in (
            "common_retrievers",
            "added_retrievers",
            "removed_retrievers",
        ):
            values = tuple(getattr(self, field_name))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise EvaluationError(
                    f"ComparabilityReport.{field_name} must contain names"
                )
            object.__setattr__(self, field_name, tuple(sorted(set(values))))

    @property
    def blocking_issues(self) -> tuple[ComparabilityIssue, ...]:
        """Return issues that prevent numeric comparison."""
        return self.issues

    @property
    def comparable(self) -> bool:
        """Return whether all required comparison preconditions hold."""
        return not self.issues

    @property
    def is_comparable(self) -> bool:
        """Alias for :attr:`comparable` used by report consumers."""
        return self.comparable

    @property
    def reasons(self) -> tuple[ComparabilityIssue, ...]:
        """Alias for the complete blocking issue tuple."""
        return self.issues


@dataclass(frozen=True)
class QueryDeltaExtreme:
    """Best or worst query-level change for one metric."""

    query_id: str
    delta: float
    directional_delta: float

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise EvaluationError("QueryDeltaExtreme.query_id must be non-empty")
        object.__setattr__(self, "delta", _finite(self.delta, "query delta"))
        object.__setattr__(
            self,
            "directional_delta",
            _finite(self.directional_delta, "query directional delta"),
        )


@dataclass(frozen=True)
class MetricDelta:
    """Aggregate or query-level metric change from baseline to candidate."""

    retriever: str
    metric: str
    cutoff: int | None
    baseline: float
    candidate: float
    absolute_delta: float
    relative_delta: float | None
    relative_status: RelativeStatus
    classification: DeltaClassification
    direction: MetricDirection
    query_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("retriever", "metric"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise EvaluationError(f"MetricDelta.{field_name} must be non-empty")
        if self.cutoff is not None and (
            isinstance(self.cutoff, bool)
            or not isinstance(self.cutoff, int)
            or self.cutoff <= 0
        ):
            raise EvaluationError("MetricDelta.cutoff must be positive or None")
        for field_name in ("baseline", "candidate", "absolute_delta"):
            object.__setattr__(
                self,
                field_name,
                _finite(getattr(self, field_name), f"MetricDelta.{field_name}"),
            )
        if self.relative_delta is not None:
            object.__setattr__(
                self,
                "relative_delta",
                _finite(self.relative_delta, "MetricDelta.relative_delta"),
            )
        if self.relative_status not in (
            "defined",
            "baseline_zero_both_zero",
            "baseline_zero",
        ):
            raise EvaluationError("MetricDelta.relative_status is invalid")
        if self.classification not in ("improved", "regressed", "unchanged"):
            raise EvaluationError("MetricDelta.classification is invalid")
        if self.direction not in ("higher_is_better", "lower_is_better"):
            raise EvaluationError("MetricDelta.direction is invalid")
        if self.query_id is not None and (
            not isinstance(self.query_id, str) or not self.query_id.strip()
        ):
            raise EvaluationError("MetricDelta.query_id must be non-empty or None")


@dataclass(frozen=True)
class MetricComparison:
    """Aggregate metric delta and deterministic per-query evidence."""

    retriever: str
    metric: str
    cutoff: int | None
    aggregate: MetricDelta
    query_deltas: tuple[MetricDelta, ...] = ()
    best: QueryDeltaExtreme | None = None
    worst: QueryDeltaExtreme | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.aggregate, MetricDelta):
            raise EvaluationError("MetricComparison.aggregate must be MetricDelta")
        if (
            self.retriever != self.aggregate.retriever
            or self.metric != self.aggregate.metric
            or self.cutoff != self.aggregate.cutoff
        ):
            raise EvaluationError(
                "MetricComparison identity must match its aggregate delta"
            )
        values = tuple(self.query_deltas)
        if not all(isinstance(value, MetricDelta) for value in values):
            raise EvaluationError("MetricComparison.query_deltas must contain deltas")
        if any(value.query_id is None for value in values):
            raise EvaluationError("MetricComparison.query_deltas require query IDs")
        if any(
            value.retriever != self.retriever
            or value.metric != self.metric
            or value.cutoff != self.cutoff
            for value in values
        ):
            raise EvaluationError(
                "MetricComparison.query_deltas must match the comparison identity"
            )
        if self.best is not None and not isinstance(self.best, QueryDeltaExtreme):
            raise EvaluationError("MetricComparison.best must be a query extreme")
        if self.worst is not None and not isinstance(self.worst, QueryDeltaExtreme):
            raise EvaluationError("MetricComparison.worst must be a query extreme")
        object.__setattr__(self, "query_deltas", values)

    @property
    def delta(self) -> MetricDelta:
        """Return the aggregate delta."""
        return self.aggregate


@dataclass(frozen=True)
class RunComparison:
    """Deterministic comparison for the common retrievers in two runs."""

    baseline_run_id: str
    candidate_run_id: str
    comparability: ComparabilityReport
    metrics: Mapping[str, tuple[MetricComparison, ...]]

    def __post_init__(self) -> None:
        for field_name in ("baseline_run_id", "candidate_run_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise EvaluationError(f"RunComparison.{field_name} must be non-empty")
        if not isinstance(self.comparability, ComparabilityReport):
            raise EvaluationError("RunComparison.comparability must be a report")
        if not isinstance(self.metrics, Mapping):
            raise EvaluationError("RunComparison.metrics must be a mapping")
        normalized: dict[str, tuple[MetricComparison, ...]] = {}
        for name, values in self.metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise EvaluationError("RunComparison.metrics names must be non-empty")
            normalized[name] = tuple(values)
            if not all(isinstance(value, MetricComparison) for value in values):
                raise EvaluationError(
                    f"RunComparison.metrics[{name!r}] contains an invalid comparison"
                )
        object.__setattr__(
            self, "metrics", MappingProxyType(dict(sorted(normalized.items())))
        )

    @property
    def retrievers(self) -> Mapping[str, tuple[MetricComparison, ...]]:
        """Compatibility-friendly alias for the per-retriever comparisons."""
        return self.metrics

    @property
    def comparisons(self) -> Mapping[str, tuple[MetricComparison, ...]]:
        """Return per-retriever metric comparisons."""
        return self.metrics


def _manifest_value(result: EvaluationResult, field: str) -> object:
    return result.manifest.get(field)


def _json_value(value: object) -> JSONValue | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return None


def _strict_issue(
    field: str,
    reason: str,
    baseline: object = None,
    candidate: object = None,
    *,
    blocking: bool = True,
) -> ComparabilityIssue:
    return ComparabilityIssue(
        field=field,
        reason=reason,
        baseline_value=_json_value(baseline),
        candidate_value=_json_value(candidate),
        blocking=blocking,
    )


def _query_ids(result: EvaluationResult) -> tuple[str, ...] | None:
    raw = _manifest_value(result, "query_ids")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return None
    values = tuple(raw)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        return None
    if len(set(values)) != len(values):
        return None
    return tuple(sorted(values))


def _top_k(result: EvaluationResult) -> tuple[int, ...] | None:
    raw = _manifest_value(result, "top_k")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return None
    values = tuple(raw)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        return None
    if len(set(values)) != len(values):
        return None
    return tuple(sorted(values))


def _metric_shapes(
    result: EvaluationResult,
    retrievers: Sequence[str] | None = None,
) -> dict[str, dict[str, frozenset[tuple[str, int]]]]:
    shapes: dict[str, dict[str, set[tuple[str, int]]]] = {}
    names = (
        result.query_results
        if retrievers is None
        else {name: result.query_results[name] for name in retrievers}
    )
    for retriever, query_results in names.items():
        for query in query_results:
            shape = shapes.setdefault(retriever, {}).setdefault(query.query_id, set())
            shape.update(
                (name, cutoff)
                for cutoff, values in query.metrics_by_cutoff.items()
                for name in values
            )
    return {
        retriever: {
            query_id: frozenset(values) for query_id, values in query_shapes.items()
        }
        for retriever, query_shapes in shapes.items()
    }


def _shape_labels(
    shape: Sequence[tuple[str, int]] | frozenset[tuple[str, int]],
) -> list[str]:
    return [f"{name}@{cutoff}" for name, cutoff in sorted(shape)]


def _result_query_map(
    result: EvaluationResult,
    retriever: str,
) -> dict[str, QueryEvaluation]:
    return {query.query_id: query for query in result.query_results[retriever]}


def _variable_issues(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
) -> tuple[ComparabilityIssue, ...]:
    fields = (
        "corpus_hash",
        "chunk_hash",
        "chunking",
        "retriever_settings",
        "config",
        "seed",
        "runtime",
        "run_id",
    )
    differences: list[ComparabilityIssue] = []
    for field in fields:
        baseline_value: object = (
            baseline.run_id if field == "run_id" else _manifest_value(baseline, field)
        )
        candidate_value: object = (
            candidate.run_id if field == "run_id" else _manifest_value(candidate, field)
        )
        if baseline_value != candidate_value:
            differences.append(
                ComparabilityIssue(
                    field=field,
                    reason="experimental variable differs",
                    baseline_value=_json_value(baseline_value),
                    candidate_value=_json_value(candidate_value),
                    blocking=False,
                )
            )
    return tuple(differences)


def check_comparability(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
) -> ComparabilityReport:
    """Diagnose whether two results satisfy all strict comparison contracts."""
    if not isinstance(baseline, EvaluationResult):
        raise EvaluationError("baseline must be an EvaluationResult")
    if not isinstance(candidate, EvaluationResult):
        raise EvaluationError("candidate must be an EvaluationResult")

    issues: list[ComparabilityIssue] = []
    for field in _STRICT_FIELDS:
        left = _manifest_value(baseline, field)
        right = _manifest_value(candidate, field)
        if left is None:
            issues.append(_strict_issue(f"manifest.{field}", "missing from baseline"))
        if right is None:
            issues.append(_strict_issue(f"manifest.{field}", "missing from candidate"))
        if left is not None and right is not None:
            if field == "top_k":
                left_normalized = _top_k(baseline)
                right_normalized = _top_k(candidate)
                if left_normalized is None:
                    issues.append(
                        _strict_issue(
                            "manifest.top_k", "invalid in baseline", left, right
                        )
                    )
                elif right_normalized is None:
                    issues.append(
                        _strict_issue(
                            "manifest.top_k", "invalid in candidate", left, right
                        )
                    )
                elif left_normalized != right_normalized:
                    issues.append(
                        _strict_issue(
                            "manifest.top_k",
                            "values differ",
                            left_normalized,
                            right_normalized,
                        )
                    )
            else:
                if field == "metric_version":
                    valid_left = (
                        isinstance(left, int)
                        and not isinstance(left, bool)
                        and left > 0
                    )
                    valid_right = (
                        isinstance(right, int)
                        and not isinstance(right, bool)
                        and right > 0
                    )
                    if not valid_left or not valid_right:
                        issues.append(
                            _strict_issue(
                                f"manifest.{field}",
                                "must be a positive integer",
                                left,
                                right,
                            )
                        )
                if left != right:
                    issues.append(
                        _strict_issue(f"manifest.{field}", "values differ", left, right)
                    )

    baseline_query_ids = _query_ids(baseline)
    candidate_query_ids = _query_ids(candidate)
    if baseline_query_ids is None:
        issues.append(
            _strict_issue("manifest.query_ids", "missing or invalid in baseline")
        )
    if candidate_query_ids is None:
        issues.append(
            _strict_issue("manifest.query_ids", "missing or invalid in candidate")
        )
    if (
        baseline_query_ids is not None
        and candidate_query_ids is not None
        and baseline_query_ids != candidate_query_ids
    ):
        issues.append(
            _strict_issue(
                "manifest.query_ids",
                "query ID sets differ",
                list(baseline_query_ids),
                list(candidate_query_ids),
            )
        )

    baseline_names = set(baseline.metrics)
    candidate_names = set(candidate.metrics)
    common = tuple(sorted(baseline_names & candidate_names))
    added = tuple(sorted(candidate_names - baseline_names))
    removed = tuple(sorted(baseline_names - candidate_names))
    if not common:
        issues.append(
            _strict_issue(
                "retrievers",
                "baseline and candidate have no common retriever",
                sorted(baseline_names),
                sorted(candidate_names),
            )
        )

    baseline_shapes = _metric_shapes(baseline, common)
    candidate_shapes = _metric_shapes(candidate, common)
    for retriever in common:
        baseline_query_shapes = baseline_shapes.get(retriever, {})
        candidate_query_shapes = candidate_shapes.get(retriever, {})
        baseline_aggregate_shape = frozenset(
            _aggregate_metric_shape(baseline, retriever)
        )
        candidate_aggregate_shape = frozenset(
            _aggregate_metric_shape(candidate, retriever)
        )
        if baseline_query_ids is not None:
            baseline_result_ids = tuple(sorted(baseline_query_shapes))
            if baseline_result_ids != baseline_query_ids:
                issues.append(
                    _strict_issue(
                        f"query_results[{retriever}].query_ids",
                        "query ID set does not match baseline manifest",
                        list(baseline_result_ids),
                        list(baseline_query_ids),
                    )
                )
        if candidate_query_ids is not None:
            candidate_result_ids = tuple(sorted(candidate_query_shapes))
            if candidate_result_ids != candidate_query_ids:
                issues.append(
                    _strict_issue(
                        f"query_results[{retriever}].query_ids",
                        "query ID set does not match candidate manifest",
                        list(candidate_result_ids),
                        list(candidate_query_ids),
                    )
                )
        for query_id in sorted(
            set(baseline_query_shapes) | set(candidate_query_shapes)
        ):
            left = baseline_query_shapes.get(query_id)
            right = candidate_query_shapes.get(query_id)
            if left is not None and left != baseline_aggregate_shape:
                issues.append(
                    _strict_issue(
                        f"query_results[{retriever}].metric_shape",
                        "query shape does not match baseline aggregate",
                        {query_id: _shape_labels(left)},
                        _shape_labels(baseline_aggregate_shape),
                    )
                )
            if right is not None and right != candidate_aggregate_shape:
                issues.append(
                    _strict_issue(
                        f"query_results[{retriever}].metric_shape",
                        "query shape does not match candidate aggregate",
                        _shape_labels(candidate_aggregate_shape),
                        {query_id: _shape_labels(right)},
                    )
                )
            if left != right:
                issues.append(
                    _strict_issue(
                        f"query_results[{retriever}].metric_shape",
                        "metric names/cutoffs differ for a query",
                        {query_id: _shape_labels(left or ())},
                        {query_id: _shape_labels(right or ())},
                    )
                )

    for retriever in common:
        baseline_shape = _aggregate_metric_shape(baseline, retriever)
        candidate_shape = _aggregate_metric_shape(candidate, retriever)
        if baseline_shape != candidate_shape:
            issues.append(
                _strict_issue(
                    f"retrievers[{retriever!r}].metric_shape",
                    "aggregate metric names/cutoffs differ",
                    [f"{name}@{cutoff}" for name, cutoff in baseline_shape],
                    [f"{name}@{cutoff}" for name, cutoff in candidate_shape],
                )
            )

    diagnostics: list[ComparabilityIssue] = []
    if baseline.latency and not candidate.latency:
        diagnostics.append(
            _strict_issue(
                "latency",
                "latency is missing from candidate",
                True,
                False,
                blocking=False,
            )
        )
    elif candidate.latency and not baseline.latency:
        diagnostics.append(
            _strict_issue(
                "latency",
                "latency is missing from baseline",
                False,
                True,
                blocking=False,
            )
        )
    elif baseline.latency and candidate.latency:
        for retriever in common:
            baseline_missing = sum(
                query.search_latency_ms is None
                for query in baseline.query_results[retriever]
            )
            candidate_missing = sum(
                query.search_latency_ms is None
                for query in candidate.query_results[retriever]
            )
            if baseline_missing or candidate_missing:
                diagnostics.append(
                    _strict_issue(
                        f"latency[{retriever}].per_query",
                        "per-query latency is incomplete; aggregate comparison "
                        "remains available",
                        baseline_missing,
                        candidate_missing,
                        blocking=False,
                    )
                )

    return ComparabilityReport(
        issues=tuple(issues),
        diagnostics=tuple(diagnostics),
        variable_differences=_variable_issues(baseline, candidate),
        common_retrievers=common,
        added_retrievers=added,
        removed_retrievers=removed,
    )


def _direction(metric: str) -> MetricDirection:
    return "lower_is_better" if metric.startswith("latency_") else "higher_is_better"


def _relative(
    baseline: float,
    candidate: float,
    tolerance: ComparisonTolerance,
) -> tuple[float | None, RelativeStatus]:
    if baseline == 0.0:
        if candidate == 0.0:
            return 0.0, "baseline_zero_both_zero"
        return None, "baseline_zero"
    return (candidate - baseline) / abs(baseline), "defined"


def _classification(
    baseline: float,
    candidate: float,
    direction: MetricDirection,
    tolerance: ComparisonTolerance,
) -> DeltaClassification:
    directional = candidate - baseline
    if direction == "lower_is_better":
        directional = -directional
    if tolerance.close(baseline, candidate):
        return "unchanged"
    return "improved" if directional > 0.0 else "regressed"


def _metric_delta(
    *,
    retriever: str,
    metric: str,
    cutoff: int | None,
    baseline: float,
    candidate: float,
    tolerance: ComparisonTolerance,
    query_id: str | None = None,
) -> MetricDelta:
    left = _finite(baseline, f"baseline {metric}")
    right = _finite(candidate, f"candidate {metric}")
    relative_delta, relative_status = _relative(left, right, tolerance)
    return MetricDelta(
        retriever=retriever,
        metric=metric,
        cutoff=cutoff,
        baseline=left,
        candidate=right,
        absolute_delta=right - left,
        relative_delta=relative_delta,
        relative_status=relative_status,
        classification=_classification(left, right, _direction(metric), tolerance),
        direction=_direction(metric),
        query_id=query_id,
    )


def _query_extremes(
    query_deltas: Sequence[MetricDelta],
) -> tuple[QueryDeltaExtreme | None, QueryDeltaExtreme | None, tuple[MetricDelta, ...]]:
    ordered = tuple(
        sorted(
            query_deltas,
            key=lambda value: (
                -(
                    value.absolute_delta
                    if value.direction == "higher_is_better"
                    else -value.absolute_delta
                ),
                cast(str, value.query_id),
            ),
        )
    )
    if not ordered:
        return None, None, ordered
    best_delta = ordered[0]
    worst_delta = min(
        ordered,
        key=lambda value: (
            value.absolute_delta
            if value.direction == "higher_is_better"
            else -value.absolute_delta,
            cast(str, value.query_id),
        ),
    )
    best = QueryDeltaExtreme(
        query_id=cast(str, best_delta.query_id),
        delta=best_delta.absolute_delta,
        directional_delta=(
            best_delta.absolute_delta
            if best_delta.direction == "higher_is_better"
            else -best_delta.absolute_delta
        ),
    )
    worst = QueryDeltaExtreme(
        query_id=cast(str, worst_delta.query_id),
        delta=worst_delta.absolute_delta,
        directional_delta=(
            worst_delta.absolute_delta
            if worst_delta.direction == "higher_is_better"
            else -worst_delta.absolute_delta
        ),
    )
    return best, worst, ordered


def _metric_comparison(
    *,
    retriever: str,
    metric: str,
    cutoff: int | None,
    baseline_value: float,
    candidate_value: float,
    baseline_queries: Mapping[str, QueryEvaluation] | None,
    candidate_queries: Mapping[str, QueryEvaluation] | None,
    tolerance: ComparisonTolerance,
) -> MetricComparison:
    aggregate = _metric_delta(
        retriever=retriever,
        metric=metric,
        cutoff=cutoff,
        baseline=baseline_value,
        candidate=candidate_value,
        tolerance=tolerance,
    )
    query_deltas: list[MetricDelta] = []
    if baseline_queries is not None and candidate_queries is not None:
        common_queries = sorted(set(baseline_queries) & set(candidate_queries))
        for query_id in common_queries:
            left_query = baseline_queries[query_id]
            right_query = candidate_queries[query_id]
            if cutoff is None:
                left_value = left_query.search_latency_ms
                right_value = right_query.search_latency_ms
                if left_value is None or right_value is None:
                    continue
            else:
                if (
                    cutoff not in left_query.metrics_by_cutoff
                    or cutoff not in right_query.metrics_by_cutoff
                    or metric not in left_query.metrics_by_cutoff[cutoff]
                    or metric not in right_query.metrics_by_cutoff[cutoff]
                ):
                    continue
                left_value = left_query.metrics_by_cutoff[cutoff][metric]
                right_value = right_query.metrics_by_cutoff[cutoff][metric]
            query_deltas.append(
                _metric_delta(
                    retriever=retriever,
                    metric=metric,
                    cutoff=cutoff,
                    baseline=left_value,
                    candidate=right_value,
                    tolerance=tolerance,
                    query_id=query_id,
                )
            )
    best, worst, ordered = _query_extremes(query_deltas)
    return MetricComparison(
        retriever=retriever,
        metric=metric,
        cutoff=cutoff,
        aggregate=aggregate,
        query_deltas=ordered,
        best=best,
        worst=worst,
    )


def _aggregate_metric_shape(
    result: EvaluationResult,
    retriever: str,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted(
            (name, cutoff)
            for cutoff, values in result.metrics[retriever].metrics_by_cutoff.items()
            for name in values
        )
    )


def _latency_value(result: EvaluationResult, retriever: str, metric: str) -> float:
    stats = result.latency[retriever]
    field = metric.removeprefix("latency_")
    return _finite(getattr(stats, field), f"{retriever}.{metric}")


def compare_runs(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    *,
    tolerance: ComparisonTolerance = _DEFAULT_TOLERANCE,
) -> RunComparison:
    """Compare two compatible results and return deterministic metric deltas."""
    checked_tolerance = _require_tolerance(tolerance)
    report = check_comparability(baseline, candidate)
    if not report.comparable:
        raise IncomparableRunError(
            f"evaluation runs are not comparable "
            f"({len(report.issues)} blocking issue(s)); "
            "inspect .issues",
            issues=report.issues,
        )

    comparisons: dict[str, list[MetricComparison]] = {}
    for retriever in report.common_retrievers:
        baseline_queries = _result_query_map(baseline, retriever)
        candidate_queries = _result_query_map(candidate, retriever)
        values: list[MetricComparison] = []
        for metric, cutoff in _aggregate_metric_shape(baseline, retriever):
            if (metric, cutoff) not in _aggregate_metric_shape(candidate, retriever):
                continue
            values.append(
                _metric_comparison(
                    retriever=retriever,
                    metric=metric,
                    cutoff=cutoff,
                    baseline_value=baseline.metrics[retriever].metrics_by_cutoff[
                        cutoff
                    ][metric],
                    candidate_value=candidate.metrics[retriever].metrics_by_cutoff[
                        cutoff
                    ][metric],
                    baseline_queries=baseline_queries,
                    candidate_queries=candidate_queries,
                    tolerance=checked_tolerance,
                )
            )
        if baseline.latency and candidate.latency:
            for metric in _LATENCY_METRICS:
                values.append(
                    _metric_comparison(
                        retriever=retriever,
                        metric=metric,
                        cutoff=None,
                        baseline_value=_latency_value(baseline, retriever, metric),
                        candidate_value=_latency_value(candidate, retriever, metric),
                        baseline_queries=baseline_queries,
                        candidate_queries=candidate_queries,
                        tolerance=checked_tolerance,
                    )
                )
        comparisons[retriever] = values

    return RunComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        comparability=report,
        metrics={name: tuple(values) for name, values in comparisons.items()},
    )


__all__ = [
    "ComparabilityIssue",
    "ComparabilityReport",
    "ComparisonTolerance",
    "MetricComparison",
    "MetricDelta",
    "QueryDeltaExtreme",
    "RunComparison",
    "check_comparability",
    "compare_runs",
]
