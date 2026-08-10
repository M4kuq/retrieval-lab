"""Quality-gate evaluation over typed evaluation results."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

from retrieval_lab.comparison import (
    ComparisonTolerance,
    MetricDelta,
    RunComparison,
    compare_runs,
)
from retrieval_lab.config.models import (
    SUPPORTED_LATENCY_METRICS,
    QualityGateConfig,
)
from retrieval_lab.domain import (
    ConstraintType,
    EvaluationResult,
    QualityGateCheck,
    QualityGateReport,
    QualityGateResult,
)
from retrieval_lab.exceptions import ConfigurationError, EvaluationError

_DEFAULT_TOLERANCE = ComparisonTolerance()


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise EvaluationError(f"{field_name} must be a finite number")
    return normalized


def _require_tolerance(value: object) -> ComparisonTolerance:
    if not isinstance(value, ComparisonTolerance):
        raise ConfigurationError("tolerance must be a ComparisonTolerance")
    return value


def _metric_reference(metric: str) -> tuple[str, int | None]:
    if metric in SUPPORTED_LATENCY_METRICS:
        return metric, None
    name, raw_cutoff = metric.split("@")
    return name, int(raw_cutoff)


def _candidate_value(
    candidate: EvaluationResult,
    retriever: str,
    metric: str,
    cutoff: int | None,
) -> float:
    if retriever not in candidate.metrics:
        raise ConfigurationError("quality gate references an unavailable retriever")
    if cutoff is None:
        if not candidate.latency or retriever not in candidate.latency:
            raise ConfigurationError("quality gate references an unavailable metric")
        stats = candidate.latency[retriever]
        field = metric.removeprefix("latency_")
        return _finite(getattr(stats, field), "quality gate candidate metric")
    values = candidate.metrics[retriever].metrics_by_cutoff.get(cutoff)
    if values is None or metric not in values:
        raise ConfigurationError("quality gate references an unavailable metric")
    return _finite(values[metric], "quality gate candidate metric")


def _comparison_delta(
    comparison: RunComparison,
    retriever: str,
    metric: str,
    cutoff: int | None,
) -> MetricDelta:
    metrics = comparison.metrics
    if retriever not in metrics:
        raise ConfigurationError("quality gate references an unavailable retriever")
    for item in metrics[retriever]:
        if item.metric == metric and item.cutoff == cutoff:
            return item.aggregate
    raise ConfigurationError("quality gate references an unavailable metric")


def _boundary_passes(
    actual: float,
    threshold: float,
    *,
    minimum: bool,
    tolerance: ComparisonTolerance,
) -> bool:
    if tolerance.close(actual, threshold):
        return True
    return actual >= threshold if minimum else actual <= threshold


def _absolute_regression(
    delta: MetricDelta,
    tolerance: ComparisonTolerance,
) -> float:
    if tolerance.close(delta.baseline, delta.candidate):
        return 0.0
    if delta.direction == "higher_is_better":
        return max(delta.baseline - delta.candidate, 0.0)
    return max(delta.candidate - delta.baseline, 0.0)


def _relative_regression(
    delta: MetricDelta,
) -> float | None:
    if delta.relative_status == "baseline_zero_both_zero":
        return 0.0
    if delta.relative_status == "baseline_zero":
        if delta.classification == "regressed":
            return None
        return 0.0
    if delta.relative_delta is None:
        return None
    if delta.classification != "regressed":
        return 0.0
    if delta.direction == "higher_is_better":
        return max(-delta.relative_delta, 0.0)
    return max(delta.relative_delta, 0.0)


def _check_absolute_constraint(
    *,
    gate: QualityGateConfig,
    constraint: str,
    actual: float,
    threshold: float,
    tolerance: ComparisonTolerance,
    candidate: EvaluationResult,
    baseline: EvaluationResult | None,
) -> QualityGateCheck:
    minimum = constraint == "min_value"
    passed = _boundary_passes(actual, threshold, minimum=minimum, tolerance=tolerance)
    reason = (
        "candidate meets minimum"
        if minimum and passed
        else "candidate is below minimum"
        if minimum
        else "candidate meets maximum"
        if passed
        else "candidate exceeds maximum"
    )
    return QualityGateCheck(
        retriever=gate.retriever,
        metric=gate.metric,
        constraint=cast(ConstraintType, constraint),
        actual=actual,
        threshold=threshold,
        passed=passed,
        reason=reason,
        candidate_run_id=candidate.run_id,
        baseline_run_id=baseline.run_id if baseline is not None else None,
        absolute_tolerance=tolerance.absolute,
        relative_tolerance=tolerance.relative,
    )


def _check_drop_constraint(
    *,
    gate: QualityGateConfig,
    constraint: str,
    delta: MetricDelta,
    threshold: float,
    tolerance: ComparisonTolerance,
    candidate: EvaluationResult,
    baseline: EvaluationResult,
) -> QualityGateCheck:
    actual = (
        _absolute_regression(delta, tolerance)
        if constraint == "max_absolute_drop"
        else _relative_regression(delta)
    )
    if actual is None:
        return QualityGateCheck(
            retriever=gate.retriever,
            metric=gate.metric,
            constraint=cast(ConstraintType, constraint),
            actual=None,
            threshold=threshold,
            passed=False,
            reason="relative regression is undefined because baseline is zero",
            candidate_run_id=candidate.run_id,
            baseline_run_id=baseline.run_id,
            absolute_tolerance=tolerance.absolute,
            relative_tolerance=tolerance.relative,
            status="undefined_baseline_zero_regression",
        )
    passed = _boundary_passes(actual, threshold, minimum=False, tolerance=tolerance)
    reason = (
        "regression is within threshold" if passed else "regression exceeds threshold"
    )
    return QualityGateCheck(
        retriever=gate.retriever,
        metric=gate.metric,
        constraint=cast(ConstraintType, constraint),
        actual=actual,
        threshold=threshold,
        passed=passed,
        reason=reason,
        candidate_run_id=candidate.run_id,
        baseline_run_id=baseline.run_id,
        absolute_tolerance=tolerance.absolute,
        relative_tolerance=tolerance.relative,
    )


def evaluate_quality_gates(
    candidate: EvaluationResult,
    gates: Sequence[QualityGateConfig],
    *,
    baseline: EvaluationResult | None = None,
    tolerance: ComparisonTolerance = _DEFAULT_TOLERANCE,
) -> QualityGateReport:
    """Evaluate all configured absolute and baseline-relative constraints."""

    if not isinstance(candidate, EvaluationResult):
        raise EvaluationError("candidate must be an EvaluationResult")
    if isinstance(gates, (str, bytes)) or not isinstance(gates, Sequence):
        raise ConfigurationError("quality gates must be a sequence")
    if baseline is not None and not isinstance(baseline, EvaluationResult):
        raise EvaluationError("baseline must be an EvaluationResult")
    checked_tolerance = _require_tolerance(tolerance)
    configs = tuple(gates)
    if not all(isinstance(gate, QualityGateConfig) for gate in configs):
        raise ConfigurationError("quality gates must contain QualityGateConfig")
    has_drop = any(
        gate.max_absolute_drop is not None or gate.max_relative_drop is not None
        for gate in configs
    )
    if has_drop and baseline is None:
        raise ConfigurationError("drop quality gates require a baseline")

    comparison = None
    if baseline is not None:
        comparison = compare_runs(baseline, candidate, tolerance=checked_tolerance)

    results = []
    for gate_index, gate in enumerate(configs):
        metric, cutoff = _metric_reference(gate.metric)
        candidate_actual = _candidate_value(candidate, gate.retriever, metric, cutoff)
        delta = (
            None
            if comparison is None
            else _comparison_delta(comparison, gate.retriever, metric, cutoff)
        )
        checks: list[QualityGateCheck] = []
        if gate.min_value is not None:
            checks.append(
                _check_absolute_constraint(
                    gate=gate,
                    constraint="min_value",
                    actual=candidate_actual,
                    threshold=gate.min_value,
                    tolerance=checked_tolerance,
                    candidate=candidate,
                    baseline=baseline,
                )
            )
        if gate.max_value is not None:
            checks.append(
                _check_absolute_constraint(
                    gate=gate,
                    constraint="max_value",
                    actual=candidate_actual,
                    threshold=gate.max_value,
                    tolerance=checked_tolerance,
                    candidate=candidate,
                    baseline=baseline,
                )
            )
        if gate.max_absolute_drop is not None:
            if delta is None or baseline is None:
                raise ConfigurationError("drop quality gates require a baseline")
            checks.append(
                _check_drop_constraint(
                    gate=gate,
                    constraint="max_absolute_drop",
                    delta=delta,
                    threshold=gate.max_absolute_drop,
                    tolerance=checked_tolerance,
                    candidate=candidate,
                    baseline=baseline,
                )
            )
        if gate.max_relative_drop is not None:
            if delta is None or baseline is None:
                raise ConfigurationError("drop quality gates require a baseline")
            checks.append(
                _check_drop_constraint(
                    gate=gate,
                    constraint="max_relative_drop",
                    delta=delta,
                    threshold=gate.max_relative_drop,
                    tolerance=checked_tolerance,
                    candidate=candidate,
                    baseline=baseline,
                )
            )
        results.append(
            QualityGateResult(
                retriever=gate.retriever,
                metric=gate.metric,
                checks=tuple(checks),
                passed=all(check.passed for check in checks),
                candidate_run_id=candidate.run_id,
                baseline_run_id=baseline.run_id if baseline is not None else None,
                gate_index=gate_index,
            )
        )
    return QualityGateReport(
        candidate_run_id=candidate.run_id,
        baseline_run_id=baseline.run_id if baseline is not None else None,
        results=tuple(results),
    )


__all__ = ["evaluate_quality_gates"]
