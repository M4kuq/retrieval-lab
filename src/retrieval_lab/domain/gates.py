"""Typed, serializable quality-gate result records."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal, cast

from retrieval_lab.exceptions import EvaluationError

from .json_types import JSONValue

ConstraintType = Literal[
    "min_value",
    "max_value",
    "max_absolute_drop",
    "max_relative_drop",
]

_CONSTRAINTS: tuple[ConstraintType, ...] = (
    "min_value",
    "max_value",
    "max_absolute_drop",
    "max_relative_drop",
)
_UNDEFINED_RELATIVE_STATUS = "undefined_baseline_zero_regression"
_REASONS = {
    "min_value": ("candidate meets minimum", "candidate is below minimum"),
    "max_value": ("candidate meets maximum", "candidate exceeds maximum"),
    "max_absolute_drop": (
        "regression is within threshold",
        "regression exceeds threshold",
    ),
    "max_relative_drop": (
        "regression is within threshold",
        "regression exceeds threshold",
    ),
}


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise EvaluationError(f"{field_name} must be a finite number")
    return normalized


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class QualityGateCheck:
    """One evaluated constraint for one retriever metric."""

    retriever: str
    metric: str
    constraint: ConstraintType
    actual: float | None
    threshold: float
    passed: bool
    reason: str
    candidate_run_id: str
    baseline_run_id: str | None = None
    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-9
    status: Literal["defined", "undefined_baseline_zero_regression"] = "defined"

    def __post_init__(self) -> None:
        _string(self.retriever, "QualityGateCheck.retriever")
        _string(self.metric, "QualityGateCheck.metric")
        if self.constraint not in _CONSTRAINTS:
            raise EvaluationError("QualityGateCheck.constraint is invalid")
        if self.actual is None and self.constraint != "max_relative_drop":
            raise EvaluationError(
                "QualityGateCheck.actual may be absent only for relative drop"
            )
        if self.actual is not None:
            object.__setattr__(
                self, "actual", _finite(self.actual, "QualityGateCheck.actual")
            )
        object.__setattr__(
            self, "threshold", _finite(self.threshold, "QualityGateCheck.threshold")
        )
        if (
            self.constraint in ("max_absolute_drop", "max_relative_drop")
            and self.threshold < 0.0
        ):
            raise EvaluationError(
                "QualityGateCheck drop threshold must be non-negative"
            )
        for field_name in ("absolute_tolerance", "relative_tolerance"):
            tolerance = _finite(getattr(self, field_name), field_name)
            if tolerance < 0.0:
                raise EvaluationError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, tolerance)
        if self.status not in ("defined", _UNDEFINED_RELATIVE_STATUS):
            raise EvaluationError("QualityGateCheck.status is invalid")
        if self.actual is None:
            if (
                self.constraint != "max_relative_drop"
                or self.status != _UNDEFINED_RELATIVE_STATUS
                or self.passed
                or self.reason
                != "relative regression is undefined because baseline is zero"
            ):
                raise EvaluationError(
                    "QualityGateCheck undefined status is inconsistent"
                )
        else:
            if self.status != "defined":
                raise EvaluationError("QualityGateCheck.status is inconsistent")
            close = math.isclose(
                self.actual,
                self.threshold,
                rel_tol=self.relative_tolerance,
                abs_tol=self.absolute_tolerance,
            )
            if self.constraint == "min_value":
                expected = close or self.actual >= self.threshold
            else:
                expected = close or self.actual <= self.threshold
            if expected != self.passed:
                raise EvaluationError("QualityGateCheck.passed is inconsistent")
            expected_reason = _REASONS[self.constraint][0 if self.passed else 1]
            if self.reason != expected_reason:
                raise EvaluationError("QualityGateCheck.reason is inconsistent")
        if not isinstance(self.passed, bool):
            raise EvaluationError("QualityGateCheck.passed must be boolean")
        _string(self.reason, "QualityGateCheck.reason")
        _string(self.candidate_run_id, "QualityGateCheck.candidate_run_id")
        if self.baseline_run_id is not None:
            _string(self.baseline_run_id, "QualityGateCheck.baseline_run_id")

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the canonical JSON-compatible check representation."""

        return {
            "actual": self.actual,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "absolute_tolerance": self.absolute_tolerance,
            "constraint": self.constraint,
            "metric": self.metric,
            "passed": self.passed,
            "reason": self.reason,
            "retriever": self.retriever,
            "relative_tolerance": self.relative_tolerance,
            "status": self.status,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class QualityGateResult:
    """All constraints evaluated for one configured gate."""

    retriever: str
    metric: str
    checks: tuple[QualityGateCheck, ...]
    passed: bool
    candidate_run_id: str
    baseline_run_id: str | None = None
    gate_index: int = 0

    def __post_init__(self) -> None:
        _string(self.retriever, "QualityGateResult.retriever")
        _string(self.metric, "QualityGateResult.metric")
        checks = tuple(self.checks)
        if not checks:
            raise EvaluationError("QualityGateResult.checks must not be empty")
        if not all(isinstance(check, QualityGateCheck) for check in checks):
            raise EvaluationError("QualityGateResult.checks must contain checks")
        if len({check.constraint for check in checks}) != len(checks):
            raise EvaluationError(
                "QualityGateResult.checks must not contain duplicates"
            )
        for check in checks:
            if (check.retriever, check.metric) != (self.retriever, self.metric):
                raise EvaluationError("QualityGateResult.check identity differs")
            if check.candidate_run_id != self.candidate_run_id:
                raise EvaluationError("QualityGateResult candidate run ID differs")
            if check.baseline_run_id != self.baseline_run_id:
                raise EvaluationError("QualityGateResult baseline run ID differs")
            if (
                check.constraint in ("max_absolute_drop", "max_relative_drop")
                and check.baseline_run_id is None
            ):
                raise EvaluationError(
                    "QualityGateResult drop checks require a baseline run ID"
                )
        if not isinstance(self.passed, bool):
            raise EvaluationError("QualityGateResult.passed must be boolean")
        if (
            isinstance(self.gate_index, bool)
            or not isinstance(self.gate_index, int)
            or self.gate_index < 0
        ):
            raise EvaluationError("QualityGateResult.gate_index must be non-negative")
        if self.passed != all(check.passed for check in checks):
            raise EvaluationError("QualityGateResult.passed is inconsistent")
        _string(self.candidate_run_id, "QualityGateResult.candidate_run_id")
        if self.baseline_run_id is not None:
            _string(self.baseline_run_id, "QualityGateResult.baseline_run_id")
        order = {constraint: index for index, constraint in enumerate(_CONSTRAINTS)}
        object.__setattr__(
            self,
            "checks",
            tuple(sorted(checks, key=lambda check: order[check.constraint])),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return the canonical JSON-compatible gate representation."""

        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "checks": cast(JSONValue, [check.to_dict() for check in self.checks]),
            "gate_index": self.gate_index,
            "metric": self.metric,
            "passed": self.passed,
            "retriever": self.retriever,
        }


@dataclass(frozen=True)
class QualityGateReport:
    """Deterministic results for every configured quality gate."""

    candidate_run_id: str
    baseline_run_id: str | None
    results: tuple[QualityGateResult, ...]

    def __post_init__(self) -> None:
        _string(self.candidate_run_id, "QualityGateReport.candidate_run_id")
        if self.baseline_run_id is not None:
            _string(self.baseline_run_id, "QualityGateReport.baseline_run_id")
        results = tuple(self.results)
        if not all(isinstance(result, QualityGateResult) for result in results):
            raise EvaluationError("QualityGateReport.results must contain results")
        if len({result.gate_index for result in results}) != len(results):
            raise EvaluationError(
                "QualityGateReport.results must not contain duplicate gate indexes"
            )
        for result in results:
            if result.candidate_run_id != self.candidate_run_id:
                raise EvaluationError("QualityGateReport candidate run ID differs")
            if result.baseline_run_id != self.baseline_run_id:
                raise EvaluationError("QualityGateReport baseline run ID differs")
        ordered = tuple(sorted(results, key=lambda item: item.gate_index))
        if tuple(result.gate_index for result in ordered) != tuple(range(len(ordered))):
            raise EvaluationError(
                "QualityGateReport gate indexes must be contiguous from zero"
            )
        object.__setattr__(self, "results", ordered)

    @property
    def passed(self) -> bool:
        """Whether every gate passed."""

        return all(result.passed for result in self.results)

    @property
    def failed(self) -> tuple[QualityGateResult, ...]:
        """Return failed gates in deterministic configured order."""

        return tuple(result for result in self.results if not result.passed)

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-compatible report wrapper."""

        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "passed": self.passed,
            "quality_gates": cast(
                JSONValue, [result.to_dict() for result in self.results]
            ),
        }

    def to_json(self) -> str:
        """Return deterministic UTF-8 JSON with a final newline."""

        try:
            payload = json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            return f"{payload}\n"
        except (TypeError, ValueError) as exc:
            raise EvaluationError(
                "quality gate report could not be serialized"
            ) from exc


__all__ = [
    "ConstraintType",
    "QualityGateCheck",
    "QualityGateReport",
    "QualityGateResult",
]
