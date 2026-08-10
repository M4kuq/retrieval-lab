"""Pure latency aggregation helpers.

Latency measurements are kept separate from ranking metrics so that timing and
clock implementation details cannot affect a reproducible run identifier.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from retrieval_lab.exceptions import EvaluationError

P95_WARNING = "p95 may be unstable with fewer than 20 samples"


def nearest_rank_percentile(
    values: Sequence[float],
    percentile: float,
) -> float:
    """Return a percentile using the one-based nearest-rank definition.

    ``values`` must contain at least one finite, non-negative number.  The
    percentile is expressed as a percentage in the inclusive range ``0..100``;
    a zero percentile selects the first sorted sample, as prescribed by the
    nearest-rank method.
    """

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise EvaluationError("latency values must be a sequence")
    if not values:
        raise EvaluationError("latency values must not be empty")
    samples = tuple(
        _require_non_negative_float(value, "latency value") for value in values
    )
    normalized_percentile = _require_non_negative_float(percentile, "percentile")
    if normalized_percentile > 100.0:
        raise EvaluationError("percentile must be between 0 and 100")

    rank = max(1, math.ceil(normalized_percentile / 100.0 * len(samples)))
    return float(sorted(samples)[rank - 1])


@dataclass(frozen=True)
class LatencyStats:
    """Immutable aggregate search-latency statistics in milliseconds."""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    sample_count: int
    failure_count: int = 0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("mean_ms", "p50_ms", "p95_ms", "max_ms"):
            object.__setattr__(
                self,
                field_name,
                _require_non_negative_float(
                    getattr(self, field_name), f"LatencyStats.{field_name}"
                ),
            )
        for field_name in ("sample_count", "failure_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EvaluationError(
                    f"LatencyStats.{field_name} must be a non-negative integer"
                )
        if isinstance(self.warnings, (str, bytes)) or not isinstance(
            self.warnings, Sequence
        ):
            raise EvaluationError("LatencyStats.warnings must be a sequence")
        warnings = tuple(self.warnings)
        if any(
            not isinstance(warning, str) or not warning.strip() for warning in warnings
        ):
            raise EvaluationError(
                "LatencyStats.warnings must contain non-empty strings"
            )
        object.__setattr__(self, "warnings", warnings)

    @classmethod
    def from_samples(
        cls,
        values: Sequence[float],
        *,
        failure_count: int = 0,
    ) -> LatencyStats:
        """Aggregate successful samples and optional failed-call count."""

        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise EvaluationError("latency values must be a sequence")
        if (
            isinstance(failure_count, bool)
            or not isinstance(failure_count, int)
            or failure_count < 0
        ):
            raise EvaluationError("failure_count must be a non-negative integer")
        samples = tuple(
            _require_non_negative_float(value, "latency value") for value in values
        )
        if not samples:
            return cls(
                mean_ms=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                max_ms=0.0,
                sample_count=0,
                failure_count=failure_count,
                warnings=(P95_WARNING,),
            )
        warning_values = (P95_WARNING,) if len(samples) < 20 else ()
        return cls(
            mean_ms=float(sum(samples) / len(samples)),
            p50_ms=nearest_rank_percentile(samples, 50.0),
            p95_ms=nearest_rank_percentile(samples, 95.0),
            max_ms=float(max(samples)),
            sample_count=len(samples),
            failure_count=failure_count,
            warnings=warning_values,
        )

    def to_dict(self) -> dict[str, float | int | list[str]]:
        """Return the stable JSON field names used in result reports."""

        return {
            "failure_count": self.failure_count,
            "max_ms": self.max_ms,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "sample_count": self.sample_count,
            "warnings": list(self.warnings),
        }


def _require_non_negative_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{field_name} must be a finite non-negative number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise EvaluationError(
            f"{field_name} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(normalized) or normalized < 0.0:
        raise EvaluationError(f"{field_name} must be a finite non-negative number")
    return normalized


__all__ = ["P95_WARNING", "LatencyStats", "nearest_rank_percentile"]
