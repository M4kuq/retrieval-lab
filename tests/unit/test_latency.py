from __future__ import annotations

import math

import pytest

import retrieval_lab.evaluation as evaluation_module
from retrieval_lab import LatencyStats
from retrieval_lab.evaluation import RetrievedQueryResult, evaluate_results
from retrieval_lab.evaluation.latency import nearest_rank_percentile
from retrieval_lab.exceptions import EvaluationError


def test_nearest_rank_percentile_uses_hand_calculated_ranks() -> None:
    values = [4.0, 1.0, 3.0, 2.0]

    assert nearest_rank_percentile(values, 50) == 2.0
    assert nearest_rank_percentile(values, 95) == 4.0


@pytest.mark.parametrize(
    ("values", "percentile"),
    [([], 50), ([math.nan], 50), ([math.inf], 50), ([-1], 50), ([1], -1), ([1], 101)],
)
def test_nearest_rank_percentile_rejects_invalid_inputs(
    values: list[float], percentile: float
) -> None:
    with pytest.raises(EvaluationError):
        nearest_rank_percentile(values, percentile)


def test_nearest_rank_percentile_rejects_string_values() -> None:
    with pytest.raises(EvaluationError):
        nearest_rank_percentile("1", 50)  # type: ignore[arg-type]


def test_latency_stats_warns_only_for_small_samples() -> None:
    small = LatencyStats.from_samples([1.0, 2.0])
    large = LatencyStats.from_samples([float(value) for value in range(20)])

    assert small.mean_ms == 1.5
    assert small.p50_ms == 1.0
    assert small.p95_ms == 2.0
    assert small.max_ms == 2.0
    assert small.sample_count == 2
    assert small.failure_count == 0
    assert small.warnings
    assert large.warnings == ()


def test_latency_stats_supports_empty_samples_and_serialization() -> None:
    stats = LatencyStats.from_samples([], failure_count=1)

    assert stats.sample_count == 0
    assert stats.failure_count == 1
    assert stats.to_dict()["warnings"]
    assert RetrievedQueryResult.__name__ == "RetrievedQueryResult"
    assert evaluate_results.__name__ == "evaluate_results"
    with pytest.raises(AttributeError):
        evaluation_module.__getattr__("not_an_evaluation_api")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mean_ms": -1.0},
        {"sample_count": -1},
        {"failure_count": True},
        {"warnings": "warning"},
        {"warnings": [""]},
    ],
)
def test_latency_stats_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "mean_ms": 1.0,
        "p50_ms": 1.0,
        "p95_ms": 1.0,
        "max_ms": 1.0,
        "sample_count": 1,
        "failure_count": 0,
        "warnings": (),
    }
    values.update(kwargs)
    with pytest.raises(EvaluationError):
        LatencyStats(**values)  # type: ignore[arg-type]
