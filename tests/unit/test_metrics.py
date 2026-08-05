from __future__ import annotations

import math

import pytest

from retrieval_lab.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ("d3", "d1", "d2", "d4")
RELEVANT = frozenset({"d1", "d2"})


def test_binary_metrics_match_hand_calculated_example() -> None:
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 1) == 0.0
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 3) == 1.0
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == 1.0
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(2 / 3)
    assert reciprocal_rank(RETRIEVED, RELEVANT) == 0.5
    assert reciprocal_rank(RETRIEVED, RELEVANT, 1) == 0.0
    assert average_precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(7 / 12)


def test_precision_counts_missing_positions_as_non_relevant() -> None:
    assert precision_at_k(("d1",), frozenset({"d1"}), 3) == pytest.approx(1 / 3)


def test_average_precision_uses_minimum_of_relevant_count_and_k() -> None:
    retrieved = ("d1", "d2", "x")
    relevant = frozenset({"d1", "d2", "d3", "d4"})
    assert average_precision_at_k(retrieved, relevant, 2) == 1.0


def test_ndcg_matches_hand_calculated_graded_example() -> None:
    relevance = {"d1": 3, "d2": 1}
    observed_dcg = 7 / math.log2(3) + 1 / math.log2(4)
    ideal_dcg = 7 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RETRIEVED, relevance, 3) == pytest.approx(observed_dcg / ideal_dcg)


def test_ndcg_is_one_for_ideal_ranking_and_zero_for_no_hits() -> None:
    relevance = {"high": 3, "low": 1}
    assert ndcg_at_k(("high", "low"), relevance, 2) == 1.0
    assert ndcg_at_k(("x", "y"), relevance, 2) == 0.0


@pytest.mark.parametrize(
    ("retrieved", "relevance", "k"),
    [
        ((), {"a": 1}, 1),
        (("a",), {"a": 1}, 5),
        (("x", "low", "high"), {"high": 3, "low": 1}, 3),
        (("medium", "x", "high"), {"high": 3, "medium": 2}, 2),
    ],
)
def test_ndcg_remains_between_zero_and_one_for_multiple_rankings(
    retrieved: tuple[str, ...], relevance: dict[str, int], k: int
) -> None:
    assert 0.0 <= ndcg_at_k(retrieved, relevance, k) <= 1.0


@pytest.mark.parametrize(
    "retrieved,relevant",
    [
        ((), frozenset({"a"})),
        (("x",), frozenset({"a"})),
        (("a",), frozenset({"a"})),
        (("x", "a", "b"), frozenset({"a", "b"})),
    ],
)
def test_normalized_binary_metrics_remain_between_zero_and_one(
    retrieved: tuple[str, ...], relevant: frozenset[str]
) -> None:
    values = (
        hit_rate_at_k(retrieved, relevant, 3),
        recall_at_k(retrieved, relevant, 3),
        precision_at_k(retrieved, relevant, 3),
        reciprocal_rank(retrieved, relevant, 3),
        average_precision_at_k(retrieved, relevant, 3),
    )
    assert all(0.0 <= value <= 1.0 for value in values)


def test_recall_is_monotonic_as_cutoff_increases() -> None:
    retrieved = ("x", "a", "y", "b")
    relevant = frozenset({"a", "b"})
    recalls = [recall_at_k(retrieved, relevant, k) for k in range(1, 5)]
    hit_rates = [hit_rate_at_k(retrieved, relevant, k) for k in range(1, 5)]
    assert recalls == sorted(recalls)
    assert hit_rates == sorted(hit_rates)


def test_metrics_reject_a_single_string_instead_of_an_identifier_sequence() -> None:
    with pytest.raises(ValueError, match="sequence, not a string"):
        recall_at_k("abc", {"abc"}, 1)
    with pytest.raises(ValueError, match="collection, not a string"):
        recall_at_k(("abc",), "abc", 1)


@pytest.mark.parametrize("identifier", ["", 1])
def test_metrics_reject_invalid_retrieved_identifiers(identifier: object) -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        recall_at_k((identifier,), {"a"}, 1)  # type: ignore[arg-type]


def test_metrics_reject_duplicate_or_invalid_relevant_identifiers() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        recall_at_k(("a",), ["a", "a"], 1)
    with pytest.raises(ValueError, match="non-empty strings"):
        recall_at_k(("a",), {""}, 1)


@pytest.mark.parametrize("metric", [hit_rate_at_k, recall_at_k, precision_at_k])
@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_binary_metric_rejects_invalid_cutoff(metric: object, k: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        metric(("a",), {"a"}, k)  # type: ignore[operator]


@pytest.mark.parametrize(
    "metric",
    [
        hit_rate_at_k,
        recall_at_k,
        precision_at_k,
        reciprocal_rank,
        average_precision_at_k,
    ],
)
def test_binary_metric_rejects_empty_relevance(metric: object) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        metric(("a",), set(), 1)  # type: ignore[operator]


@pytest.mark.parametrize(
    "metric",
    [
        hit_rate_at_k,
        recall_at_k,
        precision_at_k,
        reciprocal_rank,
        average_precision_at_k,
    ],
)
def test_binary_metric_rejects_duplicate_retrieved_identifiers(metric: object) -> None:
    with pytest.raises(ValueError, match="must be unique"):
        metric(("a", "a"), {"a"}, 2)  # type: ignore[operator]


def test_reciprocal_rank_rejects_invalid_optional_cutoff() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        reciprocal_rank(("a",), {"a"}, 0)


@pytest.mark.parametrize(
    "relevance",
    [
        {},
        {"a": 0},
        {"a": -1},
        {"a": math.nan},
        {"a": math.inf},
        {"a": True},
        {"a": 10**10_000},
    ],
)
def test_ndcg_rejects_invalid_relevance(relevance: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ndcg_at_k(("a",), relevance, 1)  # type: ignore[arg-type]


def test_ndcg_rejects_invalid_identifier_cutoff_and_excessive_gain() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        ndcg_at_k(("a",), {"": 1}, 1)
    with pytest.raises(ValueError, match="positive integer"):
        ndcg_at_k(("a",), {"a": 1}, 0)
    with pytest.raises(ValueError, match="too large"):
        ndcg_at_k(("a",), {"a": 1024}, 1)
