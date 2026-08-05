"""Pure information-retrieval metric functions."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _validated_retrieved_ids(retrieved_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(retrieved_ids, str):
        raise ValueError("retrieved identifiers must be a sequence, not a string")
    normalized = tuple(retrieved_ids)
    if any(
        not isinstance(identifier, str) or not identifier for identifier in normalized
    ):
        raise ValueError("retrieved identifiers must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("retrieved identifiers must be unique")
    return normalized


def _validated_relevant_ids(relevant_ids: Collection[str]) -> frozenset[str]:
    if isinstance(relevant_ids, str):
        raise ValueError("relevant identifiers must be a collection, not a string")
    normalized = frozenset(relevant_ids)
    if not normalized:
        raise ValueError("relevant identifiers must not be empty")
    if len(normalized) != len(relevant_ids):
        raise ValueError("relevant identifiers must be unique")
    if any(
        not isinstance(identifier, str) or not identifier for identifier in normalized
    ):
        raise ValueError("relevant identifiers must be non-empty strings")
    return normalized


def _binary_inputs(
    retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int
) -> tuple[tuple[str, ...], frozenset[str]]:
    _validate_k(k)
    return (
        _validated_retrieved_ids(retrieved_ids),
        _validated_relevant_ids(relevant_ids),
    )


def hit_rate_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int
) -> float:
    """Return 1.0 when a relevant identifier occurs in the first ``k`` results."""

    retrieved, relevant = _binary_inputs(retrieved_ids, relevant_ids, k)
    return float(any(identifier in relevant for identifier in retrieved[:k]))


def recall_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int
) -> float:
    """Return the fraction of relevant identifiers retrieved in the first ``k``."""

    retrieved, relevant = _binary_inputs(retrieved_ids, relevant_ids, k)
    hits = sum(identifier in relevant for identifier in retrieved[:k])
    return hits / len(relevant)


def precision_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int
) -> float:
    """Return relevant results divided by ``k``; missing positions are misses."""

    retrieved, relevant = _binary_inputs(retrieved_ids, relevant_ids, k)
    hits = sum(identifier in relevant for identifier in retrieved[:k])
    return hits / k


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int | None = None,
) -> float:
    """Return the reciprocal rank of the first relevant result, optionally at ``k``."""

    if k is not None:
        _validate_k(k)
    retrieved = _validated_retrieved_ids(retrieved_ids)
    relevant = _validated_relevant_ids(relevant_ids)
    ranked = retrieved if k is None else retrieved[:k]
    for rank, identifier in enumerate(ranked, start=1):
        if identifier in relevant:
            return 1.0 / rank
    return 0.0


def _validated_relevance(
    relevance_by_id: Mapping[str, int | float],
) -> dict[str, float]:
    if not relevance_by_id:
        raise ValueError("relevance mapping must not be empty")

    normalized: dict[str, float] = {}
    for identifier, relevance in relevance_by_id.items():
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("relevance identifiers must be non-empty strings")
        if isinstance(relevance, bool) or not isinstance(relevance, int | float):
            raise ValueError("relevance values must be finite non-negative numbers")
        try:
            numeric_relevance = float(relevance)
        except OverflowError as error:
            raise ValueError(
                "relevance values must be finite non-negative numbers"
            ) from error
        if not math.isfinite(numeric_relevance) or numeric_relevance < 0:
            raise ValueError("relevance values must be finite non-negative numbers")
        normalized[identifier] = numeric_relevance

    if not any(relevance > 0 for relevance in normalized.values()):
        raise ValueError("relevance mapping must contain a positive value")
    return normalized


def _gain(relevance: float) -> float:
    try:
        gain = math.pow(2.0, relevance) - 1.0
    except OverflowError as error:
        raise ValueError("relevance values are too large to calculate nDCG") from error
    return gain


def _discounted_cumulative_gain(relevances: Sequence[float]) -> float:
    return sum(
        _gain(relevance) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_by_id: Mapping[str, int | float],
    k: int,
) -> float:
    """Return graded nDCG@k using gain ``2**relevance - 1``."""

    _validate_k(k)
    retrieved = _validated_retrieved_ids(retrieved_ids)
    relevance = _validated_relevance(relevance_by_id)
    observed = [relevance.get(identifier, 0.0) for identifier in retrieved[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = _discounted_cumulative_gain(ideal)
    return _discounted_cumulative_gain(observed) / ideal_dcg


def average_precision_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int
) -> float:
    """Return AP@k with denominator ``min(number of relevant items, k)``."""

    retrieved, relevant = _binary_inputs(retrieved_ids, relevant_ids, k)
    hits = 0
    precision_sum = 0.0
    for rank, identifier in enumerate(retrieved[:k], start=1):
        if identifier in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant), k)
