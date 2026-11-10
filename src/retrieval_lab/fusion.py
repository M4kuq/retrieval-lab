"""Reciprocal-rank fusion of precomputed rankings as an evaluation target."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from retrieval_lab.datasets import EvaluationDataset
from retrieval_lab.domain import EvaluationResult
from retrieval_lab.evaluation.precomputed import RetrievedQueryResult, evaluate_results
from retrieval_lab.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    RetrieverContractError,
)


def fuse_rankings(
    rankings: Mapping[str, Sequence[str]],
    *,
    rrf_k: int = 60,
) -> tuple[str, ...]:
    """Fuse named identifier rankings with reciprocal-rank fusion.

    Each identifier scores the sum of ``1 / (rrf_k + rank)`` over the rankings
    containing it, using one-based ranks. Sources accumulate in sorted-name
    order and ties break by identifier, matching ``HybridRetriever``.
    """

    normalized_k = _validate_rrf_k(rrf_k)
    normalized = _validate_rankings(rankings)
    scores: dict[str, float] = {}
    for name in sorted(normalized):
        for rank, identifier in enumerate(normalized[name], start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (
                normalized_k + rank
            )
    return tuple(
        sorted(scores, key=lambda identifier: (-scores[identifier], identifier))
    )


def fuse_results(
    results: Mapping[str, Sequence[RetrievedQueryResult]],
    *,
    rrf_k: int = 60,
) -> tuple[RetrievedQueryResult, ...]:
    """Fuse per-query rankings from two or more named sources.

    Every source must provide exactly one ranking for the same set of query
    IDs. The fused rankings are returned in ascending query ID order.
    """

    normalized_k = _validate_rrf_k(rrf_k)
    by_query = _validate_results(results)
    return tuple(
        RetrievedQueryResult(
            query_id=query_id,
            retrieved_document_ids=fuse_rankings(
                {
                    name: rankings[query_id]
                    for name, rankings in sorted(by_query.items())
                },
                rrf_k=normalized_k,
            ),
        )
        for query_id in sorted(by_query[next(iter(by_query))])
    )


def evaluate_fused_results(
    *,
    dataset: EvaluationDataset,
    results: Mapping[str, Sequence[RetrievedQueryResult]],
    rrf_k: int = 60,
    top_k: Sequence[int] = (1, 3, 5, 10),
    name: str = "hybrid",
) -> EvaluationResult:
    """Evaluate the RRF fusion of precomputed rankings as its own run.

    The fused rankings are evaluated with the same metrics, dataset hash, and
    manifest contract as ``evaluate_results``, so the returned run is a
    first-class ``EvaluationResult``: it can be saved, reported, gated, and
    diffed with ``compare_runs`` against other fused runs, for example across
    ``rrf_k`` values or source sets.
    """

    return evaluate_results(
        dataset=dataset,
        retrieved_results=fuse_results(results, rrf_k=rrf_k),
        top_k=top_k,
        name=name,
    )


def _validate_rrf_k(rrf_k: int) -> int:
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ConfigurationError("rrf_k must be a positive integer")
    return rrf_k


def _validate_rankings(
    rankings: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if not isinstance(rankings, Mapping):
        raise DatasetValidationError("rankings must be a mapping of source rankings")
    if len(rankings) < 2:
        raise DatasetValidationError("fusion requires at least two sources")
    normalized: dict[str, tuple[str, ...]] = {}
    for name, ranking in rankings.items():
        if not isinstance(name, str) or not name.strip():
            raise DatasetValidationError("rankings source names must be non-empty")
        if isinstance(ranking, (str, bytes)) or not isinstance(ranking, Sequence):
            raise RetrieverContractError(
                f"rankings[{name!r}] must be a sequence of identifiers, not a string"
            )
        identifiers = tuple(ranking)
        if any(
            not isinstance(identifier, str) or not identifier.strip()
            for identifier in identifiers
        ):
            raise RetrieverContractError(
                f"rankings[{name!r}] must contain only non-empty strings"
            )
        if len(set(identifiers)) != len(identifiers):
            raise RetrieverContractError(
                f"rankings[{name!r}] must contain unique identifiers"
            )
        normalized[name] = identifiers
    return normalized


def _validate_results(
    results: Mapping[str, Sequence[RetrievedQueryResult]],
) -> dict[str, dict[str, tuple[str, ...]]]:
    if not isinstance(results, Mapping):
        raise DatasetValidationError("results must be a mapping of source results")
    if len(results) < 2:
        raise DatasetValidationError("fusion requires at least two sources")
    by_query: dict[str, dict[str, tuple[str, ...]]] = {}
    expected_ids: set[str] | None = None
    for name, source_results in results.items():
        if not isinstance(name, str) or not name.strip():
            raise DatasetValidationError("results source names must be non-empty")
        if isinstance(source_results, (str, bytes)) or not isinstance(
            source_results, Sequence
        ):
            raise DatasetValidationError(
                f"results[{name!r}] must be a sequence of RetrievedQueryResult values"
            )
        normalized = tuple(source_results)
        if not all(isinstance(result, RetrievedQueryResult) for result in normalized):
            raise DatasetValidationError(
                f"results[{name!r}] must contain only RetrievedQueryResult values"
            )
        result_ids = [result.query_id for result in normalized]
        duplicate_ids = sorted(
            identifier for identifier, count in Counter(result_ids).items() if count > 1
        )
        if duplicate_ids:
            raise DatasetValidationError(
                f"results[{name!r}] contain duplicate query IDs: {duplicate_ids!r}"
            )
        actual_ids = set(result_ids)
        if expected_ids is None:
            expected_ids = actual_ids
        elif actual_ids != expected_ids:
            missing_ids = sorted(expected_ids - actual_ids)
            unknown_ids = sorted(actual_ids - expected_ids)
            raise DatasetValidationError(
                "fusion sources must cover the same query IDs; "
                f"results[{name!r}] missing={missing_ids!r}, "
                f"unknown={unknown_ids!r}"
            )
        by_query[name] = {
            result.query_id: result.retrieved_document_ids for result in normalized
        }
    return by_query


__all__ = ["evaluate_fused_results", "fuse_rankings", "fuse_results"]
