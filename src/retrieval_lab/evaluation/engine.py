"""Shared ranking evaluation primitives used by every application entry point."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from retrieval_lab.domain import (
    EvaluationQuery,
    JSONValue,
    QueryEvaluation,
    RetrieverMetrics,
)
from retrieval_lab.evaluation.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from retrieval_lab.exceptions import ConfigurationError, EvaluationError

METRIC_NAMES = ("hit_rate", "recall", "precision", "mrr", "ndcg", "ap")


def normalize_top_k(top_k: Sequence[int]) -> tuple[int, ...]:
    """Validate, sort, and freeze metric cutoffs."""

    if isinstance(top_k, (str, bytes)) or not isinstance(top_k, Sequence):
        raise ConfigurationError("top_k must be a sequence of positive integers")
    normalized = tuple(top_k)
    if not normalized:
        raise ConfigurationError("top_k must not be empty")
    if any(
        isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff <= 0
        for cutoff in normalized
    ):
        raise ConfigurationError("top_k values must be positive integers")
    if len(set(normalized)) != len(normalized):
        raise ConfigurationError("top_k values must be unique")
    return tuple(sorted(normalized))


def evaluate_ranking(
    *,
    query_id: str,
    retrieved_ids: Sequence[str],
    relevance_grades: Mapping[str, int],
    top_k: Sequence[int],
    search_latency_ms: float | None = None,
    warnings: Sequence[str] = (),
) -> QueryEvaluation:
    """Evaluate one deterministic ranking with every supported metric."""

    relevant_ids = frozenset(relevance_grades)
    metrics_by_cutoff: dict[int, dict[str, float]] = {}
    try:
        for cutoff in top_k:
            metrics_by_cutoff[cutoff] = {
                "hit_rate": hit_rate_at_k(retrieved_ids, relevant_ids, cutoff),
                "recall": recall_at_k(retrieved_ids, relevant_ids, cutoff),
                "precision": precision_at_k(retrieved_ids, relevant_ids, cutoff),
                "mrr": reciprocal_rank(retrieved_ids, relevant_ids, cutoff),
                "ndcg": ndcg_at_k(retrieved_ids, relevance_grades, cutoff),
                "ap": average_precision_at_k(retrieved_ids, relevant_ids, cutoff),
            }
    except (OverflowError, ValueError) as exc:
        raise EvaluationError(
            f"metrics could not be computed for query {query_id!r}"
        ) from exc
    return QueryEvaluation(
        query_id=query_id,
        retrieved_ids=tuple(retrieved_ids),
        metrics_by_cutoff=metrics_by_cutoff,
        search_latency_ms=search_latency_ms,
        warnings=tuple(warnings),
        retrieved_ids_by_cutoff={
            cutoff: tuple(retrieved_ids[:cutoff]) for cutoff in top_k
        },
    )


def evaluate_cutoff_rankings(
    *,
    query_id: str,
    retrieved_ids: Sequence[str],
    retrieved_ids_by_cutoff: Mapping[int, Sequence[str]],
    relevance_grades: Mapping[str, int],
    top_k: Sequence[int],
    search_latency_ms: float | None = None,
    warnings: Sequence[str] = (),
) -> QueryEvaluation:
    """Evaluate cutoff-specific rankings while retaining one evidence ranking."""

    metrics_by_cutoff: dict[int, dict[str, float]] = {}
    for cutoff in top_k:
        try:
            cutoff_ids = retrieved_ids_by_cutoff[cutoff]
        except KeyError as exc:
            raise EvaluationError(
                f"cutoff ranking is missing for query {query_id!r} at {cutoff}"
            ) from exc
        evaluation = evaluate_ranking(
            query_id=query_id,
            retrieved_ids=cutoff_ids,
            relevance_grades=relevance_grades,
            top_k=(cutoff,),
            search_latency_ms=search_latency_ms,
            warnings=warnings,
        )
        metrics_by_cutoff[cutoff] = dict(evaluation.metrics_by_cutoff[cutoff])
    return QueryEvaluation(
        query_id=query_id,
        retrieved_ids=tuple(retrieved_ids),
        metrics_by_cutoff=metrics_by_cutoff,
        search_latency_ms=search_latency_ms,
        warnings=tuple(warnings),
        retrieved_ids_by_cutoff={
            cutoff: tuple(retrieved_ids_by_cutoff[cutoff]) for cutoff in top_k
        },
    )


def aggregate_metrics(
    evaluations: Sequence[QueryEvaluation],
    top_k: Sequence[int],
) -> RetrieverMetrics:
    """Macro-average query metrics without dropping empty rankings."""

    if not evaluations:
        raise EvaluationError("at least one query evaluation is required")
    metrics_by_cutoff: dict[int, dict[str, float]] = {}
    for cutoff in top_k:
        metrics_by_cutoff[cutoff] = {
            metric_name: sum(
                evaluation.metrics_by_cutoff[cutoff][metric_name]
                for evaluation in evaluations
            )
            / len(evaluations)
            for metric_name in METRIC_NAMES
        }
    return RetrieverMetrics(metrics_by_cutoff=metrics_by_cutoff)


def dataset_payload(
    queries: Sequence[EvaluationQuery],
    relevance_grades_by_query: Mapping[str, Mapping[str, int]],
) -> list[JSONValue]:
    """Return the canonical retrieval-evaluation dataset payload for hashing."""

    payload: list[JSONValue] = []
    for query in sorted(queries, key=lambda item: item.id):
        grades: dict[str, JSONValue] = {
            identifier: relevance_grades_by_query[query.id][identifier]
            for identifier in sorted(relevance_grades_by_query[query.id])
        }
        payload.append(
            {
                "metadata": plain_json(query.metadata),
                "query": query.query,
                "query_id": query.id,
                "relevance_grades": grades,
            }
        )
    return payload


def content_hash(value: JSONValue) -> str:
    """Hash a canonical JSON-compatible value using SHA-256."""

    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (OverflowError, TypeError, ValueError) as exc:
        raise EvaluationError("evaluation inputs could not be hashed") from exc
    return hashlib.sha256(payload).hexdigest()


def plain_json(value: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Copy a validated JSON mapping into mutable built-in containers."""

    result: dict[str, JSONValue] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[key] = plain_json(item)
        elif isinstance(item, list):
            result[key] = [
                plain_json(child) if isinstance(child, Mapping) else child
                for child in item
            ]
        else:
            result[key] = item
    return result


__all__ = [
    "METRIC_NAMES",
    "aggregate_metrics",
    "content_hash",
    "dataset_payload",
    "evaluate_ranking",
    "normalize_top_k",
    "plain_json",
]
